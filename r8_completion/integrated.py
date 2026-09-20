"""Integrated R8 agent: native experience, temporal credit, memory and eight pairs.

This is a new bounded realization, not the complete original architecture or a
consciousness claim. All utility targets and viability thresholds are designed.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "collective"))
from bridge_v2.engine import BridgeConfig, BridgeEngine
from p1_completion.ecology import step_physics
from r8_completion.io import atomic_json, atomic_npz, sha256, verify_npz, write_manifest
from r8_completion.learned_network import (LearnedNetwork, JointRidgePredictor,
    operational_tension, to_signed_intensity, from_signed_intensity)

VERSION = "R8.I.1"
PROTOCOL = {
    "version": VERSION,
    "training_blocks": 4, "episodes_per_block": 3, "steps": 400,
    "memory_length": 16, "horizons": [1, 4, 12], "return_horizon": 40,
    "max_fit_samples": 4096, "head_ridge": .1, "readout_ridge": 1.,
    "arbitration_ridge": 64., "ecology_weights": [0., 2., 8.],
    "network_gates": [0., .25, .5, 1.], "energy_threshold": .12,
    "validation_alive_tolerance": .005, "train_epsilon": .15,
    "test_episodes": 1, "credit_shift_probe_events": 5,
    "pilot_seeds": [940151, 940152, 940153],
    "final_seeds": list(range(942001, 942031)),
    "domains": {"id": 5, "ood_delay2": 2, "ood_delay9": 9},
    "objective": "designed viability constraint plus observed resource conservation",
}
VARIANTS = ("full", "noEcology", "noMemory", "noNetwork", "miscredit",
            "coordinate_equivalent", "generic_joint", "generic_flat", "network_fixed", "network_random")


def config(domain="id", training=False):
    return BridgeConfig(agents=8, groups=2, generations=12, warmup=4,
        steps=PROTOCOL["steps"], tail_generations=3, replace_groups=0, mutation=0.,
        restraint_cost=0., initial_restraint_prior=0.,
        regeneration_delay=PROTOCOL["domains"][domain],
        epsilon=PROTOCOL["train_epsilon"] if training else 0.)


def base_features(resource, energy, history, no_memory=False):
    """Sixteen public features; history contains only preceding observations."""
    r, e = np.asarray(resource), np.asarray(energy)
    h = np.asarray(history)
    if no_memory:
        h = np.zeros_like(h)
        h[..., 0] = r[..., None]
        h[..., 1] = e[..., None]
    r1, r4, r12 = h[..., -1, 0], h[..., -4, 0], h[..., -12, 0]
    e1, e4 = h[..., -1, 1], h[..., -4, 1]
    return np.stack((np.ones_like(r), r, e, r*r, e*e, r*e, r1, r4, r12,
        r-r1, (r-r4)/4., e-e1, (e-e4)/4., h[..., -1, 2]/3.,
        h[..., :, 0].mean(-1), h[..., :, 0].std(-1)), -1)


def encode_poles(feat, history, visit_total, no_memory=False):
    """Sixteen operational channels. No pair is forced to sum to one."""
    r, e = feat[..., 1], feat[..., 2]
    h = np.asarray(history)
    if no_memory:
        h = np.zeros_like(h)
        h[..., 0], h[..., 1] = r[..., None], e[..., None]
    income = h[..., -1, 3]
    actions = h[..., :, 2]
    pairs = (
        (e, np.maximum(-feat[..., 12]*12., 0.)),
        (income, np.maximum(-feat[..., 11]*8., 0.)),
        (r, h[..., :, 0].std(-1)*5.),
        (h[..., -1, 2]/3., (actions == 0).mean(-1)),
        (np.maximum(.65-e, 0.)/.65, np.maximum(.65-r, 0.)/.65),
        (1./np.sqrt(1.+visit_total), (np.diff(actions, axis=-1) == 0).mean(-1)),
        (h[..., :, 1].min(-1), np.abs(h[..., -1, 2]-actions.mean(-1))/3.),
        (r/.65, e/.65),
    )
    return np.clip(np.stack([np.stack(pair, -1) for pair in pairs], -2), 0., 1.)


def action_features(x, a):
    x, a = np.asarray(x), np.asarray(a, dtype=int)
    shape = np.broadcast_shapes(x.shape[:-1], a.shape)
    x = np.broadcast_to(x, shape+(16,))
    a = np.broadcast_to(a, shape)
    return (np.eye(4)[a][..., :, None] * x[..., None, :]).reshape(shape+(64,))


class ConsequenceHead:
    def __init__(self):
        self.fitted = False

    def fit(self, features, action, future_public):
        x = action_features(features, action)
        y = np.asarray(future_public).reshape(len(x), 6)
        self.coef = np.linalg.solve(x.T@x + PROTOCOL["head_ridge"]*np.eye(64), x.T@y)
        error = y-x@self.coef
        self.energy_margin = float(np.quantile(np.abs(error[:, 1]), .9))
        self.fitted = True
        return self

    def predict_all(self, features):
        x = np.asarray(features)
        if not self.fitted:
            fallback = np.stack((x[..., 1], x[..., 2]), -1)
            return np.broadcast_to(fallback[..., None, None, :], x.shape[:-1]+(4,3,2)).copy()
        lifted = action_features(x[..., None, :], np.arange(4))
        return (lifted@self.coef).reshape(x.shape[:-1]+(4,3,2))


def default_calibration():
    return {"head_scales": np.ones(3), "global_weights": np.ones(3),
            "state_weights": np.ones((20,3)), "shuffled_state_weights": np.ones((20,3))}


def arbitration(qheads, state_index, calibration):
    from r8_completion.population import arbitrate_q
    return arbitrate_q(qheads, state_index, calibration, mode="state")


class IntegratedAgent:
    def __init__(self, seed):
        self.seed = int(seed)
        self.base = BridgeEngine(config(training=True), [seed], .5, study="C", variant="full")
        self.head, self.shifted_head = ConsequenceHead(), ConsequenceHead()
        self.network = LearnedNetwork(seed=seed)
        self.generic_network = JointRidgePredictor(seed=seed)
        self.readout = {"learned": np.zeros(17), "generic_joint": np.zeros(17)}
        self.calibration = default_calibration()
        self.ecology_weight, self.network_gate = 2., .25
        self.generic_ecology_weight, self.generic_gate = 2., .25
        self.fit_count = 0

    def head_values(self, state_index):
        indices = np.indices(state_index.shape)
        return np.stack([self.base.state[key][0][*indices, state_index] for key in ("qt","qv","qg")], -2)

    def visit_total(self, state_index):
        indices = np.indices(state_index.shape)
        return self.base.state["visits"][0][*indices,state_index].sum(-1)

    def evaluate(self, features, poles, tension, state_index, variant="full",
                 ecology_weight=None, gate=None):
        qheads = self.head_values(state_index)
        qscore = arbitration(qheads, state_index, self.calibration)
        is_generic = variant == "generic_joint"
        network = self.generic_network if is_generic else self.network
        ecological_weight = (self.generic_ecology_weight if is_generic else self.ecology_weight) if ecology_weight is None else ecology_weight
        network_gate = (self.generic_gate if is_generic else self.network_gate) if gate is None else gate
        if variant == "noEcology":
            ecological_weight = 0.
        if variant == "noNetwork":
            network_gate = 0.
        head = self.shifted_head if variant == "miscredit" else self.head
        consequence = head.predict_all(features)
        eco_value = consequence[..., :, :, 0].mean(-1)
        eco_advantage = eco_value-eco_value.mean(-1,keepdims=True)
        # Fitted consequence-based viability guard is shared with noEcology.
        if head.fitted:
            lower_energy = consequence[..., :, 0, 1]-head.energy_margin
            feasible = lower_energy >= PROTOCOL["energy_threshold"]
            fallback = np.eye(4,dtype=bool)[lower_energy.argmax(-1)]
            feasible = np.where(feasible.any(-1)[...,None], feasible, fallback)
        else:
            feasible = np.ones_like(qscore,dtype=bool)
        if network.fitted:
            if variant == "coordinate_equivalent":
                d, intensity = to_signed_intensity(poles)
                poles = from_signed_intensity(d,intensity)
            network_variant = {"generic_flat":"generic_flat","network_fixed":"fixed","network_random":"random"}.get(variant,"learned")
            predicted_poles = network.predict_all(poles,tension,variant=network_variant,gate=network_gate)
            netx = np.concatenate((np.ones(predicted_poles.shape[:-2]+(1,)),predicted_poles.reshape(predicted_poles.shape[:-2]+(16,))),-1)
            predicted_value = netx@self.readout["generic_joint" if is_generic else "learned"]
            net_advantage = predicted_value-predicted_value.mean(-1,keepdims=True)
        else:
            predicted_poles = np.repeat(poles[...,None,:,:],4,axis=-3)
            net_advantage = np.zeros_like(qscore)
        unguarded = qscore+ecological_weight*eco_advantage+4.*net_advantage
        scores = np.where(feasible,unguarded,-1e6)
        return {"scores":scores,"unguarded_scores":unguarded,"qheads":qheads,
                "consequence":consequence,"feasible":feasible,
                "eco_advantage":eco_advantage,"network_advantage":net_advantage,
                "predicted_poles":predicted_poles}


def rollout(agent, episode, domain="id", variant="full", learn=False,
            ecology_weight=None, gate=None, steps=None, probe_rate=None):
    c = config(domain, training=learn)
    if steps is not None:
        c = replace(c,steps=steps)
    if probe_rate is not None:
        c = replace(c,epsilon=float(probe_rate))
    b = BridgeEngine(c,[agent.seed],.5,study="C",variant="full")
    tape = {k:value[0] for k,value in b.random_tape(episode).items()}
    resource = b.state["resource"][0].copy()
    energy = b.state["vitality"][0].copy()
    alive = b.state["alive"][0].copy()
    resource_history = b.state["resource_history"][0].copy()
    capacity = b.capacity
    public_r = np.broadcast_to((resource+tape["obs_noise"][0])[:,None]/capacity,energy.shape)
    history = np.zeros(energy.shape+(PROTOCOL["memory_length"],4))
    history[...,0],history[...,1] = public_r[...,None],(energy/c.max_vitality)[...,None]
    previous_prediction = None
    rows = {k:[] for k in ("features","poles","tension","next_poles","state_index","qheads",
        "action","uniform_probe","alive_before","alive_after","resource_before_DIAGNOSTIC",
        "resource_after_DIAGNOSTIC","energy_before","energy_after","rewards","scores",
        "feasible","eco_advantage","network_advantage","consequence","previous_prediction",
        "successor_action","td_targets","issued_prediction")}
    public_sequence = [np.stack((public_r,energy/c.max_vitality),-1)]
    transition_hash = hashlib.sha256()
    for t in range(c.steps):
        features = base_features(public_r,energy/c.max_vitality,history,variant=="noMemory")
        rb = np.clip((public_r*5).astype(int),0,4)
        eb = np.clip((energy/c.max_vitality*4).astype(int),0,3)
        state_index = eb*5+rb
        poles = encode_poles(features,history,agent.visit_total(state_index),variant=="noMemory")
        tension = operational_tension(poles,poles if previous_prediction is None else previous_prediction,chi=.2)
        policy = agent.evaluate(features,poles,tension,state_index,variant,ecology_weight,gate)
        exploit = (policy["scores"]/c.temperature+tape["gumbel"][t]).argmax(-1)
        probe = tape["explore"][t] & alive
        action = np.where(probe,tape["random_action"][t],exploit)
        action = np.where(alive,action,0)
        before_r,before_e,before_alive = resource.copy(),energy.copy(),alive.copy()
        growth = c.growth_mean+c.growth_amplitude*np.sin(2*np.pi*(t+tape["offset"])/c.growth_period)
        resource,energy,alive,resource_history,take,action = step_physics(resource,energy,alive,resource_history,action,growth,c)
        died = before_alive & ~alive
        delta_v = np.where(before_alive,take-c.metabolism,0.)
        rv = delta_v-c.death_penalty*died
        rg = (rv.sum(-1,keepdims=True)-rv)/(c.agents-1)
        rewards = np.stack((take,rv,rg),-1)
        obs_next = np.broadcast_to((resource+tape["obs_noise"][t+1])[:,None]/capacity,energy.shape)
        history = np.concatenate((history[...,1:,:],np.stack((public_r,before_e/c.max_vitality,action,take/c.max_vitality),-1)[...,None,:]),axis=-2)
        indices = np.indices(action.shape)
        if learn:
            agent.base.state["visits"][0][*indices,state_index,action] += before_alive
        next_features = base_features(obs_next,energy/c.max_vitality,history,variant=="noMemory")
        next_index = np.clip((energy/c.max_vitality*4).astype(int),0,3)*5+np.clip((obs_next*5).astype(int),0,4)
        issued_prediction = policy["predicted_poles"][*indices,action]
        next_poles = encode_poles(next_features,history,agent.visit_total(next_index),variant=="noMemory")
        next_tension = operational_tension(next_poles,issued_prediction,chi=.2)
        successor = np.zeros_like(action)
        td_targets = np.zeros(action.shape+(3,))
        if learn:
            next_policy = agent.evaluate(next_features,next_poles,next_tension,next_index,variant,ecology_weight,gate)
            successor = next_policy["scores"].argmax(-1)
            # Freeze all next-head values before updating any of the three Qs.
            future_heads = next_policy["qheads"].copy()
            for j,(key,gamma) in enumerate(zip(("qt","qv","qg"),(c.gamma_take,c.gamma_vitality,c.gamma_group))):
                q = agent.base.state[key][0]
                old = q[*indices,state_index,action]
                bootstrap = future_heads[...,j,:][*indices,successor]
                target = rewards[...,j]+gamma*bootstrap*alive
                td_targets[...,j] = target
                q[*indices,state_index,action] = old+c.alpha*(target-old)*before_alive
        values = {"features":features,"poles":poles,"tension":tension,"next_poles":next_poles,
            "state_index":state_index,"qheads":policy["qheads"],"action":action,"uniform_probe":probe,
            "alive_before":before_alive,"alive_after":alive.copy(),
            "resource_before_DIAGNOSTIC":before_r,"resource_after_DIAGNOSTIC":resource.copy(),
            "energy_before":before_e,"energy_after":energy.copy(),"rewards":rewards,
            "scores":policy["scores"],"feasible":policy["feasible"],
            "eco_advantage":policy["eco_advantage"],"network_advantage":policy["network_advantage"],
            "consequence":policy["consequence"],
            "previous_prediction":poles if previous_prediction is None else previous_prediction,
            "successor_action":successor,"td_targets":td_targets,"issued_prediction":issued_prediction}
        for key,value in values.items():
            rows[key].append(np.asarray(value).copy())
            transition_hash.update(np.ascontiguousarray(value).tobytes())
        public_sequence.append(np.stack((obs_next,energy/c.max_vitality),-1))
        public_r,previous_prediction = obs_next,issued_prediction.copy()
    rows = {k:np.stack(value) for k,value in rows.items()}
    rows["public_sequence"] = np.stack(public_sequence)
    rows["metadata"] = np.array(json.dumps({"seed":agent.seed,"episode":episode,"domain":domain,"variant":variant,
        "learn":learn,"config":asdict(c),"ecology_weight":ecology_weight,"gate":gate},sort_keys=True))
    rows["transition_sha256"] = np.array(transition_hash.hexdigest())
    summary = {"alive_fraction":float(rows["alive_after"].mean()),
               "resource_fraction":float(rows["resource_after_DIAGNOSTIC"].mean()/capacity),
               "public_resource_fraction":float(np.clip(rows["public_sequence"][1:,...,0],0,1).mean()),
               "action_mean":float(rows["action"].sum()/max(1,rows["alive_before"].sum())),
               "transition_sha256":transition_hash.hexdigest(),
               "guard_restriction_fraction":float((~rows["feasible"]).mean()),
               "guard_violation_fraction":float(np.sum((~np.take_along_axis(rows["feasible"],rows["action"][...,None],-1)[...,0]) & rows["alive_before"])/max(1,rows["alive_before"].sum())),
               "uniform_probe_fraction":float(rows["uniform_probe"].sum()/max(1,rows["alive_before"].sum()))}
    return rows,summary


def trajectory_samples(trace):
    """Aligned factual labels from the same continuous trajectory only."""
    time_count,groups,agents = trace["action"].shape
    usable = time_count-PROTOCOL["return_horizon"]
    if usable < 1:
        raise ValueError("Trajectory shorter than return horizon")
    public = trace["public_sequence"]
    future = np.stack([public[h:h+usable] for h in PROTOCOL["horizons"]],-2)
    future_alive = np.stack([trace["alive_after"][h:h+usable] for h in range(PROTOCOL["return_horizon"])],0).mean(0)
    future_resource = np.stack([np.clip(public[h+1:h+1+usable,...,0],0,1) for h in range(PROTOCOL["return_horizon"])],0).mean(0)
    factual_return = .5*future_alive+.5*future_resource
    action = trace["action"][:usable]
    eligible = trace["uniform_probe"][:usable] & trace["alive_before"][:usable]
    miscredited = action.copy()
    # Preserve action-label multisets within each individual's randomized probe
    # events. Five prior probe events have variable elapsed physical time.
    for g in range(groups):
        for i in range(agents):
            times = np.flatnonzero(eligible[:,g,i])
            if len(times):
                miscredited[times,g,i] = np.roll(action[times,g,i],PROTOCOL["credit_shift_probe_events"])
    data = {k:trace[k][:usable].reshape((-1,)+trace[k].shape[3:]) for k in ("features","poles","tension","next_poles","state_index","qheads")}
    data.update(action=action.ravel(),miscredited_action=miscredited.ravel(),
                future=future.reshape(-1,3,2),factual_return=factual_return.ravel(),
                eligible=eligible.ravel(),alive=trace["alive_before"][:usable].ravel())
    time_index,group_index,individual_index = np.indices((usable,groups,agents))
    episode = json.loads(str(trace["metadata"]))["episode"]
    data.update(time_index=time_index.ravel(),group_index=group_index.ravel(),
                individual_index=individual_index.ravel(),episode_index=np.full(action.size,episode,dtype=int))
    return data


def miscredit_sampled_actions(data):
    """Move only credit labels AFTER subsampling; preserve each multiset exactly."""
    incorrect = data["action"].copy()
    source_time = data["time_index"].copy()
    keys = np.stack((data["episode_index"],data["group_index"],data["individual_index"]),-1)
    for key in np.unique(keys,axis=0):
        ids = np.flatnonzero(np.all(keys==key,axis=1))
        ids = ids[np.argsort(data["time_index"][ids])]
        source_ids = np.roll(ids,PROTOCOL["credit_shift_probe_events"])
        incorrect[ids] = data["action"][source_ids]
        source_time[ids] = data["time_index"][source_ids]
    return incorrect,source_time


def fit_agent(agent,traces):
    from r8_completion.population import fit_arbitration
    batches = [trajectory_samples(t) for t in traces]
    all_data = {key:np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    ids = np.flatnonzero(all_data["eligible"])
    if len(ids) < 80:
        raise ValueError("Insufficient randomized own-action training samples")
    rng = np.random.default_rng([agent.seed,1001,agent.fit_count])
    if len(ids) > PROTOCOL["max_fit_samples"]:
        ids = np.sort(rng.choice(ids,PROTOCOL["max_fit_samples"],replace=False))
    data = {k:v[ids] for k,v in all_data.items()}
    data["miscredited_action"],data["credit_source_time"] = miscredit_sampled_actions(data)
    agent.head.fit(data["features"],data["action"],data["future"])
    agent.shifted_head.fit(data["features"],data["miscredited_action"],data["future"])
    agent.calibration = fit_arbitration(data["qheads"],data["state_index"],data["action"],
        data["factual_return"],np.ones(len(ids),dtype=bool),ridge=PROTOCOL["arbitration_ridge"])
    for name,network in (("learned",agent.network),("generic_joint",agent.generic_network)):
        network.fit(data["poles"],data["tension"],data["action"],data["next_poles"],split="train")
        predicted = network.predict_all(data["poles"],data["tension"],gate=1.)[np.arange(len(ids)),data["action"]].reshape(-1,16)
        x = np.concatenate((np.ones((len(ids),1)),predicted),axis=1)
        agent.readout[name] = np.linalg.solve(x.T@x+PROTOCOL["readout_ridge"]*np.eye(17),x.T@data["factual_return"])
    agent.fit_count += 1
    return data


def select_native_gates(agent):
    records = {}
    for variant in ("full","generic_joint"):
        scores = {}
        for weight in PROTOCOL["ecology_weights"]:
            for gate in PROTOCOL["network_gates"]:
                _,outcome = rollout(agent,500,"id",variant,False,weight,gate)
                scores[(weight,gate)] = outcome
        reference = scores[(0.,0.)]["alive_fraction"]
        feasible = [(key,value) for key,value in scores.items()
                    if value["alive_fraction"] >= reference-PROTOCOL["validation_alive_tolerance"]]
        selected = max(feasible,key=lambda item:(item[1]["public_resource_fraction"],-item[0][0],-item[0][1]))[0]
        if variant == "full":
            agent.ecology_weight,agent.network_gate = selected
            network = agent.network
        else:
            agent.generic_ecology_weight,agent.generic_gate = selected
            network = agent.generic_network
        gate_losses = {gate:min([-value["public_resource_fraction"] for (weight,g),value in feasible if g==gate] or [1e6])
                       for gate in PROTOCOL["network_gates"]}
        network.select_gate(gate_losses,split="development")
        # Lexicographic weight/gate tie-breaking is recorded at the actor level.
        network.gate = selected[1]
        records[variant] = {"reference_alive":reference,"selected":list(selected),
            "candidates":{f"ecology={key[0]},network={key[1]}":value for key,value in scores.items()},
            "selection_uses_test":False,"domain":"id","episode":500}
    return records


def save_agent(agent,path):
    arrays = {"q_"+key:agent.base.state[key] for key in ("qt","qv","qg","visits","lineage")}
    arrays.update(head_coef=agent.head.coef,shifted_head_coef=agent.shifted_head.coef,
        head_energy_margin=np.array(agent.head.energy_margin),
        shifted_energy_margin=np.array(agent.shifted_head.energy_margin),
        readout_learned=agent.readout["learned"],readout_generic=agent.readout["generic_joint"],
        metadata=np.array(json.dumps({"seed":agent.seed,"fit_count":agent.fit_count,
            "ecology_weight":agent.ecology_weight,"network_gate":agent.network_gate,
            "generic_ecology_weight":agent.generic_ecology_weight,"generic_gate":agent.generic_gate},sort_keys=True)))
    arrays["calibration_json"] = np.array(json.dumps(agent.calibration,sort_keys=True))
    # Serialize nested metadata as JSON strings; no object arrays or pickle.
    for prefix,network in (("network_",agent.network),("generic_network_",agent.generic_network)):
        arrays[prefix+"state_json"] = np.array(json.dumps(network.state_dict(),sort_keys=True,
            default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
    return atomic_npz(path,arrays)


def load_agent(path):
    with np.load(path,allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        agent = IntegratedAgent(metadata["seed"])
        for key in ("qt","qv","qg","visits","lineage"):
            agent.base.state[key] = arrays["q_"+key].copy()
        agent.head.coef,agent.shifted_head.coef = arrays["head_coef"].copy(),arrays["shifted_head_coef"].copy()
        agent.head.energy_margin = float(arrays["head_energy_margin"])
        agent.shifted_head.energy_margin = float(arrays["shifted_energy_margin"])
        agent.head.fitted = agent.shifted_head.fitted = True
        agent.readout = {"learned":arrays["readout_learned"].copy(),"generic_joint":arrays["readout_generic"].copy()}
        agent.calibration = json.loads(str(arrays["calibration_json"]))
        agent.network = LearnedNetwork.from_state_dict(json.loads(str(arrays["network_state_json"])))
        agent.generic_network = LearnedNetwork.from_state_dict(json.loads(str(arrays["generic_network_state_json"])))
        for key in ("fit_count","ecology_weight","network_gate","generic_ecology_weight","generic_gate"):
            setattr(agent,key,metadata[key])
    return agent


def predictive_metrics(agent,trace):
    data = trajectory_samples(trace)
    ids = data["eligible"]
    if not ids.any():
        raise ValueError("No held-out randomized probes")
    x,a,y = data["features"][ids],data["action"][ids],data["future"][ids]
    records = {}
    for name,head in (("aligned",agent.head),("miscredit",agent.shifted_head)):
        predicted = head.predict_all(x)[np.arange(len(x)),a]
        error = predicted-y
        records[name] = {"mse":float(np.mean(error**2)),"mse_by_horizon_variable":np.mean(error**2,axis=0).tolist(),
                         "n_randomized_probes":len(x)}
    return records


def run_seed(seed,output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    agent = IntegratedAgent(seed)
    traces,fit_records,training,artifacts = [],[],[],[]
    source_record = {"version":VERSION,"seed":seed,
        "source_sha256":{str(path.relative_to(ROOT)):sha256(path) for path in (
            Path(__file__),ROOT/"r8_completion/population.py",ROOT/"r8_completion/learned_network.py",
            ROOT/"r8_completion/io.py",ROOT/"p1_completion/ecology.py",ROOT/"collective/bridge_v2/engine.py")}}
    artifacts.append(atomic_json(output/"SOURCE.json",source_record))
    for block in range(PROTOCOL["training_blocks"]):
        for within in range(PROTOCOL["episodes_per_block"]):
            episode = block*PROTOCOL["episodes_per_block"]+within
            trace,outcome = rollout(agent,episode,learn=True)
            traces.append(trace)
            training.append(outcome)
            artifacts.append(atomic_npz(output/f"train_{episode:03d}.npz",trace))
        fit_data = fit_agent(agent,traces)
        artifacts.append(atomic_npz(output/f"fitted_samples_block_{block:02d}.npz",fit_data))
        fit_records.append({"block":block,"samples":len(fit_data["action"]),
                            "action_counts":np.bincount(fit_data["action"],minlength=4).tolist(),
                            "head_energy_margin":agent.head.energy_margin,
                            "miscredit_changed_fraction":float(np.mean(fit_data["action"]!=fit_data["miscredited_action"])),
                            "credit_elapsed_steps_quantiles":np.quantile(fit_data["time_index"]-fit_data["credit_source_time"],[0,.25,.5,.75,1]).tolist(),
                            "network_diagnostics":agent.network.fit_diagnostics,
                            "generic_network_diagnostics":agent.generic_network.fit_diagnostics})
    del traces
    selection = select_native_gates(agent)
    artifacts.append(save_agent(agent,output/"frozen_agent.npz"))
    results = {"seed":seed,"training":training,"fit_records":fit_records,"selection":selection,
               "native":{},"prediction":{},"coordinate_action_equal":True,"replay_exact":True}
    for domain in PROTOCOL["domains"]:
        prediction_trace,_ = rollout(agent,650,domain,learn=False,probe_rate=PROTOCOL["train_epsilon"])
        artifacts.append(atomic_npz(output/f"prediction_{domain}.npz",prediction_trace))
        results["prediction"][domain] = predictive_metrics(agent,prediction_trace)
        results["native"][domain] = {}
        full_actions = None
        for variant in VARIANTS:
            trace,outcome = rollout(agent,700,domain,variant,learn=False)
            artifacts.append(atomic_npz(output/f"test_{domain}_{variant}.npz",trace))
            results["native"][domain][variant] = outcome
            if variant == "full":
                full_actions = trace["action"].copy()
            if variant == "coordinate_equivalent":
                results["coordinate_action_equal"] &= bool(np.array_equal(trace["action"],full_actions))
            if variant == "full" and domain == "id":
                restored = load_agent(output/"frozen_agent.npz")
                replay,replay_summary = rollout(restored,700,domain,variant,learn=False)
                results["replay_exact"] = replay_summary == outcome and all(np.array_equal(replay[key],trace[key]) for key in trace)
    artifacts.append(atomic_json(output/"result.json",results))
    sources_unchanged = all(sha256(ROOT/name)==digest for name,digest in source_record["source_sha256"].items())
    if not sources_unchanged:
        raise ValueError("Source changed during seed execution; refusing complete publication")
    write_manifest(output/"COMPLETE.json",artifacts,metadata={"seed":seed,"version":VERSION,
        "all_artifacts_published_atomically":True,"sources_unchanged":sources_unchanged})
    return results


def summarize(results):
    result = {"seed_count":len(results),"replays_exact":all(row["replay_exact"] for row in results),
        "coordinate_actions_equal":all(row["coordinate_action_equal"] for row in results),"prediction":{},"native":{}}
    for domain in PROTOCOL["domains"]:
        result["prediction"][domain] = {name:float(np.mean([row["prediction"][domain][name]["mse"] for row in results])) for name in ("aligned","miscredit")}
        result["native"][domain] = {}
        for variant in VARIANTS:
            result["native"][domain][variant] = {metric:float(np.mean([row["native"][domain][variant][metric] for row in results])) for metric in ("alive_fraction","resource_fraction","action_mean")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("pilot","final"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pilot-count",type=int,default=3)
    parser.add_argument("--seeds",nargs="+",type=int)
    args = parser.parse_args()
    if args.phase == "final":
        from r8_completion.provenance import verify
        frozen = verify()
        if frozen["final_seeds"]["integrated"] != PROTOCOL["final_seeds"]:
            raise ValueError("Final integrated seeds disagree with prospective freeze")
        seeds = PROTOCOL["final_seeds"] if args.seeds is None else args.seeds
        if len(seeds)!=len(set(seeds)) or not set(seeds).issubset(PROTOCOL["final_seeds"]):
            raise ValueError("Final shard must use distinct prospectively frozen seeds")
    else:
        seeds = PROTOCOL["pilot_seeds"][:args.pilot_count] if args.seeds is None else args.seeds
        if len(seeds)!=len(set(seeds)) or not all(940101<=seed<=940199 for seed in seeds):
            raise ValueError("Development seeds must lie in reserved 940101..940199")
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/"RUN.json",{"phase":args.phase,"seeds":seeds,"protocol":PROTOCOL,
        "source_sha256":{str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__))}})
    results,started = [],time.monotonic()
    for seed in seeds:
        results.append(run_seed(seed,args.output/f"seed_{seed}"))
        print(json.dumps({"seed":seed,"completed":len(results),"elapsed_s":time.monotonic()-started}),flush=True)
    atomic_json(args.output/"SUMMARY.json",summarize(results))
    if args.phase == "final":
        from r8_completion.provenance import verify
        atomic_json(args.output/"SOURCE_VERIFIED_AFTER.json",verify())


if __name__ == "__main__":
    main()
