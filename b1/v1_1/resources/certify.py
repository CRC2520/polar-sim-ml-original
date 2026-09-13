"""Run read-only controller QA and write the v1.1 resource certificate.

No configuration selection, no scientific evidence, no final seed generation.
Archived P7 bundles are replay inputs, never a new experimental sample.
"""
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import platform
import resource
import time
import tracemalloc

from b1.controllers.core import Controller, Limits, get_configurations
from b1.evaluation.runner import canonical_bytes
from b1.tasks import P5Task, P6Task, P7Task
from b1.v1_1.p5.controllers import P5ConventionalController
from .accounting import (COMMON, LOGICAL_ALLOWANCE, WORKSPACE_ALLOWANCE,
    assert_within_envelope, decision_charge, limits, maximum_observations,
    source_hashes, static_bounds)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'v1_1'
AXES = ['available_information','action_space','restrictions_feasibility',
        'memory_capacity','message_width','planning_horizon','training_data',
        'training_updates','hyperparameter_search_budget','decision_operation_budget',
        'model_state_size','persistent_state','temporary_workspace','latency',
        'CPU_usage','memory_usage']


def controller(pilot, comp, cfg, cycle=1, wide=True, lesion=False):
    cap = limits() if wide else Limits()
    if pilot == 'P5' and comp == 'C4':
        return P5ConventionalController(limits=cap)
    return Controller(pilot, comp, cfg, cycle=cycle, limits=cap, lesion=lesion)


def configs(comp):
    return [x['config_id'] for x in get_configurations(comp)]


def fixtures(pilot):
    rows = []
    for case in range(3):
        cells = []
        for cell in range(3):
            if pilot == 'P5':
                args = dict(initial_reserve=(case*3+cell)%9)
            elif pilot == 'P6':
                args = dict(initial_mapping=f'q{(case+cell)%2}',
                            flip_epoch=(None,1,16)[case],
                            transfer_eligible=(case+cell)%2 == 0)
            else:
                args = dict(initial_drifts=((case+cell)%2,(case+1+cell)%2),
                            mode='state_preserving' if case != 1 else 'replacement',
                            version_switch_epoch=(None,1,16)[case],
                            drift_events={e:(e+cell)%2 for e in range(32)} if case == 2
                                         else {8:cell%2,24:(cell+1)%2})
            cells.append(dict(args,cell_id=cell))
        rows.append(cells)
    return rows


def empty_stats():
    return dict(decisions=0, maximum={}, max_wall_seconds=0., max_CPU_seconds=0.,
                total_wall_seconds=0., total_CPU_seconds=0., timed_decisions=0,
                max_traced_peak_increment_bytes=0, allocation_samples=0,
                old_vs_new_action_and_memory_equal=True, resource_cap_binding=False,
                guardrail_violations=0)


def measure(stats, pilot, comp, observations, c, *, allocation=False):
    if allocation:
        tracemalloc.start()
        start_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
    wall, cpu = time.perf_counter(), time.process_time()
    acts = c.act(observations)
    elapsed, used = time.perf_counter()-wall, time.process_time()-cpu
    if allocation:
        _, peak = tracemalloc.get_traced_memory()
        stats['max_traced_peak_increment_bytes'] = max(stats['max_traced_peak_increment_bytes'], peak-start_mem)
        stats['allocation_samples'] += 1
        tracemalloc.stop()
    else:
        stats['max_wall_seconds'] = max(stats['max_wall_seconds'],elapsed)
        stats['max_CPU_seconds'] = max(stats['max_CPU_seconds'],used)
        stats['total_wall_seconds'] += elapsed
        stats['total_CPU_seconds'] += used
        stats['timed_decisions'] += 1
    measured = decision_charge(observations,c)
    assert_within_envelope(pilot,comp,measured)
    for k,v in measured.items():
        stats['maximum'][k] = max(stats['maximum'].get(k,0),v)
    stats['decisions'] += 1
    return acts


def run_stress():
    all_stats = {}
    fingerprint = sha256()
    for pilot in ('P5','P6','P7'):
        task = {'P5':P5Task,'P6':P6Task,'P7':P7Task}[pilot]
        for comp in ('C0','C1','C2','C3','C4','C5'):
            stats = empty_stats()
            all_stats[pilot+'/'+comp] = stats
            for cfg in configs(comp):
                for cycle in ((1,2) if comp == 'C2' else (1,)):
                    # Dimension envelope tests use legal schemas but may combine
                    # maxima not simultaneously reachable in a real episode.
                    cap, old = controller(pilot,comp,cfg,cycle), controller(pilot,comp,cfg,cycle,False)
                    maxobs = maximum_observations(pilot)
                    for repeat in range(2):
                        acts = measure(stats,pilot,comp,maxobs,cap,allocation=(repeat == 0))
                        assert acts == old.act(maxobs)
                        assert cap.memory == old.memory
                    for case,args in enumerate(fixtures(pilot)):
                        tasks = [task(**x) for x in args]
                        cap, old = controller(pilot,comp,cfg,cycle), controller(pilot,comp,cfg,cycle,False)
                        for epoch in range(32):
                            obs = [t.observe() for t in tasks]
                            acts = measure(stats,pilot,comp,obs,cap,
                                           allocation=cfg == configs(comp)[0] and case == 0 and epoch in (0,15,31))
                            assert acts == old.act(obs)
                            assert cap.memory == old.memory
                            fingerprint.update(canonical_bytes([pilot,comp,cfg,cycle,case,epoch,acts]))
                            for t,a in zip(tasks,acts):
                                event = t.step(a)
                                assert not any(event['guardrails'].values())
    return all_stats,fingerprint.hexdigest()


def replay_p7():
    path = ROOT/'development_results/calibration/bundles.json'
    selected_path = ROOT/'SELECTED_CONFIG.json'
    selected = json.loads(selected_path.read_text())['configurations']['P7']
    bundles = [b for b in json.loads(path.read_text()) if b['pilot'] == 'P7']
    digest = sha256()
    decisions = episodes = 0
    for bundle in bundles:
        for comp in ('C0','C1','C2','C3','C4','C5'):
            for cycle in ((1,2) if comp == 'C2' else (1,)):
                args = deepcopy(bundle['cells'])
                for a in args:
                    a['drift_events'] = {int(k):v for k,v in a['drift_events'].items()}
                tasks = [P7Task(**a) for a in args]
                wide = controller('P7',comp,selected[comp],cycle)
                old = controller('P7',comp,selected[comp],cycle,False)
                for epoch in range(32):
                    obs = [t.observe() for t in tasks]
                    acts = wide.act(obs)
                    assert acts == old.act(obs)
                    assert wide.memory == old.memory
                    assert wide.parameters == old.parameters
                    assert_within_envelope('P7',comp,decision_charge(obs,wide))
                    digest.update(canonical_bytes([bundle['bundle_index'],comp,cycle,epoch,acts]))
                    for task,act in zip(tasks,acts):
                        event = task.step(act)
                        assert not any(event['guardrails'].values())
                    decisions += 1
                episodes += 1
    # Acute route bypass is a separate intervention; all public information,
    # capacities and task feasibility are preserved and no cap is binding.
    acute_checks = 0
    for mode in ('state_preserving','replacement'):
        tasks = [P7Task(initial_drifts=(1,1), mode=mode, version_switch_epoch=16,
                        drift_events={8:0,24:1},cell_id=i) for i in range(3)]
        intact = controller('P7','C1',selected['C1'])
        lesion = controller('P7','C1',selected['C1'],lesion=True)
        for _ in range(32):
            obs = [t.observe() for t in tasks]
            acts = intact.act(obs)
            assert acts == lesion.act(obs)  # Selected reactive realization only.
            for c in (intact,lesion):
                assert c.last_decision_report['raw_information_preserved']
                assert_within_envelope('P7','C1',decision_charge(obs,c))
            for t,a in zip(tasks,acts):
                t.step(a)
            acute_checks += 1
    return dict(status='PASS',purpose='QA replay of archived inputs, not new scientific observations',
                namespace='PD-B1-D-v1.1-QA',source_namespace='PD-B1-D-v1',
                archived_bundle_count=len(bundles),episodes=episodes,decisions=decisions,
                selected_configurations=selected, C1_selected='cfg01',
                action_memory_parameter_equality=True,
                acute_selected_reactive_action_equality_decisions=acute_checks,
                action_digest_sha256=digest.hexdigest(),
                archived_bundle_file_sha256=sha256(path.read_bytes()).hexdigest(),
                selected_config_file_sha256=sha256(selected_path.read_bytes()).hexdigest(),
                scientific_rerun_required=False,development_only=True,
                confirmatory=False,reusable_as_final=False)


def row(pilot,comp,axis,status,allowed,requested,actual,limit=None):
    return dict(pilot=pilot,comparator=comp,axis=axis,status=status,
                allowed=allowed,requested=requested,actual=actual,claim_limit=limit)


def make_rows(stats):
    rows = []
    training_hash = sha256((ROOT/'development_results/tuning/bundles.json').read_bytes()).hexdigest()
    for pilot in ('P5','P6','P7'):
        for comp in ('C0','C1','C2','C3','C4','C5'):
            s = stats[pilot+'/'+comp]
            m = s['maximum']
            big = comp == 'C5'
            factor = 2 if big else 1
            def add(axis,status,allowed,requested,actual,limit=None):
                rows.append(row(pilot,comp,axis,status,allowed,requested,actual,limit))
            same = 'complete public observations of three cells, public context and task histories; no hidden future'
            add('available_information','exact_match',same,same,same)
            actions = {'P5':'A/B normalized requests','P6':'A/B plus authorized routine and assay candidate',
                       'P7':'A/B plus instance IDs and approved target version'}[pilot]
            add('action_space','exact_match',actions,actions,actions)
            add('restrictions_feasibility','exact_match','frozen common task physical/authorization gate',
                'same permissions and physical budgets',{'guardrail_violations':0})
            add('memory_capacity','unmatched_but_quantified' if big else 'common_upper_bound',
                COMMON['memory_scalars']*factor,static_bounds(pilot,big)['persistent_budgeted_scalars'],
                m['persistent_budgeted_scalars'],'C5 is descriptive only' if big else None)
            slots = 6 if big else 3
            used_slots = 0 if pilot == 'P5' and comp == 'C4' else slots
            add('message_width','unmatched_but_quantified' if big else 'lower_actual_usage' if used_slots == 0 else 'exact_match',
                {'slots':slots,'max_payload_scalar_slots':10}, {'slots':used_slots},
                {'slots':used_slots,'payload_scalar_slots':static_bounds(pilot,big)['route_payload_scalars'] if used_slots else 0},
                'C5 wider route is declared, no matched superiority claim' if big else 'C4 may reconstruct from raw information without dummy messages' if used_slots == 0 else None)
            add('planning_horizon','common_upper_bound',1,1 if pilot == 'P5' else 0,
                {'max_implemented_future_epochs':1 if pilot == 'P5' else 0,'declared_horizon_cap':1},
                'P5 next-demand planning may look one epoch ahead; P6/P7 act on current available records. No search tree or truncated conventional plan')
            add('training_data','lower_actual_usage' if comp == 'C4' or pilot == 'P5' else 'exact_match',
                {'v1_tuning_bundle_file_sha256':training_hash,'bundles_per_pilot':32},
                0 if comp == 'C4' or pilot == 'P5' else 32,
                0 if comp == 'C4' or pilot == 'P5' else 32,
                'No new tuning; P5 prospective cfg00 and fixed C4 chosen from contract, not outcomes; original v1 data remain archival')
            add('training_updates','exact_match',0,0,0,'Finite configuration selection is separately recorded, no learned weights')
            add('hyperparameter_search_budget','lower_actual_usage' if comp == 'C4' or pilot == 'P5' else 'common_upper_bound',
                {'configuration_slots':8,'cycle_evaluations_per_slot_per_bundle_max':2},
                {'slots':1 if comp == 'C4' or pilot == 'P5' else 8,'cycles':2 if comp == 'C2' else 1},
                {'v1_slots':0 if pilot == 'P5' else 1 if comp == 'C4' else 8,
                 'v1_1_scientific_tuning_slots':0,'cycles':2 if comp == 'C2' else 1},
                'C2 retains both cycles; C4 one analytical rule. Maximum opportunity matched prospectively; actual total training compute is not claimed equal.')
            add('decision_operation_budget','unmatched_but_quantified' if big else 'common_upper_bound',
                {'legacy_operations':COMMON['decision_operations']*factor,'expanded_logical_operations':LOGICAL_ALLOWANCE*factor},
                {k:static_bounds(pilot,big)[k] for k in ('legacy_operations','expanded_logical_operations')},
                {k:m[k] for k in ('legacy_operations','expanded_logical_operations')},
                'Declared deterministic proxies, not exact CPU instructions; C5 descriptive' if big else 'Declared deterministic proxies, not exact CPU instructions')
            add('model_state_size','unmatched_but_quantified' if big else 'common_upper_bound',16384*factor,
                static_bounds(pilot,big)['total_controller_state_scalar_bound'],m['total_controller_state_scalars'],
                'Scalar/key slots include stored report and configuration; immutable Python class/runtime bytes measured separately')
            add('persistent_state','unmatched_but_quantified' if big else 'common_upper_bound',COMMON['memory_scalars']*factor,
                static_bounds(pilot,big)['persistent_budgeted_scalars'],m['persistent_budgeted_scalars'],
                'Episode reset; C5 retains two raw snapshots; C0-C4 at most one, C4 may use less')
            add('temporary_workspace','unmatched_but_quantified' if big else 'common_upper_bound',WORKSPACE_ALLOWANCE*factor,
                static_bounds(pilot,big)['temporary_workspace_scalar_proxy'],m['temporary_workspace_scalar_proxy'],
                'Conservative declared buffer-pass proxy, not a proven CPython byte allocation bound; allocation samples in memory_usage')
            add('latency','unmatched_but_quantified','no hard decision deadline; same host process class',
                'canonical untruncated algorithm',{'max_wall_seconds':s['max_wall_seconds'],'samples':s['timed_decisions']},
                'No hard real-time or equal-latency claim; wall-clock is secondary and does not truncate policy')
            add('CPU_usage','unmatched_but_quantified','one serial controller invocation on same host class',
                'canonical finite algorithm',{'max_CPU_seconds':s['max_CPU_seconds'],'total_CPU_seconds':s['total_CPU_seconds'],'samples':s['timed_decisions']},
                'Host-specific sampled CPU; no exact actual-compute equality claim')
            add('memory_usage','unmatched_but_quantified','same process class; no per-controller OS limit',
                'canonical finite controller allocation',{'max_traced_peak_increment_bytes':s['max_traced_peak_increment_bytes'],'allocation_samples':s['allocation_samples']},
                'tracemalloc covers Python allocations during act; RSS is shared whole QA process; no per-arm RSS or worst-case CPython-byte equality claim')
    return rows


def main():
    before = source_hashes()
    started = time.perf_counter()
    stats,digest = run_stress()
    replay = replay_p7()
    assert before == source_hashes()
    rows = make_rows(stats)
    assert len(rows) == 3*6*16
    for p in ('P5','P6','P7'):
        for c in ('C0','C1','C2','C3','C4','C5'):
            b = static_bounds(p,c == 'C5')
            assert b['persistent_budgeted_scalars'] < (8192 if c == 'C5' else 4096)
            assert b['legacy_operations'] < (16384 if c == 'C5' else 8192)
    cert = dict(version='B1-D-v1.1-resource-certificate-1',status='CERTIFIED_WITH_EXPLICIT_SCOPE',
                blocker_id='B1-RESOURCE-CERTIFICATE-PENDING',blocker_status='RESOLVED',
                namespace='PD-B1-D-v1.1-QA',development_only=True,confirmatory=False,reusable_as_final=False,
                spec_sha256=sha256((OUT/'RESOURCE_MATCHING_SPEC.json').read_bytes()).hexdigest(),
                source_hashes_before=before,source_hashes_after=source_hashes(),
                frozen_sources_unchanged=True,
                p5_adapter_sha256=sha256((OUT/'p5/controllers.py').read_bytes()).hexdigest(),
                measured=stats,axes=rows,axis_rows=len(rows),required_axes=AXES,
                static_bounds={p:{'C0_C4':static_bounds(p),'C5':static_bounds(p,True)} for p in ('P5','P6','P7')},
                resource_cap_binding=False,old_caps_also_exceed_static_bounds=True,
                old_vs_new_action_and_memory_equal=True,
                stress_action_digest_sha256=digest,
                P7_archived_input_replay=replay,
                invariants_pass={p:{c:c != 'C5' for c in ('C0','C1','C2','C3','C4','C5')} for p in ('P5','P6','P7')},
                invariant_claim_scope='equal maximum declared logical opportunity, task information, actions, physical restrictions and horizon; all actual-use and host-axis differences explicitly quantified',
                source_sham_claim_limit='Existing joint_manipulation=true remains; no selective Gamma verdict from a source/sham contrast',
                acute_route_claim_limit='Selected P7 cfg01 is reactive and bypass action effect remains zero; preserving resources does not establish route dependence',
                host={'python':platform.python_version(),'platform':platform.platform(),
                      'QA_total_wall_seconds':time.perf_counter()-started,
                      'QA_process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      'RSS_scope':'whole shared QA process high-water mark, not per-comparator allocation'},
                limitations=['No smallest-cap or hardware instruction certificate',
                             'No exact actual CPU or byte allocation equality; host metrics secondary',
                             'Finite frozen task schema only; longer horizons/new task schemas require a new bound',
                             'C5 remains descriptive capacity frontier',
                             'P7 null v1 results are unchanged and not relabeled as new science'])
    (OUT/'RESOURCE_MATCHING_CERTIFICATE.json').write_text(json.dumps(cert,indent=2,sort_keys=True)+'\n')
    (OUT/'resources/QA_REPLAY_AUDIT.json').write_text(json.dumps(replay,indent=2,sort_keys=True)+'\n')
    report = ['# Resource matching: B1-D v1.1','',
              'Status: CERTIFIED_WITH_EXPLICIT_SCOPE. The resource blocker is resolved for the frozen finite synthetic controller/task family. This certifies declared maximum logical opportunity, not equal physical CPU use, hardware instructions, or a hard real-time service level.','',
              'All 288 pilot/comparator/axis records separate allowed budget, requested implementation needs, and actual measured usage. C0–C4 receive memory 16,384 scalar slots, 65,536 legacy counter units, 262,144 expanded logical units and 131,072 temporary-workspace proxy units. C5 gets twice the allowance and remains descriptive. No dummy operations or training trials are added to C4.','',
              'Bounds come from three cells, 32 epochs, one-step operation queues, at most 32 report/selector/snapshot records, two service jobs, and two P7 instances. They sum all branches conservatively, including mutually exclusive branches. The worst C0–C4 P6 persistent bound is 3,408 scalar slots and legacy decision bound is 4,119, already below the unchanged v1 caps of 4,096 and 8,192. Caps therefore do not truncate any supported canonical policy, including the full conventional algorithm. The wider v1.1 envelope adds prospective headroom without changing the old files.','',
              'Logical accounting is reproducible and covers primitive dispatch/record/sorting work plus explicitly charged scalar-buffer passes and route/candidate work. Its unit is a documented approximation. Temporary workspace uses a conservative scalar-pass proxy; Python byte allocations are measured independently with tracemalloc. The certificate makes no scalar-to-byte conversion or claim that the proxy is an exact CPython allocation proof.','',
              'P5 conventional C4 uses complete raw observations and performs the minimal-next-demand calculation with no terminal refill. It requests fewer route slots and stores fewer observations; the opportunity envelope is shared. The full-refill witness remains separate. P5 prospective policies receive no outcome-based tuning. P6/P7 retain all v1 selected configurations. C2 evaluates both fixed cycles; its doubled evaluation schedule and C4 analytical exception are explicitly quantified, and no equality of actual total training compute is claimed.','',
              f'QA exercised {sum(s["decisions"] for s in stats.values()):,} metered decisions across every frozen configuration, both C2 cycles, maximum-schema observations and three deterministic stress schedules per pilot. Wider-cap and original-cap controllers produced identical actions and memory. All task guardrails were zero. P7 additionally replayed all {replay["archived_bundle_count"]} archived calibration input bundles: {replay["episodes"]} episodes and {replay["decisions"]} decisions with identical actions, parameters and memory. These are infrastructure QA, not new experimental samples.','',
              'The selected P7 C1 stays cfg01/reactive. Its 64 acute intact/bypass QA decisions remain action-identical across both contexts. No scientific rerun or retuning is required by this resource-envelope change. Existing source/sham joint manipulation remains an attribution limit.','',
              '| Pilot | Maximum logical charge | Maximum budgeted scalar state | Maximum traced act allocation (bytes) |','|---|---:|---:|---:|']
    for pilot in ('P5','P6','P7'):
        ss = [stats[pilot+'/'+c] for c in ('C0','C1','C2','C3','C4')]
        report.append(f'| {pilot} | {max(s["maximum"]["expanded_logical_operations"] for s in ss)} | {max(s["maximum"]["persistent_budgeted_scalars"] for s in ss)} | {max(s["max_traced_peak_increment_bytes"] for s in ss)} |')
    report += ['', 'Latency, CPU and allocation numbers are measured on this QA host and appear per comparator in the JSON. They are not portable hard upper bounds. No deadline truncates decisions. CPU, latency and byte-memory axes are `unmatched_but_quantified`; the decisive invariant status is scoped to maximum logical opportunity with these explicit claim limits. C5 never receives matched-invariant PASS.','',
               'No final seeds or B1-E runs were created. Frozen source hashes are recorded before and after; all match.']
    (OUT/'RESOURCE_MATCHING_REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'status':cert['status'],'axis_rows':len(rows),'QA_decisions':sum(s['decisions'] for s in stats.values()),'P7_replay':replay['decisions'],'seconds':cert['host']['QA_total_wall_seconds']}))


if __name__ == '__main__':
    main()
