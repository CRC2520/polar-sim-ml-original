"""Dimension-based accounting, independent of scores and wall-clock timing.

The expanded charge is a reproducible conservative logical-work proxy, not an
instruction counter. It charges declared full-buffer passes as well as the
legacy controller's primitive counter. The certificate does not equate a scalar
with a byte or claim a hard operating-system scheduling deadline.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from b1.controllers.core import Limits, scalar_count
from b1.tasks import P5Task, P6Task, P7Task
from b1.tasks.core import CLOCK, PILOTS

COMMON = dict(memory_scalars=16384, decision_operations=65536,
              planning_horizon=1, latency_seconds=None)
LOGICAL_ALLOWANCE = 262144
WORKSPACE_ALLOWANCE = 131072


def limits():
    return Limits(**COMMON)


def maximum_observations(pilot):
    """Schema envelope; simultaneous maxima may be unreachable, hence conservative.

    At most one P6 report and selector row, and one P7 snapshot, can be added
    each of 32 epochs. One-step queues hold at most one operation of each type.
    P7 completion lists contain at most one repair and one deployment. Bounds
    include terminal record sizes although act sees only epochs 0 through 31.
    """
    h = CLOCK['episode_horizon']
    rows = []
    for cell in range(CLOCK['replicated_cells']):
        if pilot == 'P5':
            o = P5Task(8, cell_id=cell).observe()
            o['pending_deliveries'] = [dict(source_cell_id=cell, start_epoch=30,
                                          due_epoch=32, units=2)]
        elif pilot == 'P6':
            o = P6Task(cell_id=cell).observe()
            r = dict(source_cell_id=cell, candidate_id='q1', report_value=1,
                     transfer_eligible=True, assay_epoch=0, report_epoch=1,
                     valid=True, noninformative_sham=False,
                     report_age_epochs=30, assay_age_epochs=31)
            o['reports'] = [deepcopy(r) for _ in range(h)]
            o['arriving_reports'] = [deepcopy(r)]
            o['feedback'] = [dict(job_id=f'{cell}:30:{i}', routine_id='q0',
                                  correct=0, epoch=30) for i in range(2)]
            o['selector_history'] = [dict(epoch=i, old_routine_id='q0',
                                         requested_routine_id='q1', authorized=True,
                                         provenance='controller_policy', changed=True)
                                     for i in range(h)]
            o['pending_assays'] = [dict(source_cell_id=cell, candidate_id='q1',
                                        assay_epoch=31, report_epoch=32)]
        else:
            o = P7Task(cell_id=cell).observe()
            snap = dict(instance_id=0, target_version=0, current_version=0, drift=0)
            o['completion_events'] = [
                dict(operation='repair', source_cell_id=cell, instance_id=0,
                     target_version=0, post_repair_drift=0, completion_epoch=30,
                     completion_phase='end'),
                dict(operation='deployment', source_cell_id=cell, instance_id=1,
                     target_version=1, post_deployment_drift=0, start_epoch=30,
                     completion_epoch=31, completion_phase='start', snapshot=deepcopy(snap))]
            o['pending_deployments'] = [dict(instance_id=0, target_version=1,
                                           start_epoch=31, due_epoch=32, snapshot=deepcopy(snap))]
            o['snapshot_history'] = [deepcopy(snap) for _ in range(h)]
            o['rollback_events'] = []  # Frozen transition never appends a rollback.
        o['epoch'] = 31
        rows.append(o)
    return rows


def static_bounds(pilot, larger=False):
    raw = scalar_count(maximum_observations(pilot))
    n = CLOCK['replicated_cells']
    h = CLOCK['episode_horizon']
    payload = {'P5': 6, 'P6': 10, 'P7': 10}[pilot]
    route = {'P5': 2, 'P6': 3, 'P7': 3}[pilot]
    # P6 includes mutually exclusive feedback, typed-route, and full-history
    # branches simultaneously; sorting charged R*bit_length(R), as in source.
    decide = {'P5': 16, 'P6': 4 + 3 + 4 + 1 + h + h*h.bit_length() + 4,
              'P7': 4 + 2*(7+3) + 5}[pilot]
    action = {'P5': 12, 'P6': 30, 'P7': 30}[pilot]
    slots = 6 if larger else 3
    retention = 2 if larger else 1
    # Fixed configuration/parameters have fewer than 64 counted scalar/key slots.
    parameter = 64
    memory = 2 + action + retention*raw + parameter
    legacy = raw + 2*n*route + slots*payload + n*(decide+3) + action
    # The standalone P5 conventional adapter is bounded by the same envelope.
    # Includes old/new memory buffer passes, route transfers, and bookkeeping.
    workspace = 12*raw + 8*memory + 16*slots*payload + 512
    logical = legacy + 12*raw + 4*memory + 16*slots*payload + 32*h*n + 512
    return dict(raw_observation_scalars=raw, parameter_scalars=parameter,
                persistent_budgeted_scalars=memory, legacy_operations=legacy,
                temporary_workspace_scalar_proxy=workspace,
                expanded_logical_operations=logical,
                total_controller_state_scalar_bound=memory+2048,
                route_payload_scalars=payload, route_slots=slots)


def decision_charge(observations, controller):
    """Read-only post-decision accounting; cannot modify policy input or output."""
    raw = scalar_count(observations)
    mem = scalar_count(controller.memory) + scalar_count(controller.parameters)
    report = controller.last_decision_report
    payload = sum(scalar_count(s['payload']) for s in report.get('route_slots', []))
    legacy = report['useful_operations']
    candidates = sum(len(o.get('reports', [])) for o in observations)
    logical = legacy + 12*raw + 4*mem + 16*payload + 32*candidates + 512
    workspace = 12*raw + 8*mem + 16*payload + 512
    return dict(raw_observation_scalars=raw, persistent_budgeted_scalars=mem,
                legacy_operations=legacy, temporary_workspace_scalar_proxy=workspace,
                expanded_logical_operations=logical,
                total_controller_state_scalars=scalar_count(controller.__dict__))


def assert_within_envelope(pilot, comparator, measured):
    multiplier = 2 if comparator == 'C5' else 1
    caps = dict(persistent_budgeted_scalars=COMMON['memory_scalars']*multiplier,
                legacy_operations=COMMON['decision_operations']*multiplier,
                expanded_logical_operations=LOGICAL_ALLOWANCE*multiplier,
                temporary_workspace_scalar_proxy=WORKSPACE_ALLOWANCE*multiplier,
                total_controller_state_scalars=16384*multiplier)
    if any(measured[k] >= v for k, v in caps.items()):
        raise AssertionError('resource_cap_binding')
    bound = static_bounds(pilot, comparator == 'C5')
    for key in ('raw_observation_scalars', 'persistent_budgeted_scalars',
                'legacy_operations', 'expanded_logical_operations',
                'temporary_workspace_scalar_proxy'):
        if measured[key] > bound[key]:
            raise AssertionError('static_bound_exceeded:'+key)
    if measured['total_controller_state_scalars'] > bound['total_controller_state_scalar_bound']:
        raise AssertionError('static_bound_exceeded:total_controller_state_scalars')
    return True


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ['controllers/core.py', 'controllers/c6.py', 'tasks/core.py',
             'tasks/p5.py', 'tasks/p6.py', 'tasks/p7.py', 'SELECTED_CONFIG.json',
             'contracts/frozen_b0/PILOT_CONTRACTS_v1.json']
    return {f'b1/{p}': sha256((root/p).read_bytes()).hexdigest() for p in paths}


def specification():
    bounds = {p: {c: static_bounds(p, c == 'C5') for c in ('C0','C1','C2','C3','C4','C5')}
              for p in ('P5','P6','P7')}
    return dict(version='B1-D-v1.1-resource-spec-1', development_only=True,
                confirmatory=False, reusable_as_final=False,
                classification='resource_envelope_only_and_read_only_instrumentation',
                namespace='PD-B1-D-v1.1-QA',
                common_allowed_resource_envelope={**COMMON,
                    'expanded_logical_operations': LOGICAL_ALLOWANCE,
                    'temporary_workspace_scalar_proxy': WORKSPACE_ALLOWANCE,
                    'model_state_scalar_bound': 16384,
                    'communication_slots': 3, 'max_payload_scalar_slots': 10,
                    'hyperparameter_slots': 8, 'learned_parameter_updates': 0},
                C5={'allowance_multiplier':2,'communication_slots':6,
                    'claim_limit':'descriptive_capacity_frontier_not_resource_matched'},
                C4={'fixed_analytical_controller':True,'requested_search_slots':1,
                    'search_opportunity_slots':8,'padding_forbidden':True,
                    'planning_horizon':1,'raw_observations':'complete three cells'},
                old_caps={'memory_scalars':4096,'decision_operations':8192},
                cap_rationale='powers-of-two conservative headroom above finite source/schema bounds, never scores; no smallest-cap claim',
                dimensions={'cells':3,'epochs':32,'demand_per_cell':2,
                    'P5_pending_deliveries':1,'P6_reports':32,'P6_selectors':32,
                    'P6_feedback':2,'P6_arriving_reports':1,'P6_pending_assays':1,
                    'P6_candidates':2,'P7_instances':2,'P7_completion_events':2,
                    'P7_pending_deployments':1,'P7_snapshots':32,'P7_rollbacks':0},
                deterministic_accounting={
                    'unit':'declared logical primitive/scalar-pass proxy; not CPU instructions or exact Python opcode count',
                    'legacy':'frozen _tick counter: scalar copies, record inspections, route payloads, policy primitives and sorting proxy',
                    'expanded':'legacy + 12*raw_scalars + 4*budgeted_state_scalars + 16*route_payload_scalars + 32*report_candidates + 512',
                    'workspace':'12*raw_scalars + 8*budgeted_state_scalars + 16*route_payload_scalars + 512; conservative scalar-pass workspace proxy, not measured bytes',
                    'model':'finite rules and configuration; no neural matrices, planning tree, learned weights or optimization updates',
                    'physical_memory':'measured independently in bytes with tracemalloc and process RSS; no scalar-to-byte equivalence assumed'},
                bounds=bounds, source_hashes=source_hashes(),
                P7_policy='unchanged v1 SELECTED_CONFIG.json; C1 cfg01 reactive; no tuning, no new scientific sample',
                measurement_policy='same serial process class; actual CPU, wall latency, traced allocation and RSS are secondary sampled host metrics, no forced equal use or hard real-time deadline',
                host_claim_limit='No claim of exact CPU/memory-byte equality or hard worst-case host latency; logical matched-opportunity claims only.',
                admissible_input_scope='frozen finite task-generated schemas and canonical controller identifiers; arbitrary malformed observations outside scope fail validation',
                failure_policy='bound breach or binding cap blocks certification; change envelope prospectively and repeat affected development only')


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1]/'RESOURCE_MATCHING_SPEC.json'
    out.write_text(json.dumps(specification(), indent=2, sort_keys=True)+'\n')
    print(out)
