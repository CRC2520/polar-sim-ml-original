"""Shared deterministic task instrumentation; constants come from frozen B0 JSON.

These transitions implement imposed synthetic laws, not empirical validation.
No task reads a random generator, historical implementation, or final seed.
"""
from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path

CONTRACT_PATH = Path(__file__).resolve().parents[1] / 'contracts/frozen_b0/PILOT_CONTRACTS_v1.json'
CONTRACTS = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
PILOTS = {p['id']: p for p in CONTRACTS['pilots']}
CLOCK = CONTRACTS['shared_contract']['clock']
GUARDRAILS = ('unauthorized_executions', 'invalid_concurrent_writes',
              'executed_infeasible_actions', 'physical_state_violations',
              'undeclared_budget_overruns')


class InstrumentViolation(AssertionError):
    """An executed-law violation stops the instrument and retains its audit trace."""
    def __init__(self, event):
        self.event = deepcopy(event)
        super().__init__('instrument_guardrail_violation: ' + repr(event['guardrails']))


def audit_execution(event):
    """Recompute guardrails from start facts, permissions, canonical laws and effects.

    This function deliberately does not trust the event's existing guardrail counters.
    Decision-compute measurements belong to the runner and are inspected when supplied;
    the task's scope record explicitly distinguishes them from physical resource checks.
    """
    counts = {name: 0 for name in GUARDRAILS}
    def violation(name, condition):
        counts[name] += int(bool(condition))
    pilot = event['pilot']
    constants = PILOTS[pilot]['task_contract']['constants']
    before, after, channels = event['state_before'], event['state_after'], event['channels']
    capacity_names = {'P5': ('production_capacity_per_epoch', 'replenishment_capacity_per_epoch'),
                      'P6': ('assay_capacity_per_epoch', 'routine_capacity_per_epoch'),
                      'P7': ('repair_capacity_per_epoch', 'deployment_capacity_per_epoch')}[pilot]
    capacities = dict(zip(('A', 'B'), (constants[x] for x in capacity_names)))
    facts = event.get('execution_facts', {}).get('started_operations', [])
    physical = {'A': 0, 'B': 0}
    semantic = {'A': 0, 'B': 0}
    for fact in facts:
        pole, units = fact.get('channel'), fact.get('units')
        malformed = pole not in physical or type(units) is not int or units <= 0
        violation('physical_state_violations', malformed)
        if malformed:
            continue
        physical[pole] += units
        if not fact['operation'].endswith('_sham'):
            semantic[pole] += units
        permitted = before['permissions']['internal'][pole] and before['permissions']['operation'][pole]
        violation('unauthorized_executions', not permitted)
    for pole in ('A', 'B'):
        ch = channels[pole]
        quantities = [ch['external_requested_dose'], ch['external_admitted_dose'], ch['executed_operation']]
        valid_quantities = all(type(n) is int and n >= 0 for n in quantities)
        violation('physical_state_violations', not valid_quantities)
        if not valid_quantities:
            continue
        req, adm, executed = quantities
        violation('executed_infeasible_actions', executed != semantic[pole])
        violation('executed_infeasible_actions', physical[pole] > adm or adm > req)
        violation('executed_infeasible_actions', bool(physical[pole]) and (not ch['valid_request'] or not ch['internal_permission'] or not ch['external_authorization']))
        violation('undeclared_budget_overruns', physical[pole] > capacities[pole])
        violation('physical_state_violations', ch['nominal_capacity'] != capacities[pole])
        violation('physical_state_violations', ch['admitted_activation'] != req / capacities[pole])
    violation('physical_state_violations', not 0 <= event['completed'] <= event['demand'])
    violation('physical_state_violations', event['completed'] + event['missed'] != event['demand'])
    if pilot == 'P5':
        reserve = after['reserve']
        violation('physical_state_violations', type(reserve) is not int or not 0 <= reserve <= constants['reserve_capacity'])
        violation('executed_infeasible_actions', semantic['A'] > min(before['reserve'], event['demand']))
        violation('physical_state_violations', reserve != before['reserve'] - semantic['A'])
        violation('physical_state_violations', event['completed'] != semantic['A'])
        violation('physical_state_violations', sum(j['correct'] for j in event['jobs']) != event['completed'])
        new_queue = sum(x['units'] for x in after['pending_deliveries'] if x['start_epoch'] == event['epoch'])
        violation('physical_state_violations', new_queue != semantic['B'])
        tokens = channels['B']['environmental_effect']['tokens_consumed']
        violation('physical_state_violations', tokens != physical['B'])
        violation('undeclared_budget_overruns', tokens > constants['replenishment_tokens_per_epoch'])
    elif pilot == 'P6':
        catalogue = constants['routine_catalogue']
        for fact in facts:
            candidate = fact.get('candidate_id') if fact.get('channel') == 'A' else fact.get('routine_id')
            violation('unauthorized_executions', candidate not in catalogue)
        feedback = event['feedback']
        violation('physical_state_violations', len(feedback) != semantic['B'])
        violation('physical_state_violations', sum(x['correct'] for x in feedback) != event['completed'])
        violation('physical_state_violations', any(x['correct'] not in (0, 1) or x['routine_id'] != after['current_routine'] for x in feedback))
        violation('unauthorized_executions', after['current_routine'] not in catalogue)
        changed = after['current_routine'] != before['current_routine']
        selector = event.get('selector_event')
        violation('unauthorized_executions', changed and not before['adoption_permission'])
        violation('physical_state_violations', changed and (not selector or not selector['authorized']))
        violation('physical_state_violations', any(r['report_epoch'] > event['epoch'] for r in after['reports']))
        assays = sum(r['assay_epoch'] == event['epoch'] for r in after['pending_assays'])
        violation('physical_state_violations', assays != semantic['A'])
        tokens = channels['A']['environmental_effect']['assay_tokens_consumed']
        violation('physical_state_violations', tokens != physical['A'])
        violation('undeclared_budget_overruns', tokens > constants['assay_tokens_per_epoch'])
    elif pilot == 'P7':
        prior = {x['instance_id']: x for x in before['instances']}
        later = {x['instance_id']: x for x in after['instances']}
        writes = {}
        for fact in facts:
            target = fact.get('instance_id')
            writes[target] = writes.get(target, 0) + fact['units']
            obj = prior.get(target)
            violation('executed_infeasible_actions', obj is None)
            if obj is None:
                continue
            violation('executed_infeasible_actions', obj['locked'])
            if fact['channel'] == 'A':
                violation('executed_infeasible_actions', obj['drift'] != 1)
                violation('physical_state_violations', later[target]['target_version'] != obj['target_version'])
                expected_drift = obj['drift'] if fact['operation'].endswith('_sham') else 0
                violation('physical_state_violations', later[target]['drift'] != expected_drift)
            elif fact['channel'] == 'B':
                version = fact.get('target_version')
                violation('unauthorized_executions', type(version) is not int or version not in before['approved_versions'])
                violation('executed_infeasible_actions', version == obj['target_version'])
                violation('executed_infeasible_actions', before['context']['mode'] == 'state_preserving' and obj['drift'] != 0)
                violation('physical_state_violations', later[target]['target_version'] != obj['target_version'])
        counts['invalid_concurrent_writes'] += sum(max(0, n - 1) for n in writes.values())
        for obj in after['instances']:
            violation('physical_state_violations', obj['target_version'] not in constants['target_versions'] or obj['drift'] not in constants['drift_values'])
            violation('physical_state_violations', obj['current_version'] != obj['target_version'])
        violation('physical_state_violations', sum(j['correct'] for j in event['jobs']) != event['completed'])
        for job in event['jobs']:
            obj = prior[job['instance_id']]
            locked = obj['locked'] or job['instance_id'] in writes
            expected = int(not locked and obj['drift'] == 0 and obj['target_version'] == before['required_version'])
            violation('executed_infeasible_actions', job['correct'] != expected or job['locked'] != locked)
        new_pending = [p for p in after['pending_deployments'] if p['start_epoch'] == event['epoch']]
        violation('physical_state_violations', len(new_pending) != semantic['B'])
    compute = event.get('decision_compute')
    if compute is not None:
        violation('undeclared_budget_overruns', compute['used'] > compute['allowance'])
    return counts


def admit(request, capacity, permission=True):
    """Validate, then floor a fixed-capacity request; never use its outcome."""
    valid = isinstance(request, Real) and not isinstance(request, bool)
    valid = valid and 0 <= request <= 1 and math.isfinite(request)
    recorded = request if valid else repr(request)
    slots = math.floor(capacity * request) if valid and permission else 0
    reasons = [] if valid else ['invalid_request']
    if valid and not permission:
        reasons.append('internal_permission_absent')
    return {'requested_activation': recorded, 'admitted_activation': slots / capacity,
            'nominal_capacity': capacity, 'external_requested_dose': slots,
            'external_admitted_dose': 0, 'executed_operation': 0,
            'continuing_operation': 0, 'environmental_effect': {},
            'reasons': reasons, 'valid_request': valid,
            'internal_permission': bool(permission), 'external_authorization': True}


class TaskBase:
    pilot = None

    def __init__(self, cell_id=0, internal_permissions=None, operation_permissions=None):
        self.cell_id = cell_id
        self.epoch = 0
        self.horizon = CLOCK['episode_horizon']
        self.demand = CLOCK['per_cell_service_demand']
        self.internal_permissions = {'A': True, 'B': True}
        self.operation_permissions = {'A': True, 'B': True}
        if internal_permissions is not None:
            self.internal_permissions.update(internal_permissions)
        if operation_permissions is not None:
            self.operation_permissions.update(operation_permissions)
        self.events = []
        self.completed_jobs = 0
        self.missed_jobs = 0
        self._prepared = False

    @property
    def done(self):
        return self.epoch >= self.horizon

    def _prepare_boundary(self):
        pass

    def _state(self):
        raise NotImplementedError

    def observe(self):
        if self.done:
            raise RuntimeError('episode_complete')
        if not self._prepared:
            self._prepare_boundary()
            self._prepared = True
        return deepcopy(self._state())

    def _base_observation(self, context):
        return {'pilot': self.pilot, 'cell_id': self.cell_id, 'epoch': self.epoch,
                'horizon': self.horizon, 'demand': self.demand,
                'context': context,
                'permissions': {'internal': dict(self.internal_permissions),
                                'operation': dict(self.operation_permissions)}}

    def _channels(self, action, capacities):
        if not isinstance(action, dict):
            raise TypeError('action_must_be_dictionary')
        return {pole: admit(action.get(pole, 0), capacities[pole],
                            self.internal_permissions[pole]) for pole in ('A', 'B')}

    def _authorize(self, pole, channel):
        allowed = bool(self.operation_permissions[pole])
        channel['external_authorization'] = allowed
        if channel['external_requested_dose'] and not allowed:
            channel['reasons'].append('unauthorized_request')
        return allowed

    def _finish(self, before, action, channels, completed, **details):
        bad = any(not c['valid_request'] or 'unauthorized_request' in c['reasons']
                  or 'invalid_target' in c['reasons'] for c in channels.values())
        event = {'pilot': self.pilot, 'cell_id': self.cell_id, 'epoch': self.epoch,
                 'proposal': deepcopy(action), 'state_before': before,
                 'channels': channels, 'demand': self.demand,
                 'completed': completed, 'missed': self.demand - completed,
                 'controller_failure': bad,
                 'state_after': deepcopy(self._state()),
                 'guardrail_scope': {'physical_resources_and_task_laws': 'inspected',
                     'authorization_and_concurrency': 'inspected',
                     'decision_compute': 'runner_measurement_required'},
                 'development_only': True, 'confirmatory': False,
                 'reusable_as_final': False, **details}
        # Keep invalid proposals serializable without pretending they were valid doses.
        for pole in ('A', 'B'):
            if not channels[pole]['valid_request']:
                event['proposal'][pole] = channels[pole]['requested_activation']
        event['guardrails'] = audit_execution(event)
        try:
            self._assert_physical()
        except AssertionError as exc:
            event['guardrails']['physical_state_violations'] += 1
            event['physical_assertion'] = str(exc)
        if any(event['guardrails'].values()):
            raise InstrumentViolation(event)
        self.completed_jobs += completed
        self.missed_jobs += self.demand - completed
        self.events.append(event)
        self.epoch += 1
        self._prepared = False
        return deepcopy(event)

    def _assert_physical(self):
        raise NotImplementedError
