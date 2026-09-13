"""Fixed state-clone source/sham checks of the B0 synthetic transition laws.

Passing these imposed contrasts verifies instrument behavior, not pairing utility.
No random seed is generated or consumed. Route lesions belong to controllers.
"""
from copy import deepcopy
from b1.tasks import PILOTS, P5Task, P6Task, P7Task


def _positive_report_selector(observation):
    eligible = [r for r in observation['arriving_reports']
                if r['transfer_eligible'] and r['report_value'] == 1 and r['valid']]
    eligible.sort(key=lambda r: (r['candidate_id'], r['assay_epoch']))
    if not eligible:
        return {}
    record = eligible[0]
    return {'routine_id': record['candidate_id'],
            'selector_source': {'type': 'arriving_positive_transfer_report',
                                'source_cell_id': record['source_cell_id'],
                                'assay_epoch': record['assay_epoch'],
                                'report_epoch': record['report_epoch']}}


def _diagnostic_arm(pilot, kappa, sham):
    if pilot == 'P5':
        c = PILOTS[pilot]['task_contract']['constants']
        task = P5Task(0 if kappa == 'kappa1' else c['reserve_capacity'], diagnostic_sham=sham)
        task.step({'A': 0, 'B': 1})
        receiver = task.step({'A': 1, 'B': 0})
        value = receiver['completed'] / c['production_capacity_per_epoch']
        delivery_check = receiver['channels']['B']['continuing_operation'] == (0 if sham else c['replenishment_capacity_per_epoch'])
        protocol_ok = task.events[0]['channels']['B']['environmental_effect']['tokens_consumed'] == c['replenishment_tokens_per_epoch'] and delivery_check
    elif pilot == 'P6':
        c = PILOTS[pilot]['task_contract']['constants']
        task = P6Task(initial_mapping=c['routine_catalogue'][1], transfer_eligible=kappa == 'kappa1',
                      diagnostic_sham=sham)
        task.step({'A': 1, 'B': 0, 'candidate_id': c['routine_catalogue'][1]})
        obs = task.observe()
        receiver = task.step({'A': 0, 'B': 1, **_positive_report_selector(obs)})
        value = receiver['completed'] / c['routine_capacity_per_epoch']
        protocol_ok = len(obs['arriving_reports']) == c['assay_capacity_per_epoch'] and task.events[0]['channels']['A']['environmental_effect']['assay_tokens_consumed'] == c['assay_tokens_per_epoch']
    elif pilot == 'P7':
        c = PILOTS[pilot]['task_contract']['constants']
        drifts = [c['drift_values'][0]] * c['instances_per_cell']
        drifts[0] = c['drift_values'][1]
        task = P7Task(initial_drifts=drifts, mode='state_preserving' if kappa == 'kappa1' else 'replacement',
                      diagnostic_sham=sham)
        task.step({'A': 1, 'B': 0, 'repair_target': 0})
        task.step({'A': 0, 'B': 1, 'deploy_target': 0, 'target_version': c['target_versions'][1]})
        obs = task.observe()
        completions = [e for e in obs['completion_events'] if e['operation'] == 'deployment' and e['instance_id'] == 0]
        value = len(completions) / c['deployment_capacity_per_epoch']
        source_job = next(j for j in task.events[0]['jobs'] if j['instance_id'] == 0)
        protocol_ok = source_job['locked'] and source_job['correct'] == 0 and task.events[0]['channels']['A']['external_admitted_dose'] == c['repair_capacity_per_epoch']
    else:
        raise ValueError('unknown_pilot')
    guardrails_ok = all(all(v == 0 for v in event['guardrails'].values()) for event in task.events)
    return {'arm': 'source_sham' if sham else 'source_operation', 'outcome': value,
            'instrument_delivery_valid': bool(protocol_ok and guardrails_ok),
            'trace': deepcopy(task.events)}


def mechanism_diagnostic(pilot):
    contexts = {}
    for kappa in ('kappa1', 'kappa2'):
        active = _diagnostic_arm(pilot, kappa, False)
        sham = _diagnostic_arm(pilot, kappa, True)
        expected = PILOTS[pilot]['gamma']['prediction'][kappa]
        effect = active['outcome'] - sham['outcome']
        contexts[kappa] = {'source': active, 'sham': sham, 'effect': effect,
                           'expected_imposed_effect': expected, 'task_law_pass': effect == expected,
                           'instrument_delivery_valid': active['instrument_delivery_valid'] and sham['instrument_delivery_valid']}
    return {'pilot': pilot, 'experiment_class': 'deterministic_instrument_source_diagnostic',
            'development_only': True, 'confirmatory': False, 'reusable_as_final': False,
            'contexts': contexts,
            'moderation_difference': contexts['kappa1']['effect'] - contexts['kappa2']['effect'],
            'task_law_pass': all(x['task_law_pass'] for x in contexts.values()),
            'mediator_contract_verified': all(x['instrument_delivery_valid'] for x in contexts.values()),
            'joint_manipulation': True,
            'joint_manipulation_reason': 'Source operation changes physical input or information; this is not a Gamma-only organizational route intervention.',
            'pairing_specific_verdict_permitted': False,
            'scope': 'Imposed transition and instrument check only; no independent empirical validation or pairing-necessity inference.'}


def all_mechanism_diagnostics():
    return {pilot: mechanism_diagnostic(pilot) for pilot in ('P5', 'P6', 'P7')}
