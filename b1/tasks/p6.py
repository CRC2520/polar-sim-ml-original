"""P6: isolated assays, delayed provenance records, and authorized routine selection."""
from copy import deepcopy
from .core import PILOTS, TaskBase

C = PILOTS['P6']['task_contract']['constants']


class P6Task(TaskBase):
    pilot = 'P6'

    def __init__(self, initial_mapping='q0', flip_epoch=None, transfer_eligible=True,
                 cell_id=0, *, diagnostic_sham=False, adoption_permission=True,
                 internal_permissions=None, operation_permissions=None):
        super().__init__(cell_id, internal_permissions, operation_permissions)
        if initial_mapping not in C['routine_catalogue']:
            raise ValueError('invalid_initial_mapping')
        if flip_epoch is not None and (type(flip_epoch) is not int or not 0 <= flip_epoch < self.horizon):
            raise ValueError('invalid_flip_epoch')
        self._mapping = initial_mapping
        self._flip_epoch = flip_epoch
        self.transfer_eligible = bool(transfer_eligible)
        self.current_routine = C['routine_catalogue'][0]
        self._pending_reports = []
        self.reports = []
        self.arriving_reports = []
        self.feedback = []
        self.diagnostic_sham = diagnostic_sham
        self.adoption_permission = adoption_permission
        self.selector_history = []

    def _prepare_boundary(self):
        self.arriving_reports = [r for r in self._pending_reports if r['report_epoch'] == self.epoch]
        self._pending_reports = [r for r in self._pending_reports if r['report_epoch'] != self.epoch]
        self.reports.extend(deepcopy(self.arriving_reports))
        if self.epoch == self._flip_epoch:
            self._mapping = next(q for q in C['routine_catalogue'] if q != self._mapping)

    def _state(self):
        out = self._base_observation({'transfer_eligible': self.transfer_eligible})
        def visible_report(report):
            return {**deepcopy(report), 'report_age_epochs': self.epoch - report['report_epoch'],
                    'assay_age_epochs': self.epoch - report['assay_epoch']}
        out.update(current_routine=self.current_routine, routine_catalogue=list(C['routine_catalogue']),
                   reports=[visible_report(r) for r in self.reports],
                   arriving_reports=[visible_report(r) for r in self.arriving_reports],
                   feedback=deepcopy(self.feedback), adoption_permission=bool(self.adoption_permission),
                   selector_history=deepcopy(self.selector_history),
                   pending_assays=[{key: r[key] for key in ('source_cell_id', 'candidate_id', 'assay_epoch', 'report_epoch')}
                                   for r in self._pending_reports])
        return out

    def step(self, action):
        before = self.observe()
        ch = self._channels(action, {'A': C['assay_capacity_per_epoch'], 'B': C['routine_capacity_per_epoch']})
        selector_event = None
        if 'routine_id' in action:
            q = action['routine_id']
            allowed = q in C['routine_catalogue'] and bool(self.adoption_permission)
            selector_event = {'epoch': self.epoch, 'old_routine_id': self.current_routine,
                'requested_routine_id': q, 'authorized': allowed,
                'provenance': deepcopy(action.get('selector_source', 'controller_policy')),
                'changed': allowed and q != self.current_routine}
            if allowed:
                self.current_routine = q
            elif q != self.current_routine:
                ch['B']['reasons'].append('unauthorized_request')
            self.selector_history.append(selector_event)
        candidate = action.get('candidate_id', C['routine_catalogue'][1])
        qa = ch['A']['external_requested_dose']
        assay = min(qa, C['assay_tokens_per_epoch']) if self._authorize('A', ch['A']) else 0
        if qa and candidate not in C['routine_catalogue']:
            ch['A']['reasons'].append('unauthorized_request')
            assay = 0
        routine = min(ch['B']['external_requested_dose'], self.demand) if self._authorize('B', ch['B']) else 0
        if assay:
            value = int(self.transfer_eligible and not self.diagnostic_sham and candidate == self._mapping)
            self._pending_reports.append({'source_cell_id': self.cell_id, 'candidate_id': candidate,
                'report_value': value, 'transfer_eligible': self.transfer_eligible and not self.diagnostic_sham,
                'assay_epoch': self.epoch, 'report_epoch': self.epoch + C['report_delay_epochs'],
                'valid': True, 'noninformative_sham': bool(self.diagnostic_sham)})
        correct = int(self.current_routine == self._mapping)
        completed = routine * correct
        self.feedback = [{'job_id': f'{self.cell_id}:{self.epoch}:{i}', 'routine_id': self.current_routine,
                          'correct': correct, 'epoch': self.epoch} for i in range(routine)]
        for pole, n in [('A', assay), ('B', routine)]:
            ch[pole]['external_admitted_dose'] = n
            ch[pole]['executed_operation'] = n
        ch['A']['continuing_operation'] = len(self.arriving_reports)
        ch['A']['environmental_effect'] = {'arriving_reports': deepcopy(self.arriving_reports),
            'assay_tokens_consumed': assay, 'live_service_directly_completed': 0}
        ch['B']['environmental_effect'] = {'routine_id': self.current_routine, 'correct_jobs': completed,
                                         'incorrect_jobs': routine - completed}
        starts = []
        if assay:
            starts.append({'channel': 'A', 'operation': 'assay', 'units': assay, 'candidate_id': candidate})
        if routine:
            starts.append({'channel': 'B', 'operation': 'routine', 'units': routine, 'routine_id': self.current_routine})
        return self._finish(before, action, ch, completed, feedback=deepcopy(self.feedback),
                            selector_event=selector_event,
                            execution_facts={'started_operations': starts},
                            intervention='source_sham' if self.diagnostic_sham else 'none')

    def _assert_physical(self):
        if self.current_routine not in C['routine_catalogue']:
            raise AssertionError('physical_state_violation: unauthorized_routine')
        if any(r['report_epoch'] > self.epoch for r in self.reports):
            raise AssertionError('physical_state_violation: premature_report')
