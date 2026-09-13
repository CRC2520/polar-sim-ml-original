"""P7: positive current-target repairs and distinct approved-version deployments."""
from copy import deepcopy
from .core import PILOTS, TaskBase

C = PILOTS['P7']['task_contract']['constants']


class P7Task(TaskBase):
    pilot = 'P7'

    def __init__(self, initial_drifts=(0, 0), mode='state_preserving', version_switch_epoch=None,
                 drift_events=None, cell_id=0, *, diagnostic_sham=False,
                 internal_permissions=None, operation_permissions=None, approved_versions=None):
        super().__init__(cell_id, internal_permissions, operation_permissions)
        if len(initial_drifts) != C['instances_per_cell'] or any(type(d) is not int or d not in C['drift_values'] for d in initial_drifts):
            raise ValueError('invalid_initial_drifts')
        if mode not in ('state_preserving', 'replacement'):
            raise ValueError('invalid_migration_mode')
        if version_switch_epoch is not None and (type(version_switch_epoch) is not int or not 0 <= version_switch_epoch < self.horizon):
            raise ValueError('invalid_version_switch_epoch')
        self.instances = [{'id': i, 'instance_id': i, 'target_version': C['target_versions'][0],
                           'current_version': C['target_versions'][0], 'drift': d, 'locked': False}
                          for i, d in enumerate(initial_drifts)]
        self.mode = mode
        self.required_version = C['target_versions'][0]
        self._version_switch_epoch = version_switch_epoch
        self._drift_events = dict(drift_events or {})
        for epoch, target in self._drift_events.items():
            if type(epoch) is not int or not 0 <= epoch < self.horizon or (target is not None and not self._valid_target(target)):
                raise ValueError('invalid_drift_event')
        self.approved_versions = list(C['target_versions'] if approved_versions is None else approved_versions)
        if any(type(v) is not int or v not in C['target_versions'] for v in self.approved_versions):
            raise ValueError('invalid_approved_version')
        self._pending_deployments = []
        self.completion_events = []
        self._previous_repair_completions = []
        self.rollback_events = []
        self.snapshot_history = []
        self.diagnostic_sham = diagnostic_sham
        self._boundary_deployments = []

    def _valid_target(self, target):
        return type(target) is int and 0 <= target < C['instances_per_cell']

    def _prepare_boundary(self):
        self.completion_events = deepcopy(self._previous_repair_completions)
        self._previous_repair_completions = []
        due = [p for p in self._pending_deployments if p['due_epoch'] == self.epoch]
        self._pending_deployments = [p for p in self._pending_deployments if p['due_epoch'] != self.epoch]
        self._boundary_deployments = []
        for p in due:
            obj = self.instances[p['instance_id']]
            obj.update(target_version=p['target_version'], current_version=p['target_version'], drift=0, locked=False)
            event = {'operation': 'deployment', 'source_cell_id': self.cell_id,
                     'instance_id': p['instance_id'], 'target_version': p['target_version'],
                     'post_deployment_drift': 0, 'start_epoch': p['start_epoch'],
                     'completion_epoch': self.epoch, 'completion_phase': 'start', 'snapshot': deepcopy(p['snapshot'])}
            self.completion_events.append(event)
            self._boundary_deployments.append(event)
        if self.epoch == self._version_switch_epoch:
            self.required_version = C['target_versions'][1]
        if self.epoch in self._drift_events and self._drift_events[self.epoch] is not None:
            self.instances[self._drift_events[self.epoch]]['drift'] = 1

    def _state(self):
        out = self._base_observation({'mode': self.mode})
        out.update(instances=deepcopy(self.instances), required_version=self.required_version,
                   approved_versions=list(self.approved_versions), completion_events=deepcopy(self.completion_events),
                   pending_deployments=deepcopy(self._pending_deployments),
                   rollback_events=deepcopy(self.rollback_events), snapshot_history=deepcopy(self.snapshot_history))
        return out

    def step(self, action):
        before = self.observe()
        ch = self._channels(action, {'A': C['repair_capacity_per_epoch'], 'B': C['deployment_capacity_per_epoch']})
        repair_target, deploy_target = action.get('repair_target'), action.get('deploy_target')
        candidate = action.get('target_version', self.required_version)
        qa, qb = ch['A']['external_requested_dose'], ch['B']['external_requested_dose']
        repair_obj = deploy_obj = None
        authorized_a, authorized_b = self._authorize('A', ch['A']), self._authorize('B', ch['B'])
        if qa:
            if not self._valid_target(repair_target):
                ch['A']['reasons'].append('invalid_target')
            elif authorized_a:
                obj = self.instances[repair_target]
                if obj['locked']:
                    ch['A']['reasons'].append('write_conflict')
                elif obj['drift'] == 0:
                    ch['A']['reasons'].append('healthy_target_noop')
                else:
                    repair_obj = obj
                    obj['locked'] = True
        if qb:
            if not self._valid_target(deploy_target):
                ch['B']['reasons'].append('invalid_target')
            elif authorized_b:
                obj = self.instances[deploy_target]
                if qa and self._valid_target(repair_target) and repair_target == deploy_target:
                    ch['B']['reasons'].append('write_conflict')
                elif obj['locked']:
                    ch['B']['reasons'].append('write_conflict')
                elif type(candidate) is not int or candidate not in self.approved_versions:
                    ch['B']['reasons'].append('unauthorized_request')
                elif candidate == obj['target_version']:
                    ch['B']['reasons'].append('same_target_noop')
                elif self.mode == 'state_preserving' and obj['drift'] != 0:
                    ch['B']['reasons'].append('source_drift_precondition')
                else:
                    deploy_obj = obj
                    snapshot = {'instance_id': obj['instance_id'], 'target_version': obj['target_version'],
                                'current_version': obj['current_version'], 'drift': obj['drift']}
                    self.snapshot_history.append(deepcopy(snapshot))
                    obj['locked'] = True
                    self._pending_deployments.append({'instance_id': deploy_target,
                        'target_version': candidate, 'start_epoch': self.epoch,
                        'due_epoch': self.epoch + 1, 'snapshot': snapshot})
        jobs = [{'job_id': f'{self.cell_id}:{self.epoch}:{obj["instance_id"]}',
                 'instance_id': obj['instance_id'], 'locked': obj['locked'],
                 'correct': int(not obj['locked'] and obj['drift'] == 0 and obj['target_version'] == self.required_version)}
                for obj in self.instances]
        completed = sum(job['correct'] for job in jobs)
        repair_completion = None
        if repair_obj is not None:
            original_version = repair_obj['target_version']
            if not self.diagnostic_sham:
                repair_obj['drift'] = 0
            repair_obj['locked'] = False
            repair_completion = {'operation': 'repair_sham' if self.diagnostic_sham else 'repair',
                'source_cell_id': self.cell_id, 'instance_id': repair_obj['instance_id'],
                'target_version': repair_obj['target_version'], 'post_repair_drift': repair_obj['drift'],
                'completion_epoch': self.epoch, 'completion_phase': 'end'}
            if repair_obj['target_version'] != original_version:
                raise AssertionError('repair_changed_target_version')
            self._previous_repair_completions.append(repair_completion)
        for pole, obj in [('A', repair_obj), ('B', deploy_obj)]:
            ch[pole]['external_admitted_dose'] = int(obj is not None)
            ch[pole]['executed_operation'] = int(obj is not None)
        if self.diagnostic_sham and repair_obj is not None:
            ch['A']['executed_operation'] = 0
        ch['A']['environmental_effect'] = {'repair_completion': repair_completion,
            'target_version_changed': False, 'sham': bool(self.diagnostic_sham),
            'sham_locked_operations': int(self.diagnostic_sham and repair_obj is not None)}
        ch['B']['continuing_operation'] = len(self._boundary_deployments)
        ch['B']['environmental_effect'] = {'deployment_completions': deepcopy(self._boundary_deployments),
            'rollback_events': [], 'snapshot_created': deploy_obj is not None}
        starts = []
        if repair_obj is not None:
            starts.append({'channel': 'A', 'operation': 'repair_sham' if self.diagnostic_sham else 'repair',
                           'units': 1, 'instance_id': repair_target})
        if deploy_obj is not None:
            starts.append({'channel': 'B', 'operation': 'deployment', 'units': 1,
                           'instance_id': deploy_target, 'target_version': candidate})
        return self._finish(before, action, ch, completed, jobs=jobs,
                            execution_facts={'started_operations': starts},
                            intervention='source_sham' if self.diagnostic_sham else 'none')

    def _assert_physical(self):
        for obj in self.instances:
            if obj['target_version'] not in C['target_versions'] or obj['drift'] not in C['drift_values']:
                raise AssertionError('physical_state_violation: invalid_version_or_drift')
            if obj['current_version'] != obj['target_version']:
                raise AssertionError('physical_state_violation: target_current_mismatch')
        pending_ids = [p['instance_id'] for p in self._pending_deployments]
        if len(set(pending_ids)) != len(pending_ids):
            raise AssertionError('physical_state_violation: concurrent_writes')
