"""P5: distinct current production and one-epoch delayed replenishment."""
from copy import deepcopy
from .core import PILOTS, TaskBase

C = PILOTS['P5']['task_contract']['constants']


class P5Task(TaskBase):
    pilot = 'P5'

    def __init__(self, initial_reserve, cell_id=0, *, diagnostic_sham=False,
                 internal_permissions=None, operation_permissions=None):
        super().__init__(cell_id, internal_permissions, operation_permissions)
        if type(initial_reserve) is not int or not 0 <= initial_reserve <= C['reserve_capacity']:
            raise ValueError('invalid_initial_reserve')
        self.reserve = initial_reserve
        self.initial_reserve = initial_reserve
        self.pending_deliveries = []
        self.diagnostic_sham = diagnostic_sham
        self._boundary = {'due_delivery': 0, 'delivered': 0, 'overflow': 0}

    def _prepare_boundary(self):
        due = sum(x['units'] for x in self.pending_deliveries if x['due_epoch'] == self.epoch)
        self.pending_deliveries = [x for x in self.pending_deliveries if x['due_epoch'] != self.epoch]
        overflow = max(0, self.reserve + due - C['reserve_capacity'])
        self.reserve = min(C['reserve_capacity'], self.reserve + due)
        self._boundary = {'due_delivery': due, 'delivered': due - overflow, 'overflow': overflow}

    def _state(self):
        out = self._base_observation({'initial_reserve': self.initial_reserve})
        out.update(reserve=self.reserve, pending_deliveries=deepcopy(self.pending_deliveries),
                   replenishment_tokens=C['replenishment_tokens_per_epoch'], **self._boundary)
        return out

    def step(self, action):
        before = self.observe()
        ch = self._channels(action, {'A': C['production_capacity_per_epoch'],
                                    'B': C['replenishment_capacity_per_epoch']})
        qa = ch['A']['external_requested_dose']
        qb = ch['B']['external_requested_dose']
        production = min(qa, self.reserve, self.demand) if self._authorize('A', ch['A']) else 0
        refill = min(qb, C['replenishment_tokens_per_epoch']) if self._authorize('B', ch['B']) else 0
        if production < qa and ch['A']['external_authorization']:
            ch['A']['reasons'].append('insufficient_stock_or_demand')
        if refill < qb and ch['B']['external_authorization']:
            ch['B']['reasons'].append('insufficient_tokens')
        self.reserve -= production
        # A diagnostic source sham burns the matched token budget but supplies no reserve.
        physical_refill = 0 if self.diagnostic_sham else refill
        if physical_refill:
            self.pending_deliveries.append({'source_cell_id': self.cell_id, 'start_epoch': self.epoch,
                'due_epoch': self.epoch + C['replenishment_delay_epochs'], 'units': physical_refill})
        for pole, admitted, executed in [('A', production, production), ('B', refill, physical_refill)]:
            ch[pole]['external_admitted_dose'] = admitted
            ch[pole]['executed_operation'] = executed
        ch['A']['environmental_effect'] = {'reserve_withdrawn': production, 'service_completed': production}
        ch['B']['continuing_operation'] = self._boundary['due_delivery']
        ch['B']['environmental_effect'] = {**self._boundary,
            'tokens_consumed': refill, 'enqueued_units': physical_refill,
            'scheduled_delivery_epoch': self.epoch + C['replenishment_delay_epochs'] if physical_refill else None,
            'sham_tokens_dissipated': refill if self.diagnostic_sham else 0}
        jobs = [{'job_id': f'{self.cell_id}:{self.epoch}:{i}', 'correct': int(i < production)}
                for i in range(self.demand)]
        starts = []
        if production:
            starts.append({'channel': 'A', 'operation': 'production', 'units': production})
        if refill:
            starts.append({'channel': 'B', 'operation': 'replenishment_sham' if self.diagnostic_sham else 'replenishment', 'units': refill})
        return self._finish(before, action, ch, production, jobs=jobs,
                            execution_facts={'started_operations': starts},
                            intervention='source_sham' if self.diagnostic_sham else 'none')

    def _assert_physical(self):
        if type(self.reserve) is not int or not 0 <= self.reserve <= C['reserve_capacity']:
            raise AssertionError('physical_state_violation: reserve')
        if any(x['units'] < 0 or x['units'] > C['replenishment_capacity_per_epoch']
               for x in self.pending_deliveries):
            raise AssertionError('physical_state_violation: pending_replenishment')
