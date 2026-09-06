"""Literal adjudication for a PROPOSED experiment; not a simulator or result.

Synthetic unit tests validate branching and boundaries only. All bounds supplied
here must have been independently calculated from the complete registered seeds.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Mapping

PARAMETERS = json.loads(Path(__file__).with_name('protocol_parameters.json').read_text())
VALIDITY_KEYS = (
    'complete_unique_trials', 'complete_transitions', 'finite_traces',
    'frozen_sources_match', 'registration_precedes_results',
    'seed_exposure_audit_passed', 'no_hidden_evaluator_inputs',
    'independent_outcome_recalculation', 'paired_counterfactuals_verified',
    'gate_intervention_actually_executed', 'bounds_computed_as_registered',
)
BOUND_KEYS = ('foreign_mse', 'clean_mse', 'recall_mse', 'postshift_mse', 'action_cost')
IDENTITY_KEYS = ('signed_intensity', 'generic_equivalent', 'pole_reversal_with_decoder')


def decide(*, registered: bool, validity: Mapping[str, bool],
           seeds: int, trials: int, upper99: Mapping[str, float],
           identity_errors: Mapping[str, float], hard_violations: int) -> str:
    if registered is not True:
        return 'not_registered'
    if (set(validity) != set(VALIDITY_KEYS)
            or any(validity[k] is not True for k in VALIDITY_KEYS)
            or type(seeds) is not int or seeds != PARAMETERS['confirmation_seeds']['count']
            or type(trials) is not int or trials != PARAMETERS['expected_confirmation_trials']
            or set(upper99) != set(BOUND_KEYS)
            or set(identity_errors) != set(IDENTITY_KEYS)
            or type(hard_violations) is not int or hard_violations < 0):
        return 'invalid_experiment'
    scalars = list(upper99.values()) + list(identity_errors.values())
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in scalars):
        return 'invalid_experiment'
    if any(v < 0 or v > PARAMETERS['coordinate_action_tolerance']
           for v in identity_errors.values()):
        return 'invalid_experiment'
    primary = upper99['foreign_mse'] < -PARAMETERS['practical_mse_margin']
    guardrails = (
        upper99['clean_mse'] <= PARAMETERS['clean_mse_noninferiority_margin']
        and upper99['recall_mse'] <= PARAMETERS['recall_mse_noninferiority_margin']
        and upper99['postshift_mse'] <= PARAMETERS['postshift_mse_noninferiority_margin']
        and upper99['action_cost'] <= PARAMETERS['normalized_action_cost_margin']
        and hard_violations == 0
    )
    if primary and guardrails:
        return 'provenance_use_supported_in_scope'
    return 'provenance_use_not_supported_in_scope'
