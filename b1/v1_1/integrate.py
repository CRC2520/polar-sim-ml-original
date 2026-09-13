"""Assemble development evidence without running tasks or statistical simulations."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
FLAGS = dict(development_only=True, confirmatory=False, reusable_as_final=False,
             final_seeds_generated=False, B1E_executed=False)

def read(name):
    return json.loads((HERE / name).read_text())

def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def table(name, rows):
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    p5 = read('p5/results/RESULTS.json')
    p5audit = read('p5/P5_BENCHMARK_AUDIT.json')
    resource = read('RESOURCE_MATCHING_CERTIFICATE.json')
    export = read('EXPORTER_V1_1_AUDIT.json')
    stats = read('B1E_STATISTICAL_REDESIGN.json')
    simulation = read('statistics/SIMULATION_RESULTS.json')
    feasible = read('B1E_RESOURCE_CERTIFICATE.json')
    assert p5['status'] == 'PASS' and p5['C6_status'] == 'PASS'
    assert resource['blocker_status'] == 'RESOLVED' and not resource['resource_cap_binding']
    assert export['status'] == 'PASS' and export['archive_replay']['diagnostic_rows_exported'] == 96
    assert simulation['status'] == 'PASS' and feasible['feasible'] and feasible['N'] == stats['N_v2']
    rows = []
    def blocker(identifier, cause, resolution, change, artifacts, runs, rerun, verification, limitations):
        rows.append(dict(blocker_id=identifier, v1_status='STILL_BLOCKED', root_cause=cause,
                         resolution=resolution, resolution_type=change, affected_artifacts=artifacts,
                         affected_scientific_runs=runs, rerun_required=rerun, verification=verification,
                         v1_1_status='RESOLVED', remaining_limitations=limitations))
    blocker('B1-CONFLICT-P5-C4-REFILL-001', 'Conflicting B0 machine-readable and prose C4 refill rules',
            'Authorized prospective minimal-next-demand disposition; full-refill ceiling witness kept separately',
            'prospective_protocol_disposition;comparator_behavior_change',
            'P5_C4_DISPOSITION.json;P5_PROTOCOL_V1_1.json;p5/', 'New P5 v1.1 development only', True,
            f"Published source {p5['source_commit']};32 bundles;288 episodes;exact C6 and archive verification PASS",
            'P5 remains ceiling negative control; no positive utility or pairing eligibility')
    blocker('B1-RESOURCE-CERTIFICATE-PENDING', 'Provisional logical caps and incomplete computational fairness certificate',
            'Common logical opportunity certified for finite tasks; actual use and host axes quantified',
            'resource_envelope_only;instrumentation_only', 'RESOURCE_MATCHING_SPEC.json;RESOURCE_MATCHING_CERTIFICATE.json;resources/',
            'QA stress and replay only; selected P7 scientific data unchanged', False,
            f"{resource['axis_rows']} axis rows;C0-C4 scoped invariants PASS;no binding cap;P7 action/memory equality",
            'Host CPU, latency and byte allocation are sampled;C5 descriptive;no exact actual-use equality claim')
    blocker('B1D-ORIGINAL-EXPORT-PROVENANCE-INCIDENT', 'undetermined;technical closure omitted diagnostic completeness gate',
            'RESOLVED_FOR_FUTURE_PIPELINE', 'exporter_only', 'exporter/;EXPORTER_ROOT_CAUSE.json;EXPORTER_V1_1_AUDIT.json',
            'Historical archive representation and QA only;no new scientific observations', False,
            '96/96 rows;42 original+54 sidecar byte-identical;82176 events;corruption and end-to-end QA PASS',
            'Historical original export remains failed;root cause remains undetermined')
    blocker('B1E-RESOURCE-INFEASIBLE-CURRENT-DISK', 'v1 distribution-free 27-endpoint precision and full-trace resource projection',
            'Prospective fixed-N v2 candidate and tiered complete causal retention fit measured host estimate',
            'statistical_design_only;instrumentation_only', 'B1E_STATISTICAL_REDESIGN.json;B1E_RESOURCE_CERTIFICATE.json;TRACE_RETENTION_POLICY.json',
            'Synthetic statistical method simulations and archived replay only', False,
            f"N={stats['N_v2']};simultaneous interval validation PASS;all-failure disk estimate {feasible['disk']['all_failures_total_bytes']} bytes",
            'Estimated feasibility is not resource reservation or a worst-case runtime/compression guarantee')
    table('BLOCKER_CLOSURE_MATRIX.csv', rows)
    result = dict(version='1.1', status='PASS', completion_decision='Final B1D status is recorded in B1D_V1_1_MANIFEST.json after CI',
                  preserves_v1_results=True, results_combined_across_versions=False,
                  P5=p5, P6=dict(instrument='PASS', role='negative_control', bound='1/32',
                                positive_utility_eligible=False, positive_pairing_specificity_eligible=False,
                                scientific_rerun=False, source='immutable B1-D v1'),
                  P7=dict(instrument='PASS', selected_C1_configuration='cfg01', selected_policy='reactive',
                          v1_paired_differences='C1-C0/C2/C3/C4=0 descriptively', v1_acute_route_effect=0,
                          v1_results_preserved=True, retuning=False, scientific_rerun=False,
                          resource_QA_replay_decisions=resource['P7_archived_input_replay']['decisions']),
                  resource_matching_status=resource['status'], exporter_status=export['blocker_disposition'],
                  statistical_design_version=stats['statistical_design_version'], N_v1=stats['N_v1'], N_v2=stats['N_v2'],
                  simulation=dict(status=simulation['status'], scenario_cells=simulation['scenario_cells'],
                                  replications_per_cell=simulation['repetitions_per_scenario'],
                                  minimum_family_coverage=simulation['minimum_family_coverage'],
                                  maximum_family_false_declaration_rate=simulation['maximum_family_false_declaration_rate']),
                  resource_feasibility=feasible['feasibility_kind'], estimated_feasible=feasible['feasible'],
                  disk_before_bytes=373568924236.125, disk_after=feasible['disk'],
                  runtime_before_serial_hours=614.271911866, runtime_after=feasible['CPU_and_wall_time'],
                  blockers=rows, ready_for_b1e_confirmatory_run=False,
                  joint_manipulation=True,
                  positive_aggregate_claim_limit='Current joint source/sham fixtures fail the frozen source-causal admissibility guard; all four positive aggregate gates remain closed even if scalar comparisons improve. This guard and reactive P7 are not altered.',
                  epistemic_contract='Implementation, instrument, causality, utility, pairing, generalization and consciousness relevance are separate. Program B remains B-design.', **FLAGS)
    write('B1D_V1_1_RESULTS.json', result)
    sources = ['p5/results/RESULTS.json','p5/P5_BENCHMARK_AUDIT.json','p5/P5_SOURCE_FREEZE.json',
               'RESOURCE_MATCHING_CERTIFICATE.json','EXPORTER_V1_1_AUDIT.json',
               'B1E_STATISTICAL_REDESIGN.json','statistics/SIMULATION_RESULTS.json','B1E_RESOURCE_CERTIFICATE.json']
    audit = dict(status='PASS', **FLAGS, historical_code_changed=False, historical_results_changed=False,
                 b0_changed=False, b1d_v1_changed=False,
                 evidence_sha256={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in sources},
                 P5_source_commit=p5['source_commit'], P5_source_freeze_precedes_benchmark=True,
                 P5_remote_source_blobs_verified_before_execution=54,
                 P5_new_development_episodes=p5['episode_count'], P5_archive_audit=p5audit,
                 resource_axis_rows=resource['axis_rows'], original_export_incident_preserved=True,
                 original_exporter_tests_scope='Run original exact-inventory tests in detached v1 checkout; future pipeline tests run at v1.1 HEAD',
                 CI_gate='Final manifest must reference actual successful content validation; no CI success invented here')
    write('B1D_V1_1_AUDIT.json', audit)
    text = f'''# B1-D v1.1 development results

All four technical blockers have closure evidence. The final B1D completion decision belongs to the manifest after CI validation. No positive scientific claim is promoted by technical closure.

P5 executed once after published source freeze `{p5['source_commit']}`: {p5['bundle_count']} bundles, {p5['episode_count']} episodes, {p5['event_count']} events and {p5['seed_uses']} new development seed uses. Every arm attained the physical ceiling; mean loss {p5['mean_losses']['C1']}. Every paired C1-C0/C2/C3/C4 difference was exactly zero. C4 terminal refill was {p5['C4_terminal_refill_total']}; the separate full-refill witness requested {p5['full_witness_terminal_refill_total']} units. C6 passed exact checks. There were zero failures, guardrail violations or binding resource caps.

P6 remains the frozen negative control with expected bound 1/32. P7 remains the frozen reactive cfg01 development realization, with zero paired and route differences descriptively. Neither pilot was retuned or counted as a new scientific sample. Resource QA verified {resource['P7_archived_input_replay']['decisions']} P7 decisions with unchanged actions and memory.

The resource certificate contains {resource['axis_rows']} records across 16 dimensions. C0-C4 have matched maximum logical opportunity; C5 remains descriptive. Actual CPU, latency and physical memory are quantified with scope limitations.

The new exporter verified 96/96 archived diagnostics, 42 original and 54 previously recovered rows byte-identically, across 82176 archived events. Its corruption and QA checks pass. The historical export remains failed and its root cause remains undetermined; closure applies to the future pipeline.

The prospective v2 candidate fixes N={stats['N_v2']} for P7, compared with N={stats['N_v1']} in v1. It uses simultaneous empirical Bernstein intervals for nine eligible P7 endpoints and restrictive positive gatekeeping. All 27 historical endpoints remain visible. Margins are unchanged; actual final sample variance, never the development planning bound, determines interval width. The joint precision planning guarantee is at least 90%; simultaneous final coverage is at least 95% under the stated independent-bundle assumptions. Failed precision targets remain inconclusive, without sample extension.

Synthetic method validation covered {simulation['scenario_cells']} cells with {simulation['repetitions_per_scenario']} replications each: minimum family coverage {simulation['minimum_family_coverage']}, maximum false declaration rate {simulation['maximum_family_false_declaration_rate']}. These are implementation checks and provide no evidence about the model.

Storage changes from the v1 partial estimate of 373.568924236 GB to {feasible['disk']['typical_total_bytes']/1e9} GB typically and {feasible['disk']['all_failures_total_bytes']/1e9} GB with all bundles retaining full traces, including declared contingency and reserve. The planning budget is {feasible['CPU_and_wall_time']['CPU_hours_planning_budget']} CPU hours and {feasible['CPU_and_wall_time']['wall_hours_planning_budget']} wall hours with four workers and 75% parallel efficiency. Estimates do not reserve infrastructure or guarantee arbitrary compression tails. Storage exhaustion fails closed and retains completed evidence.

Current joint source/sham fixtures cannot open any of the four positive aggregate gates under the preserved causal admissibility guard, regardless of favorable scalar signs. This scientific limitation is explicit. No final seeds were generated; B1-E was not executed; confirmatory readiness remains false.
'''
    (HERE/'B1D_V1_1_RESULTS.md').write_text(text)
    table('CHANGE_CLASSIFICATION.csv', [dict(change='P5 C4 disposition and new comparison',classification='comparator_behavior_change',scientific_rerun_required=True,scope='P5 v1.1 only'),
        dict(change='Fail-closed exporter',classification='exporter_only',scientific_rerun_required=False,scope='archive replay plus QA'),
        dict(change='Common resource envelope',classification='resource_envelope_only',scientific_rerun_required=False,scope='action-preserving QA replay'),
        dict(change='Resource accounting and retention',classification='instrumentation_only',scientific_rerun_required=False,scope='read-only measurement and QA'),
        dict(change='Fixed N and interval design',classification='statistical_design_only',scientific_rerun_required=False,scope='synthetic statistical simulations'),
        dict(change='Reports and Spanish translation',classification='documentary_only',scientific_rerun_required=False,scope='pinned evidence only')])
    print(json.dumps({'status':'PASS','blockers_with_closure_evidence':len(rows),'P5_episodes':p5['episode_count'],'N_v2':stats['N_v2']}))

if __name__ == '__main__':
    main()
