"""B1-S v1.1: retrospective arithmetic on an existing ZIP; no model execution.
Python standard library only. No network, simulator, learner, random seed factory,
workflow invocation, pickle/checkpoint loading or changes to archived evidence.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import statistics as st
import zipfile

ZIP_HASH = '42504ec6a08c9bdb3cf75fd8d89fb2af2e8502396ae7baf917f7acfa1ba6eeb7'
SOURCE = '50c4e08cf3d69c998ce95b8da1e1647a09b31339'
PUBLICATION = 'de35a0995d96ff548772fa56c3620466495add72'
DOCUMENT = '6625fb6dfbd9505115017d51eb6d534add9fc35b'
FREEZE = '9a61e714e78b9c2235dfc6dfe4093be0f5b66a9fa3ae4b15ef967a12f354d7c8'
ROLES = ('S-M0', 'S-M2-fixed', 'S-M3', 'S-M4', 'S-M5', 'S-M6')
DELTA = 130 / 30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def parse(b):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(b, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def flatten(a):
    return [v for x in a for v in flatten(x)] if isinstance(a, list) else [a]


def mean(rows, key):
    return st.mean(row[key] for row in rows)


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def competent(loss_gain, displacement_gain):
    return loss_gain > DELTA and displacement_gain >= 1


def jac(a, b):
    aa = {i for i, x in enumerate(a) if x == '1'}
    bb = {i for i, x in enumerate(b) if x == '1'}
    return len(aa & bb) / len(aa | bb) if aa | bb else 1.0


def load_package(path):
    before = path.read_bytes()
    require(digest(before) == ZIP_HASH, 'Original documentary artifact ZIP hash mismatch')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [x.filename for x in infos]
        require(len(names) == len(set(names)), 'Duplicate ZIP paths')
        require(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
                    for n in names), 'Unsafe ZIP path')
        raw = {x.filename: archive.read(x) for x in infos if not x.is_dir()}
    p = parse(raw['IMPORT_PROVENANCE.json'])
    require(p['results_imported'] is True and p['scientific_source_commit'] == SOURCE
            and p['publication_commit'] == PUBLICATION, 'Import identity mismatch')
    require(p['historical_preservation']['status'] == p['translation']['status'] == 'PASS',
            'Published preservation/translation audit not PASS')
    for item in p['files']:
        require(digest(raw[item['local_path']]) == item['sha256'], 'Imported bytes mismatch')
    f = parse(raw['results/FREEZE.json']); r = parse(raw['results/PUBLICATION_RECEIPT.json'])
    require(digest(raw['results/FREEZE.json']) == r['freeze_sha256'] == FREEZE, 'Freeze bytes mismatch')
    require(f['source_commit'] == r['source_commit'] == SOURCE, 'Mixed scientific sources')
    present = 0
    for rel, h in f['files'].items():
        if 'results/' + rel in raw:
            require(digest(raw['results/' + rel]) == h, 'Frozen text bytes mismatch: ' + rel)
            present += 1
        else:
            require(not rel.endswith(('.json', '.md', '.txt', '.csv')), 'Missing scientific text')
    a = parse(raw['adjudication/ADJUDICATION.json'])
    manifest = parse(raw['adjudication/ADJUDICATION_MANIFEST.json'])
    for rel, h in manifest['files'].items():
        require(digest(raw['adjudication/' + rel]) == h, 'Adjudication hash mismatch')
    d = parse(raw['results/DECISION.json'])
    for obj in (a, d, f):
        require(obj['B1E_executed'] is False and obj['final_seeds_generated'] is False,
                'Development boundary mismatch')
        require(obj['ready_for_b1e_confirmatory_run'] is False, 'Confirmation enabled')
    require(a['B1E_disposition'] == 'ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION', 'Disposition changed')
    require(a['status'] == 'ADJUDICATION_COMPLETE' and a['arithmetic_audit'] == 'PASS', 'Missing adjudication')
    require(d['H_CAT'] == a['H_CAT'] == 'NOT_EVALUABLE'
            and d['H_TRANSFER'] == a['H_TRANSFER'] == 'NOT_EVALUATED', 'Scope mismatch')
    return raw, {'zip_sha256': ZIP_HASH, 'freeze_sha256': FREEZE,
                 'imported_entries_verified': len(p['files']), 'frozen_text_files_verified': present,
                 'binary_files_not_read_here': len(f['files']) - present, 'status': 'PASS'}


def review(raw, integrity):
    get = lambda name: parse(raw['results/' + name + '.json'])
    c = get('COMPETITIVE_RESULTS'); held = c['episodes']; zero = dict(mean_loss=mean(held['no-action'], 'loss'), mean_displacement=mean(held['no-action'], 'displacement'))
    global_rows, replicate_rows, strata = [], [], []
    for role in (*ROLES, 'no-action'):
        rows = held[role]
        require(len(rows) == 192 and {(x['rep'], x['episode']) for x in rows}
                == set(itertools.product(range(6), range(32))), 'Competitive identities mismatch')
        require(all(x['split'] == 'competitive-eval' and x['development_only'] is True for x in rows), 'Wrong split')
        require(all(near(x['loss'], -x['return_value']) for x in rows), 'Loss/return mismatch')
        for key, actual in [('mean_loss', mean(rows, 'loss')), ('mean_displacement', mean(rows, 'displacement'))]:
            require(near(c['summaries'][role][key], actual), 'Competitive summary mismatch')
        gain = zero['mean_loss'] - mean(rows, 'loss'); dx = mean(rows, 'displacement') - zero['mean_displacement']
        if role != 'no-action':
            require(competent(gain, dx) == c['competence'][role]['descriptive_competence_pass'], 'Competence mismatch')
        global_rows.append([role, mean(rows, 'loss'), mean(rows, 'displacement'), gain, dx,
                            sum(x['fallen_count'] > 0 for x in rows), sum(x['package_dropped'] for x in rows),
                            sum(x['cycles'] == 500 for x in rows), mean(rows, 'cycles'),
                            competent(gain, dx) if role != 'no-action' else None])
        for rep in range(6):
            rr = [x for x in rows if x['rep'] == rep]; zz = [x for x in held['no-action'] if x['rep'] == rep]
            require([x['seed'] for x in rr] == [x['seed'] for x in zz], 'Paired seeds differ')
            dg = mean(zz, 'loss') - mean(rr, 'loss'); dd = mean(rr, 'displacement') - mean(zz, 'displacement')
            replicate_rows.append([role, rep, mean(rr, 'loss'), mean(rr, 'displacement'), dd,
                                   competent(dg, dd) if role != 'no-action' else None])
        bad = [x for x in rows if x['fallen_count'] > 0 or x['package_dropped']]
        good = [x for x in rows if not (x['fallen_count'] > 0 or x['package_dropped'])]
        strata.append([role, len(bad), mean(bad, 'loss') if bad else None,
                       len(good), mean(good, 'loss') if good else None])
    disc = get('DISCOVERY_RESULTS'); sel = get('SELECTION_FREEZE'); support = sel['selected_support']
    masks = collections.defaultdict(dict); errors = []
    for row in disc['rows']:
        cfg = row['configuration']; errors.append(abs(mean(row['selection_rows'], 'loss') - row['selection_mean_loss']))
        if cfg['kind'] == 'graph' and cfg['mask'] != '111111':
            masks[cfg['mask']][cfg['rep']] = row['selection_mean_loss']
    require(max(errors) < 1e-9 and len(masks) == 42 and all(set(v) == set(range(6)) for v in masks.values()), 'Selection ledger mismatch')
    ranking = sorted(masks, key=lambda m: (st.mean(masks[m].values()), m.count('1'), m))
    require(ranking[0] == support, 'Selected support not arithmetic winner')
    require(all(near(st.mean(v.values()), sel['scores']['graph-' + m]) for m, v in masks.items()), 'Selection scores mismatch')
    perrep, loo, winners = [], [], []
    for rep in range(6):
        order = sorted(masks, key=lambda m: (masks[m][rep], m.count('1'), m)); winners.append(order[0])
        perrep.append([rep, order[0], order[1], masks[order[1]][rep] - masks[order[0]][rep],
                       order.index(support) + 1, masks[support][rep] - masks[order[0]][rep]])
        leave = sorted(masks, key=lambda m: (st.mean(v for i, v in masks[m].items() if i != rep), m.count('1'), m))
        loo.append([rep, leave[0], leave.index(support) + 1])
    require(winners == sel['winners_by_discovery_rep'], 'Individual winners mismatch')
    stab = get('STABILITY'); modal = max(collections.Counter(winners).values()) / 6
    mean_j = st.mean(jac(a, b) for a, b in itertools.combinations(winners, 2))
    require(near(modal, stab['modal_fraction']) and near(mean_j, stab['mean_jaccard']), 'Stability mismatch')
    curves = get('BASELINE_CALIBRATION')['checkpoint_curves']; cal = []
    configs = [x for x in disc['rows'] if x['configuration']['phase'] == 'calibration']
    for profile in sorted({x['configuration']['hp']['id'] for x in configs}):
        cc = [v for key, v in curves.items() if key.startswith('calibration-' + profile + '-')]
        require(len(cc) == 3 and all([x['steps'] for x in v] == [32768, 131072, 262144, 524288] for v in cc), 'Checkpoint inventory mismatch')
        for v in cc:
            for point in v:
                require(len(point['episodes']) == 20 and near(point['mean_loss'], mean(point['episodes'], 'loss')), 'Checkpoint arithmetic mismatch')
        terminal = [x['selection_mean_loss'] for x in configs if x['configuration']['hp']['id'] == profile]
        cal.append([profile, *[st.mean(v[i]['mean_loss'] for v in cc) for i in range(4)], st.mean(terminal)])
    causal = get('CAUSAL_RESULTS'); ce, cs = [], []; control_checks = []; fixed_receiver = 0; same_signs = 0
    for rep in causal['replicates']:
        require(rep['weights_unchanged'] is True and len(rep['rows']) == 8, 'Causal record incomplete')
        local, ext = [], []
        for row in rep['rows']:
            require(row['status'] == 'COMPLETE' and row['two_edge_interaction'] is None, 'Unexpected causal design')
            v = row['variants']; intact = v['intact']; lesion = v['edge-5']
            require(set(v) == {'intact', 'sham', 'off-support', 'edge-5'}, 'Unexpected variants')
            for ctrl in ('sham', 'off-support'):
                require(v[ctrl]['loss'] == intact['loss'] and v[ctrl]['cycles'] == intact['cycles']
                        and v[ctrl]['displacement'] == intact['displacement']
                        and v[ctrl]['snapshot']['action'] == intact['snapshot']['action'], 'Control mismatch')
                control_checks.append(True)
            require(intact['snapshot']['h'] == lesion['snapshot']['h']
                    and intact['snapshot']['m'] == lesion['snapshot']['m'], 'Hidden/message content changed')
            aa = flatten(intact['snapshot']['action']); bb = flatten(lesion['snapshot']['action'])
            da = max(abs(a-b) for a, b in zip(aa, bb)); dl = lesion['loss'] - intact['loss']
            require(near(da, row['effects']['edge-5']['max_action_delta'])
                    and near(dl, row['effects']['edge-5']['bypass_minus_intact_loss']), 'Causal arithmetic mismatch')
            fixed_receiver += all(aa[i] == bb[i] for i in (0, 1, 2, 3, 8, 9, 10, 11))
            sign = lambda x: (x > 0) - (x < 0)
            same_signs += all(sign(a) == sign(b) for a, b in zip(aa, bb))
            ce.append([rep['rep'], row['episode'], da, dl,
                       lesion['displacement'] - intact['displacement'], lesion['cycles'] - intact['cycles']])
            local.append(da); ext.append(dl)
        require(near(st.mean(ext), causal['external_difference_by_rep'][rep['rep']]), 'Causal replicate mismatch')
        cs.append([rep['rep'], min(local), max(local), sum(x == 0 for x in ext),
                   sum(x > 0 for x in ext), sum(x < 0 for x in ext), st.mean(ext)])
    probes = get('FUNCTIONAL_PROBES')['replicates']; probe_checks = []
    for rep in probes:
        require(len(rep['pairs']) == 15, 'Probe pairs incomplete')
        for pair in rep['pairs']:
            require(pair['within_predeclared_probe_margins'] == (pair['mean_abs'] <= .01 and pair['max_abs'] <= .05), 'Probe rule mismatch')
            probe_checks.append(pair['within_predeclared_probe_margins'])
    resource = get('RESOURCE_AUDIT'); rr = []
    for role in ROLES:
        fits = [x for x in resource['fits'] if x['id'].startswith('competitive-' + role + '-r')]
        require(len(fits) == 6, 'Resource fit inventory mismatch')
        keys = ('environment_steps', 'optimizer_updates', 'parameters_actor', 'parameters_critic', 'sampling_parallelism')
        require(all(len({x[k] for x in fits}) == 1 for k in keys), 'Heterogeneous resource row; expand table')
        rr.append([role, *[fits[0][k] for k in keys]])
    return dict(schema='b1s11-retrospective-review-v1', status='RETROSPECTIVE_REVIEW_COMPLETE',
        scientific_source_commit=SOURCE, scientific_publication_commit=PUBLICATION, documentary_commit=DOCUMENT,
        integrity=integrity, arithmetic_checks='PASS', source_results_modified=False,
        models_loaded=False, training_executed=False, policy_evaluations_executed=False,
        adjudication_rerun=False, final_seeds_generated=False, B1E_executed=False,
        B1E_disposition='ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION', ready_for_b1e_protocol_design=False,
        ready_for_b1e_freeze=False, ready_for_b1e_confirmatory_run=False, H_CAT='NOT_EVALUABLE', H_TRANSFER='NOT_EVALUATED',
        competence=dict(columns=['role','loss','displacement','loss_gain','displacement_gain','falls','drops','horizon500','cycles','archived_pass'],
                        rows=global_rows, displacement_required=zero['mean_displacement']+1,
                        per_rep_columns=['role','rep','loss','displacement','displacement_gain','posthoc_rep_pass'], per_rep=replicate_rows,
                        posthoc_strata_columns=['role','failure_union_n','failure_loss','no_failure_n','no_failure_loss'], posthoc_strata=strata),
        selection=dict(selected=support, candidate_count=42, top_columns=['mask','mean_loss','sd_six_fit_means','six_fit_means'],
                       top=[[m,st.mean(masks[m].values()),st.stdev(masks[m].values()),[masks[m][i] for i in range(6)]] for m in ranking[:6]],
                       first_second_gap=st.mean(masks[ranking[1]].values())-st.mean(masks[support].values()),
                       per_rep_columns=['rep','winner','second','winner_gap','selected_rank','selected_gap_to_winner'], per_rep=perrep,
                       modal_fraction=modal,mean_jaccard=mean_j,
                       loo_columns=['omitted_rep','arithmetic_winner','original_selected_rank'],loo=loo,
                       loo_scope='Post hoc deletion sensitivity only; no new fitted policy, no replacement of selection, no population stability estimate.'),
        calibration=dict(columns=['profile','checkpoint32768','checkpoint131072','checkpoint262144','checkpoint524288','terminal_selection_loss'], rows=cal,
                         scope='SAC callback checkpoint precedes last collection-block updates. Terminal saved model alone selected profiles.'),
        causal=dict(rep_columns=['rep','min_action_delta','max_action_delta','zero_loss','positive_loss','negative_loss','mean_loss_effect'],replicates=cs,
                    episode_columns=['rep','episode','action_delta','loss_effect','displacement_effect','cycles_effect'], episodes=ce,
                    controls_verified=len(control_checks),unchanged_other_motor_blocks=fixed_receiver,unchanged_motor_signs=same_signs,
                    mean_loss_effect=st.mean(x[3] for x in ce),two_edge_factorial_evaluated=False,
                    scope='One-time route-use intervention at the recorded prefix; unchanged action signs do not establish unchanged realized physics.'),
        functional=dict(panels=len(probes),observations_by_rep=[p['observations'] for p in probes],comparisons=len(probe_checks),within_margin=sum(probe_checks)),
        resources=dict(columns=['role','environment_steps','optimizer_updates','actor_parameters','critic_parameters','streams'],rows=rr),
        limitations=['No complete compressed trajectory archives or saved models read in this iteration.',
                     'No component-by-component reward decomposition or realized torque/energy audit.',
                     'Post hoc strata and leave-one-out calculations are descriptive, not new confirmatory tests.',
                     'No training replication, convergence certification, catalogue mapping or transfer test.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.archive.resolve() != args.out.resolve(), 'Output would overwrite input')
    raw, integrity = load_package(args.archive)
    result = review(raw, integrity)
    encoded = (json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    if args.out.exists():
        require(args.out.read_bytes() == encoded, 'Existing output differs; preserve it')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('xb') as f:
            f.write(encoded)
    require(digest(args.archive.read_bytes()) == ZIP_HASH, 'Source ZIP changed during review')
    print(json.dumps({'status':result['status'], 'arithmetic_checks':'PASS', 'metrics_sha256':digest(encoded),
                      'source_unchanged':True,'training_executed':False,'policy_evaluations_executed':False}))


if __name__ == '__main__':
    main()
