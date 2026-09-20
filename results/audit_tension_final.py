"""Preselected P1 replay audit, external to the frozen implementation.

Selection declared before final execution: first seed 934001; source polarities
0 and 7; both poles; noise 0/.02; all eleven pulse variants and both tasks with
all eleven task variants. Compare every stored numeric array exactly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p1_completion import tension as t


def audit(out):
    out = Path(out)
    started = json.loads((out/'RUN_STARTED.json').read_text())
    assert started['seeds'][0] == 934001
    assert t.source_hashes() == started['source_sha256']
    checks = []
    with np.load(out/'raw/seed_934001.npz', allow_pickle=False) as saved:
        for source in (0, 7):
            for pole in (0, 1):
                for sigma in (0., .02):
                    for variant in t.VARIANTS+('tau_clamp',):
                        prefix = f'p{source}_{pole}_n{sigma}_{variant}__'
                        rerun = t.pulse_pair(934001, source, pole, sigma, variant)
                        for key, value in rerun.items():
                            if not np.array_equal(saved[prefix+key], value):
                                raise AssertionError('Replay mismatch: '+prefix+key)
                        checks.append({'kind': 'pulse_pair', 'prefix': prefix,
                                       'arrays': len(rerun), 'exact': True})
        for task in ('switching_memory', 'gain_resource_shift'):
            for variant in t.TASK_VARIANTS:
                prefix = f'task_{task}_{variant}__'
                rerun = t.task_trial(934001, task, variant)
                for key, value in rerun.items():
                    if not np.array_equal(saved[prefix+key], value):
                        raise AssertionError('Replay mismatch: '+prefix+key)
                checks.append({'kind': 'task_episode', 'prefix': prefix,
                               'arrays': len(rerun), 'exact': True})
    result = {'seed': 934001, 'preselected_pulse_pairs': 88,
              'preselected_task_episodes': 22, 'all_selected_arrays_exact': True,
              'compared_arrays': sum(x['arrays'] for x in checks),
              'source_unchanged': t.source_hashes() == started['source_sha256'],
              'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checks': checks}
    (out/'PRESELECTED_REPLAY.json').write_text(json.dumps(result, indent=2)+'\n')
    return {k: v for k, v in result.items() if k != 'checks'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.out), indent=2))
