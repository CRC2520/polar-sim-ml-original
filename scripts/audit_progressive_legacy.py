"""Read-only, pinned-source engineering audit of the independent phase repository.

The f30 execution is an INSTRUMENTED 20-step replay, not a published experiment.
Requires a local checkout of CRC2520/polar-sim-ml at 22891d52600f600573d88839ae237bfd6d723a6b.
It never writes or repairs the audited repository. Run untrusted source only after review.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import pathlib
import platform
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PINNED_BLOBS = {'main_f30.py':'5e6f2fb3d2057255d28d705234b4bcac9befc418',
                'main_f7.py':'010fbb6d94bfb60c0db5d6047eb1d66dca826028'}


def read_pinned(root, name):
    data=(root/name).read_bytes()
    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if actual!=PINNED_BLOBS[name]:
        raise ValueError(f'{name}: source differs from audited commit')
    return data.decode('utf-8')


def audit(root):
    text=read_pinned(root,'main_f30.py'); tree=ast.parse(text)
    loads=sum(isinstance(n,ast.Name) and n.id=='total_inputs' and isinstance(n.ctx,ast.Load)
              for n in ast.walk(tree))
    # Explicitly change only replay duration and suppress interactive plotting.
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TIME_STEPS' for t in node.targets):
            node.value=ast.Constant(value=20)
    index=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.For)
               and isinstance(n.target,ast.Name) and n.target.id=='t')
    tree.body[index:index]=ast.parse('_actor_before = [[p.detach().clone() for p in a.actor.parameters()] for a in agents]').body
    ast.fix_missing_locations(tree)
    env={'__name__':'__legacy_audit__'}
    old_show=plt.show; plt.show=lambda:None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree,'instrumented_main_f30.py','exec'),env)
    finally:
        plt.show=old_show; plt.close('all')
    actors=env['agents']
    delta=max(float((p.detach()-q).abs().max()) for a,old in zip(actors,env['_actor_before']) for p,q in zip(a.actor.parameters(),old))
    gradient_tensors=sum(p.grad is not None for a in actors for p in a.actor.parameters())
    f7=ast.parse(read_pinned(root,'main_f7.py'))
    assignments=[n for n in ast.walk(f7) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='A_next' for t in n.targets)]
    direct_f7=any(isinstance(n,ast.Name) and n.id=='M' and isinstance(n.ctx,ast.Load) for a in assignments for n in ast.walk(a.value))
    result={'status':'engineering_audit_not_performance_or_consciousness_evidence',
            'source_repository':'CRC2520/polar-sim-ml',
            'source_commit':'22891d52600f600573d88839ae237bfd6d723a6b',
            'source_blobs':PINNED_BLOBS,'instrumented_steps':20,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},
            'f30_total_inputs_read_count':loads,
            'f30_actor_parameter_max_change':delta,
            'f30_actor_parameter_tensors_with_grad':gradient_tensors,
            'f30_min_final_activation':min(float(a.A.min()) for a in actors),
            'f30_analytic_harmony_lower_bound':1-1/env['NUM_NODES'],
            'f7_state_update_reads_influence_matrix':direct_f7,
            'limitations':['Only f7/f30 are audited mechanistically; not all intermediate phases.',
                          'Runtime is an instrumented short replay; original source hashes are checked.',
                          'The 0.9 harmony floor follows ReLU/tanh and normalized positive weights, not learned competence.']}
    assert loads==0 and delta==0 and gradient_tensors==0 and direct_f7
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
    args=p.parse_args(); report=audit(args.repo)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
