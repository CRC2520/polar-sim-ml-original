"""CLI for the guarded B1-SC-D2-R1 full prospective pipeline."""
from __future__ import annotations
import argparse
from experiments.b1sc_d2_r1_v1_0 import scientific_runner as r
from experiments.b1sc_d2_r1_v1_0 import full_execution as f


def main() -> None:
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True)
    q=sp.add_parser('qa-microfit');q.add_argument('--condition',required=True,choices=('S6-ON','S6-OFF-TRAIN'));q.add_argument('--block',required=True,type=int);q.add_argument('--init-dir',required=True);q.add_argument('--out',required=True);q.add_argument('--budget',type=int,default=512)
    x=sp.add_parser('fit');x.add_argument('--condition',required=True,choices=('S6-ON','S6-OFF-TRAIN'));x.add_argument('--block',required=True,type=int);x.add_argument('--init-dir',required=True);x.add_argument('--out',required=True)
    x=sp.add_parser('training-aggregate');x.add_argument('--fits',required=True);x.add_argument('--out',required=True)
    x=sp.add_parser('online');x.add_argument('--block',required=True,type=int);x.add_argument('--fits',required=True);x.add_argument('--out',required=True)
    x=sp.add_parser('online-aggregate');x.add_argument('--online-root',required=True);x.add_argument('--out',required=True)
    x=sp.add_parser('final');x.add_argument('--training',required=True);x.add_argument('--online',required=True);x.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='qa-microfit':r.qa_microfit(a.condition,a.block,a.init_dir,a.out,a.budget)
    elif a.cmd=='fit':r.scientific_fit(a.condition,a.block,a.init_dir,a.out)
    elif a.cmd=='training-aggregate':f.aggregate_training(a.fits,a.out)
    elif a.cmd=='online':f.online_block(a.block,a.fits,a.out)
    elif a.cmd=='online-aggregate':f.aggregate_online(a.online_root,a.out)
    elif a.cmd=='final':f.final_aggregate(a.training,a.online,a.out)

if __name__=='__main__':main()
