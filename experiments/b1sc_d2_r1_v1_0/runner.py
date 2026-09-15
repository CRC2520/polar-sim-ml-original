"""CLI entry point for B1-SC-D2-R1 scientific runner and QA microfits."""
from __future__ import annotations
import argparse
from experiments.b1sc_d2_r1_v1_0 import scientific_runner as r


def main() -> None:
    p=argparse.ArgumentParser()
    sp=p.add_subparsers(dest='cmd',required=True)
    q=sp.add_parser('qa-microfit')
    q.add_argument('--condition',required=True,choices=('S6-ON','S6-OFF-TRAIN'))
    q.add_argument('--block',required=True,type=int)
    q.add_argument('--init-dir',required=True)
    q.add_argument('--out',required=True)
    q.add_argument('--budget',type=int,default=512)
    f=sp.add_parser('fit')
    f.add_argument('--condition',required=True,choices=('S6-ON','S6-OFF-TRAIN'))
    f.add_argument('--block',required=True,type=int)
    f.add_argument('--init-dir',required=True)
    f.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='qa-microfit':
        r.qa_microfit(a.condition,a.block,a.init_dir,a.out,a.budget)
    else:
        r.scientific_fit(a.condition,a.block,a.init_dir,a.out)

if __name__=='__main__':main()
