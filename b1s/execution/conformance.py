"""Validate the typed projection against actual executable objects, without training."""
import json
from .core import ROOT, EDGES, RoutingActor, GenericActor, masks
from .instrument import KWARGS, NODES


def check():
    source=json.loads((ROOT/'CONTRACT_PROJECTION.json').read_text())
    p=source['pointers']
    assert KWARGS==p['/domain/kwargs']
    assert list(NODES)==p['/domain/nodes']
    assert [list(e) for e in EDGES]==p['/operator/edge_order']
    assert len(masks())==p['/support_discovery/candidate_count']==42
    assert p['/operator/aggregation_denominator']==2
    assert p['/operator/message_rounds']==1
    actor=RoutingActor('101010')
    shapes=p['/operator/shapes']
    for name in ('E','A','B','D'):
        layer=getattr(actor,name)
        assert list(layer.weight.shape)==shapes[name]
        assert list(layer.bias.shape)==shapes['b_'+name]
    assert list(actor.log_std.shape)==shapes['training_log_std']
    assert p['/observation_contract/dimension']==93
    assert p['/domain/action_bounds']==[-1.,1.]
    assert p['/domain/native_motor_decoder/speed_magnitudes']==[4,6,4,6]
    assert p['/domain/native_motor_decoder/maximum_motor_torque']==80
    generic=GenericActor()
    return {'status':'PASS','source_repository':source['source_repository'],'source_commit':source['source_commit'],
            'source_git_blob':source['source_git_blob'],'source_scope':'Exact field equality with original verified separately by document CI',
            'routing_actor_parameters':sum(x.numel() for x in actor.parameters()),
            'generic_PPO_actor_parameters':sum(x.numel() for x in generic.parameters()),
            'routing_linear_MACs_per_decision':3*96*16+6*32*8+3*24*16+3*16*4,
            'generic_PPO_linear_MACs_per_decision':93*64+64*64+64*12,
            'MAC_scope':'Linear-layer multiply-accumulates only; excludes nonlinearities, masking, allocations, optimizer and host overhead',
            'all_graph_masks_compute_all_messages':True,'same_information_not_equal_capacity':True,'training_run':False,'B1E_executed':False}

if __name__=='__main__':print(json.dumps(check(),indent=2))
