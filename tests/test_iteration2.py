import unittest
from copy import deepcopy
import numpy as np
from integrated_polar.projection import project
from integrated_polar.layers import CalibratedMonitor,Content,ContentWorkspace
from iteration2.planning import operators,plan,project_rows,diagnostic_step
from iteration2.controller import CausalAgent,Config
from iteration2.interfaces import DomainAdapter,FreshCoordinator,Offer
from iteration2.tasks import task,CELLS,DEVELOPMENT_SEEDS,FINAL_SEEDS,CAPABILITY_SEEDS,run_trial


def packet(target=.5,n=1,H=5,**extra):
    r=np.broadcast_to(target,(H,n)).copy()
    o={'state':np.full(n,.5),'observed':np.ones(n,bool),'cue':'A','goal':r[0],
       'priorities':np.ones(n),'forecast':r,'F':np.eye(n)*.9,'actuator_retention':.5,
       'drift_forecast':np.full((H,n),.05),'budget_forecast':np.full(H,2*n),
       'costs':np.ones(2*n),'allowed':np.ones(2*n,bool),'normative':{}}
    o.update(extra);return o


def agent(mode='full',**kw):return CausalAgent(Config(n=1,mode=mode,**kw))

def learn_known(a,o):
    u=a.act(o);y=a.belief.copy();a.learn({'state':y,'observed':np.ones(a.n,bool),'valid_transition':np.ones(a.n,bool),'token':deepcopy(a.pending['token'])});return u


class IterationTwoTests(unittest.TestCase):
    def test_01_vector_projection_matches_independent_scalar_projection(self):
        rng=np.random.default_rng(791610)
        for _ in range(30):
            x=rng.uniform(-1,2,(5,6));c=rng.uniform(.5,1.5,6);b=rng.uniform(0,2,5);mask=rng.random(6)>.2
            actual=project_rows(x,c,b,mask)
            expected=np.array([project(x[i],c,b[i],mask) for i in range(5)])
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
    def test_02_projection_rejects_impossible_obligation(self):
        with self.assertRaises(ValueError):project_rows(np.ones((2,2)),np.ones(2),[.1,.1],np.ones(2,bool),[.2,0])
    def test_03_delayed_prediction_equals_direct_rollout(self):
        rng=np.random.default_rng(791611);F=np.eye(3)*.9;B=rng.uniform(-.1,.1,(3,6));x=rng.random(3);h=rng.random(6);u=rng.random((5,6));D=rng.uniform(0,.05,(5,3))
        A,b=operators(F,B,.7,D,x,h);states=[]
        for t in range(5):h=.7*h+.3*u[t];x=F@x+D[t]+B@h;states.append(x.copy())
        np.testing.assert_allclose((A@u.ravel()+b).reshape(5,3),states,rtol=0,atol=1e-13)
    def test_04_jacobian_finite_difference(self):
        F=np.eye(2)*.9;B=np.array([[.1,-.1,.03,-.02],[.02,-.01,.1,-.1]]);D=np.full((3,2),.05);x=np.array([.3,.6]);h=np.zeros(4)
        A,b=operators(F,B,.5,D,x,h);u=np.full(12,.2);eps=1e-6
        for j in range(12):
            du=np.zeros(12);du[j]=eps
            np.testing.assert_allclose(((A@(u+du)+b)-(A@u+b))/eps,A[:,j],atol=1e-9,rtol=0)
    def test_05_signed_information_changes_only_error_readout(self):
        args=(np.array([[.1,-.1,.02,-.02],[.03,-.03,.1,-.1]]),[.7,.2],[.3,.6],[2,1],[.3,.2,.4,.1],[1,.5])
        _,signed=diagnostic_step(*args,signed=True);_,unsigned=diagnostic_step(*args,signed=False)
        np.testing.assert_array_equal(signed['conflict_gradient'],unsigned['conflict_gradient']);np.testing.assert_array_equal(signed['receiver'],unsigned['receiver'])
        self.assertNotEqual(signed['transmitted_error'],unsigned['transmitted_error'])
    def test_06_coactivation_flag_does_not_change_sign_or_routing(self):
        args=(np.array([[.1,-.1]]),[.5],[.7],[1],[.3,.4],[1])
        _,a=diagnostic_step(*args,conflict=True);_,b=diagnostic_step(*args,conflict=False)
        self.assertEqual(a['receiver'],b['receiver']);self.assertEqual(a['transmitted_error'],b['transmitted_error']);self.assertNotEqual(a['conflict_gradient'],b['conflict_gradient'])
    def test_07_receiver_flag_leaves_message_unchanged(self):
        args=(np.array([[.1,-.1]]),[.5],[.7],[1],[.3,.4],[1])
        _,a=diagnostic_step(*args,directed=True);_,b=diagnostic_step(*args,directed=False)
        self.assertEqual(a['transmitted_error'],b['transmitted_error']);self.assertEqual(a['conflict_gradient'],b['conflict_gradient'])
        self.assertNotEqual(a['receiver'],b['receiver'])
    def test_08_forecast_makes_anticipatory_action_not_future_truth_leak(self):
        a,b=agent(),agent('myopic');o=packet([[.5],[.8],[.8],[.8],[.8]])
        ua,ub=a.act(o),b.act(o);self.assertGreater(ua[0],ub[0]+.01)
    def test_09_belief_and_polar_continuity_are_distinct(self):
        a=agent();o=packet(.8,allowed=np.array([False,True]));u=a.act(o)
        self.assertEqual(u[0],0);self.assertGreater(a.p[0],0);self.assertGreater(a.intention[0],0)
        self.assertEqual(a.belief.shape,(1,));self.assertEqual(a.p.shape,(2,))
    def test_10_missing_sensor_value_cannot_change_action(self):
        a=agent();learn_known(a,packet(.6));b=deepcopy(a)
        o=packet(.65,observed=np.array([False]),state=np.array([0.]))
        np.testing.assert_array_equal(a.act(o),b.act({**o,'state':np.array([99.])}))
    def test_11_hidden_feedback_does_not_train_unobserved_rows(self):
        a=agent();a.act(packet(.7));B=a.model.B.copy();pred=a.belief.copy()
        a.learn({'state':[99.],'observed':np.array([False]),'valid_transition':np.array([False]),'token':deepcopy(a.pending['token'])})
        np.testing.assert_array_equal(a.model.B,B);np.testing.assert_array_equal(a.belief,pred)
    def test_12_wrong_actor_does_not_consume_feedback(self):
        a=agent();a.act(packet());token=deepcopy(a.pending['token'])
        with self.assertRaises(ValueError):a.learn({'state':[.5],'observed':np.array([True]),'valid_transition':np.array([True]),'token':{**token,'actor':'foreign'}})
        self.assertIsNotNone(a.pending)
    def test_13_low_success_probability_does_not_automatically_stop_new_policy(self):
        mon=CalibratedMonitor.restore({'edges':[0.],'probabilities':[0.,0.],'base':0.})
        a=CausalAgent(Config(n=1,mode='utility_meta',beta=0),mon);b=CausalAgent(Config(n=1,mode='legacy_review'),mon)
        ua,ub=a.act(packet(.8)),b.act(packet(.8))
        self.assertGreater(ua.sum(),.1);self.assertEqual(ub.sum(),0)
    def test_14_generic_formula_equivalence(self):
        a,b=agent(),agent('generic_equivalent')
        for t in range(5):
            o=packet(.52+t*.01)
            np.testing.assert_allclose(learn_known(a,o),learn_known(b,o),atol=1e-9,rtol=0)
    def test_15_individual_unit_semantics_and_rejections(self):
        w=DomainAdapter('water',2,5).parse('tank_1 = 0.75 @ 4; tank_2 = 0.25 @ 1','A')
        h=DomainAdapter('thermal',2,5).parse('zone_1 = 28 @ 4; zone_2 = 20 @ 1','A')
        self.assertEqual(w.targets,h.targets)
        for text in ('tank_1 = 28 @ 4; tank_2 = 20 @ 1','zone_1 = 40 @ 1; zone_2 = 20 @ 1'):
            with self.assertRaises(ValueError):DomainAdapter('thermal',2,5).parse(text,'A')
    def test_16_domain_changes_are_physical_not_only_names(self):
        w=task(791613,('water','strong','fixed'));h=task(791613,('thermal','strong','fixed'))
        self.assertFalse(np.array_equal(w['F'],h['F']));self.assertNotEqual(w['a'],h['a']);self.assertLess(w['B'][0,0,2],0);self.assertGreater(h['B'][0,0,2],0)
    def test_17_stale_duplicate_foreign_packets(self):
        c=FreshCoordinator(['A','B'],ttl=2);p=Offer('A',1,0,2.,1.)
        self.assertEqual(c.receive([p],0)[0]['decision'],'accepted')
        self.assertEqual(c.receive([p],0)[0]['decision'],'duplicate_or_reordered')
        self.assertEqual(c.receive([Offer('B',1,0,1,1)],3)[0]['decision'],'expired_or_future')
        self.assertEqual(c.receive([Offer('X',1,3,1,1)],3)[0]['decision'],'unknown_actor')
        alloc=c.allocate(.7,5);self.assertLessEqual(sum(alloc.values()),.7+1e-12)
    def test_18_content_competition_and_selective_cut(self):
        c=ContentWorkspace();c.cut.add('planner')
        a=Content('A','cue',((.2,),),((1.,),),salience=.1);b=Content('B','cue',((.8,),),((1.,),),salience=2.)
        delivered=c.broadcast([a,b]);self.assertIsNone(delivered['planner']);self.assertEqual(delivered['reporter'].identity,'B');self.assertEqual(delivered['memory'].identity,'B')
    def test_19_seed_partitions(self):
        self.assertFalse(set(DEVELOPMENT_SEEDS)&set(FINAL_SEEDS));self.assertFalse(set(CAPABILITY_SEEDS)&set(FINAL_SEEDS))
        self.assertEqual(len(FINAL_SEEDS)*len(CELLS)*12,2304)
    def test_20_closed_loop_trace_and_permission_constraints(self):
        r=run_trial(791614,('thermal','strong','change'),'full')
        self.assertEqual(len(r['frames']),48);self.assertEqual(r['metrics']['violations'],0)
        for f in r['frames']:self.assertIsNotNone(f['controller']['feedback'])


if __name__=='__main__':unittest.main()
