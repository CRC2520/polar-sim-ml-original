"""Mathematical/protocol contracts only; no R9 simulator is instantiated."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.stats import binomtest

from r9_completion.statistics import (
    CLAIMS, CLAIM_SPECS, DEFAULT_PLAN, StatisticalPlan, compile_from_metrics,
    conservative_power, episode_endpoints, exact_binomial_tail, holm_adjust,
    paired_mean_interval, summarize, utility_criterion,
)


def synthetic_metrics(plan):
    domains = {domain for spec in plan.claims for domain in spec.domains}
    variants = {"full", *(variant for spec in plan.claims for variant in spec.comparators)}
    return {str(seed): {domain: {variant: {"reward_mean": .75 if variant == "full" else .50,
                                         "alive_fraction": 1.}
                               for variant in variants} for domain in domains}
            for seed in plan.seeds}


class StatisticalContracts(unittest.TestCase):
    def test_exact_binomial_tails_match_independent_reference(self):
        for n in (1, 6, 30, 80):
            for wins in (0, n//2, n):
                self.assertAlmostEqual(exact_binomial_tail(wins,n),
                    binomtest(wins,n,p=.5,alternative="greater").pvalue,places=14)
        self.assertEqual(exact_binomial_tail(0,80),1.)
        self.assertEqual(exact_binomial_tail(80,80),2.**-80)
        for wins,n in ((-1,80),(81,80),(1,0),(1.2,80),(True,80)):
            with self.assertRaises(ValueError):
                exact_binomial_tail(wins,n)

    def test_holm_preserves_order_and_monotonic_sorted_adjustment(self):
        p = [.01,.003,.5,.004,.02,.04]
        np.testing.assert_allclose(holm_adjust(p),[.04,.018,.5,.02,.06,.08],atol=1e-16,rtol=0)
        np.testing.assert_allclose(holm_adjust([.01]*6),[.06]*6,atol=1e-16,rtol=0)
        for bad in ([np.nan],[-.1],[1.1],[],[[.1,.2]]):
            with self.assertRaises(ValueError):
                holm_adjust(bad)

    def test_prospective_power_at_joint_probability_point70(self):
        power = conservative_power()
        self.assertEqual(power["minimum_wins_first_holm_step"],52)
        self.assertGreater(exact_binomial_tail(51,80),.05/6)
        self.assertLessEqual(exact_binomial_tail(52,80),.05/6)
        self.assertAlmostEqual(power["power_by_joint_seed_win_probability"]["0.7"],.8633141816831041,places=14)
        impossible = conservative_power(n=1)
        self.assertIsNone(impossible["minimum_wins_first_holm_step"])
        self.assertTrue(all(p == 0 for p in impossible["power_by_joint_seed_win_probability"].values()))

    def test_reward_cannot_compensate_absolute_or_relative_mortality(self):
        result = utility_criterion([.9,.9,.9],[.2,.2,.2],[.79,.98,1.],[.1,1.,1.],reward_margin=.02)
        np.testing.assert_array_equal(result["seed_success"],[False,False,True])
        self.assertFalse(result["absolute_alive_floor_pass"][0])
        self.assertFalse(result["alive_noninferiority_pass"][1])

    def test_inclusive_boundaries_and_no_rounding_into_wins(self):
        result = utility_criterion([.625,.624],[.5,.5],[.75,.75],[.8125,.8125],
            reward_margin=.125,absolute_alive_floor=.75,maximum_alive_loss=.0625)
        np.testing.assert_array_equal(result["seed_success"],[True,False])

    def test_domain_and_comparator_conditions_are_conjoined_not_averaged(self):
        full = np.full((2,4),.75)
        other = np.full((2,4),.50)
        other[0,3] = .75
        result = utility_criterion(full,other,np.ones_like(full),np.ones_like(full),reward_margin=.02)
        np.testing.assert_array_equal(result["seed_success"],[False,True])
        self.assertEqual(int(result["condition_success"][0].sum()),3)

    def test_missing_invalid_or_unpaired_endpoints_fail_loudly(self):
        cases = [([np.nan],[.5],[1.],[1.]),([1.01],[.5],[1.],[1.]),
                 ([.6,.7],[.5],[1.,1.],[1.,1.]),([],[],[],[])]
        for values in cases:
            with self.assertRaises(ValueError):
                utility_criterion(*values,reward_margin=.01)
        with self.assertRaises(ValueError):
            utility_criterion([.8],[.5],[1.],[1.],reward_margin=.01,expected_n=80)

    def test_fixed_horizon_includes_absorbing_death_zeros(self):
        outcome = episode_endpoints([1.,1.,0.,0.],[1,1,0,0],horizon=4)
        self.assertEqual(outcome,{"reward_mean":.5,"alive_fraction":.5,"horizon":4,"terminal_alive":False})
        for reward,alive in (([1.,1.],[1,1]),([1.,1.,1.,0.],[1,1,0,0]),
                             ([1.,0.,1.,0.],[1,0,1,0]),([1.,1.,0.,0.],[1,.5,0,0])):
            with self.assertRaises(ValueError):
                episode_endpoints(reward,alive,horizon=4)

    def test_all_six_hypotheses_and_exact_seed_order_required(self):
        wins = {claim:np.ones(80,dtype=bool) for claim in CLAIMS}
        report = summarize(wins,seeds=DEFAULT_PLAN.seeds)
        self.assertEqual(len(report["rows"]),6)
        self.assertTrue(all(row["supported"] for row in report["rows"]))
        for altered in ({k:v for k,v in wins.items() if k != "REL"},dict(wins,extra=np.ones(80))):
            with self.assertRaises(ValueError):
                summarize(altered,seeds=DEFAULT_PLAN.seeds)
        with self.assertRaises(ValueError):
            summarize(wins,seeds=tuple(reversed(DEFAULT_PLAN.seeds)))
        wins["REL"] = np.r_[np.ones(79),np.nan]
        with self.assertRaises(ValueError):
            summarize(wins,seeds=DEFAULT_PLAN.seeds)

    def test_compiler_uses_frozen_roles_and_retains_each_gate_control(self):
        plan = replace(DEFAULT_PLAN,n=4,seeds=(951901,951902,951903,951904))
        data = synthetic_metrics(plan)
        data["951901"]["ecology_delay9"]["permuted_gate"]["reward_mean"] = .75
        data["951902"]["thermal_transfer"]["dense"]["reward_mean"] = .75
        report = compile_from_metrics(data,plan=plan)
        by_claim = {row["claim"]:row for row in report["rows"]}
        self.assertEqual(by_claim["GATE"]["seed_success"],[False,True,True,True])
        self.assertEqual(by_claim["TRANSFER"]["seed_success"],[True,False,True,True])
        self.assertTrue(all(by_claim[name]["wins"] == 4 for name in ("REL","TENSION","CONTENT","GENERIC")))
        self.assertEqual(len(report["components"]["GATE"]["conditions"]),4)
        self.assertEqual(len(report["components"]["REL"]["conditions"]),2)

    def test_compiler_label_aliases_do_not_change_criteria(self):
        plan = replace(DEFAULT_PLAN,n=4,seeds=(951911,951912,951913,951914))
        data = synthetic_metrics(plan)
        aliased = {seed:{"domain_"+domain:{"arm_"+variant:{"R":m["reward_mean"],"S":m["alive_fraction"]}
                    for variant,m in variants.items()} for domain,variants in domains.items()} for seed,domains in data.items()}
        domain_names = {d:"domain_"+d for d in next(iter(data.values()))}
        variant_names = {v:"arm_"+v for v in next(iter(next(iter(data.values())).values()))}
        a = compile_from_metrics(data,plan=plan)
        b = compile_from_metrics(aliased,reward_key="R",alive_key="S",variant_names=variant_names,domain_names=domain_names,plan=plan)
        self.assertEqual(a["rows"],b["rows"])
        self.assertEqual(a["components"],b["components"])

    def test_compiler_rejects_duplicate_seed_alias_or_missing_domain(self):
        plan = replace(DEFAULT_PLAN,n=4,seeds=(951921,951922,951923,951924))
        data = synthetic_metrics(plan)
        data[951921] = data["951921"]
        with self.assertRaises(ValueError):
            compile_from_metrics(data,plan=plan)
        del data[951921]
        del data["951921"]["thermal_transfer"]
        with self.assertRaises(ValueError):
            compile_from_metrics(data,plan=plan)

    def test_descriptive_bootstrap_is_paired_complete_and_deterministic(self):
        values = np.asarray([-.2,.1,.1,.2,.3])
        result = paired_mean_interval(values,resamples=1000,bootstrap_seed=953901)
        indices = np.random.default_rng(953901).integers(0,5,(1000,5))
        expected = np.quantile(values[indices].mean(1),[.025,.975])
        np.testing.assert_allclose([result["lower"],result["upper"]],expected,rtol=0,atol=1e-15)
        self.assertEqual(result,paired_mean_interval(values,resamples=1000,bootstrap_seed=953901))
        with self.assertRaises(ValueError):
            paired_mean_interval([1.,np.nan],resamples=1000)
        with self.assertRaises(ValueError):
            paired_mean_interval(values,resamples=1000,expected_n=80)

    def test_plan_rejects_changed_family_or_ambiguous_seed_count(self):
        with self.assertRaises(ValueError):
            replace(DEFAULT_PLAN,n=79)
        with self.assertRaises(ValueError):
            replace(DEFAULT_PLAN,claims=CLAIM_SPECS[:-1])
        with self.assertRaises(ValueError):
            replace(DEFAULT_PLAN,null_win_probability=.4)


if __name__ == "__main__":
    unittest.main()
