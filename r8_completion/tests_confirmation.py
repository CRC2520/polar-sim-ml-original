import unittest
import numpy as np
from r8_completion.confirmation import binomial_upper, holm_adjust, conservative_power, confirmation_family, CLAIMS


class ConfirmationContracts(unittest.TestCase):
    def test_exact_reference_cases(self):
        self.assertEqual(binomial_upper(30, 30), 2.**-30)
        self.assertEqual(binomial_upper(0, 30), 1.)
        self.assertAlmostEqual(binomial_upper(2, 3), .5)
        self.assertEqual(conservative_power()['minimum_wins_bonferroni'], 22)

    def test_holm_preserves_names_and_full_family(self):
        np.testing.assert_allclose(holm_adjust([.04, .001, .02]), [.04, .003, .04])
        data = {name: np.ones(30, dtype=bool) for name in CLAIMS}
        data['Q1'][:] = False
        result = confirmation_family(data)
        self.assertEqual(result['rows'][0]['claim'], 'Q1')
        self.assertFalse(result['rows'][0]['supported_at_family_alpha_0_05'])
        with self.assertRaises(ValueError):
            confirmation_family({'Q1': np.ones(30, dtype=bool)})


if __name__ == '__main__':
    unittest.main()
