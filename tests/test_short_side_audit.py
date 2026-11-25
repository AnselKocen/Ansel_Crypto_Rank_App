import unittest
import pandas as pd
import numpy as np

from audit_short_side import linked_leg_attribution


class ShortSideAttributionTests(unittest.TestCase):
    def test_linked_leg_pnl_reconciles_with_account_compounding(self):
        f = pd.DataFrame({"long": [.1, .2], "short": [-.05, .1]},
                         index=pd.to_datetime(["2026-01-07", "2026-01-14"]))
        before, long, short, total = linked_leg_attribution(f)
        self.assertAlmostEqual(before.iloc[1], 1.05)
        self.assertAlmostEqual(long, .31)
        self.assertAlmostEqual(short, .055)
        self.assertAlmostEqual(total, .365)
        self.assertAlmostEqual(long+short, total)

    def test_attribution_rejects_missing_week_and_unknown_leg(self):
        f = pd.DataFrame({"long": [.1, .2], "short": [-.05, .1]},
                         index=pd.to_datetime(["2026-01-07", "2026-01-21"]))
        with self.assertRaises(ValueError):
            linked_leg_attribution(f)
        f.index = pd.to_datetime(["2026-01-07", "2026-01-14"])
        f.iloc[0, 1] = np.nan
        with self.assertRaises(ValueError):
            linked_leg_attribution(f)


if __name__ == "__main__":
    unittest.main()
