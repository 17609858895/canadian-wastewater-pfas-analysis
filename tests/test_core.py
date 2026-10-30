import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import censdml as cd
import s01_load_clean as clean

class CoreChecks(unittest.TestCase):
    def test_truncated_draws_respect_typical_bounds(self):
        rng=np.random.default_rng(18)
        mu=np.zeros(5000); upper=np.linspace(-3,2,5000)
        values=cd.draw_trunc_upper(mu,0.8,upper,rng)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(values<=upper+1e-10))

    def test_detected_observations_remain_exact(self):
        V=np.arange(12,dtype=float).reshape(6,2)/10
        actual=cd.chained_impute(V,V,np.ones((6,1)),np.repeat([0,1,2],2),n_sweeps=1)
        np.testing.assert_array_equal(V,actual)

    def test_raw_dictionary_is_excluded(self):
        p=clean.RAW/'PFASWastewater_EN_FR.csv'
        if not p.exists():self.skipTest('download public inputs to raw/ for parser test')
        d=clean.read_block(p.name)
        self.assertEqual(list(d.columns),clean.COLS)
        self.assertGreater(len(d),1000)
        self.assertFalse(d.SAMPLE_NAME.str.contains('Data Dictionary',case=False,na=False).any())

if __name__=='__main__':unittest.main()
