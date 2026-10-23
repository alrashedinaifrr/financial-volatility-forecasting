import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from independent_numerical_check import reference_variance

def test_independent_filter_matches_manual_recursion():
    r=np.array([.2,-.5,.4,-.1,.3]); theta=[.03,.12,.8]
    h=reference_variance(theta,r)
    manual=[max(float(np.var(r)),1e-5)]
    for x in r[:-1]:manual.append(theta[0]+theta[1]*x*x+theta[2]*manual[-1])
    assert np.allclose(h,manual,rtol=0,atol=1e-12)
