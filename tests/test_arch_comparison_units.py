import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from verify_with_arch import objective
from full_garch_validation import fit_garch

def test_arch_comparison_same_percent_return_units():
    rng=np.random.default_rng(42)
    returns=rng.normal(0,.01,300)
    params,_=fit_garch(returns)
    value=objective(params,returns*100)
    assert np.isfinite(value)
    assert abs(value)<20

def test_objective_penalizes_invalid_parameters():
    r=np.array([.5,-.2,.4])
    assert np.isinf(objective([.1,.8,.3],r))
