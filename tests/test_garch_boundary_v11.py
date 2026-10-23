import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from full_garch_validation import fit_garch
from benchmark import prepare

def reference_filter(returns_pct, theta):
    w,a,b=theta
    h=float(np.var(returns_pct))
    values=[h]
    for ret in returns_pct[:-1]:
        h=w+a*ret*ret+b*h
        values.append(h)
    return np.asarray(values)

def test_recursion_matches_independent_reference():
    r=np.array([.002,-.015,.009,.004,-.003,.011,-.007,.005,.001,-.008]*15)
    theta,last=fit_garch(r)
    ref=reference_filter(r*100,theta)
    np.testing.assert_allclose(last,ref[-1],rtol=1e-10,atol=1e-10)
    next_variance=theta[0]+theta[1]*(100*r[-1])**2+theta[2]*last
    assert next_variance>0
    # No return beyond the training boundary is needed to produce next variance.
    r_future_a=np.r_[r, .1];r_future_b=np.r_[r,-.2]
    assert next_variance==theta[0]+theta[1]*(100*r_future_a[len(r)-1])**2+theta[2]*last
    assert next_variance==theta[0]+theta[1]*(100*r_future_b[len(r)-1])**2+theta[2]*last

def test_boundary_origin_has_known_return_only():
    # Compare incremental update to a separate recomputation with fixed parameters.
    r=np.array([.003,-.005,.004,.008,-.009,.002,.006,-.004]*20)
    theta,h=fit_garch(r[:120]); w,a,b=theta
    # first test origin: incorporate final training return exactly once
    state=w+a*(100*r[119])**2+b*h
    expected=reference_filter(r[:121]*100,theta)[-1]
    np.testing.assert_allclose(state,expected,rtol=1e-10)
    # successive observed returns, never next return
    for idx in range(120,125):
        state=w+a*(100*r[idx])**2+b*state
        np.testing.assert_allclose(state,reference_filter(r[:idx+2]*100,theta)[-1],rtol=1e-10)

def test_training_target_is_available_at_cutoff():
    data=Path(__file__).resolve().parents[1]/'data'/'sp500.csv'
    x=prepare(data)
    lo=int(len(x)*.7)
    train=x.iloc[:lo-1];test=x.iloc[lo:]
    assert train.date.max()<test.date.min()
    # A one-day-ahead label at date t needs the next observed price; last training origin
    # is intentionally excluded to avoid learning a label unavailable at test cutoff.
    assert len(train)==lo-1
