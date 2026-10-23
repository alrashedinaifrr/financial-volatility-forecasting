import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

def test_one_step_garch_no_double_update():
    omega,alpha,beta=.1,.2,.7
    h_t=2.0
    r_t=1.5
    expected=omega+alpha*r_t*r_t+beta*h_t
    h_next=omega+alpha*r_t*r_t+beta*h_t
    assert np.isclose(h_next,expected)
    assert not np.isclose(h_next,omega+alpha*r_t*r_t+beta*h_next)

def test_sequential_forecast_matches_direct_recursion():
    theta=(.1,.2,.7); r=np.array([1.5,-.5,2.0]); h=2.0
    observed=[]
    for rt in r:
        h=theta[0]+theta[1]*rt**2+theta[2]*h
        observed.append(h)
    assert np.allclose(observed,[1.95,1.515,1.9605])
