import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from benchmark import score

def test_qlike_finite():
    d=score(np.array([0.,.001]),np.array([.0002,.001]))
    assert np.isfinite(d['QLIKE']) and d['MSE']>=0
