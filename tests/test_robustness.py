import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from robustness import block_bootstrap_mean_ci, qlike_loss

def test_qlike_finite():
    assert np.isfinite(qlike_loss(np.array([0.,.01]),np.array([.001,.02]))).all()

def test_bootstrap_constant():
    lo,hi=block_bootstrap_mean_ci(np.full(100,.25),block=10,reps=100)
    assert abs(lo-.25)<1e-12 and abs(hi-.25)<1e-12
