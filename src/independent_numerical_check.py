"""Independent numerical cross-check; NOT a substitute for the arch package.
Uses scipy.signal.lfilter for the variance recursion and scipy.optimize differential_evolution
instead of the production SLSQP loop. Shared initialization is stated explicitly.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
from full_garch_validation import fit_garch

def reference_variance(theta, returns):
    r=np.asarray(returns,dtype=float)
    w,a,b=map(float,theta)
    v=max(float(np.var(r)),1e-5)
    # h[t] = w + a*r[t-1]^2 + b*h[t-1]; initial h[0]=v.
    forcing=w+a*r[:-1]**2
    tail,_=lfilter([1.0],[1.0,-b],forcing,zi=[b*v])
    return np.r_[v,tail]

def reference_loss(theta,returns):
    w,a,b=theta
    if w<=0 or a<0 or b<0 or a+b>=.995:return 1e8+1e8*max(0,a+b-.995)
    h=reference_variance(theta,returns)
    if not np.all(np.isfinite(h)) or np.any(h<=0):return 1e9
    return float(np.mean(np.log(h)+returns**2/h))

def compare(path):
    df=pd.read_csv(path).sort_values('Date')
    r=100*np.diff(np.log(df['Close'].to_numpy(dtype=float)))
    r=r[:int(.7*len(r))]
    ours,_=fit_garch(r/100)
    v=max(float(np.var(r)),1e-5)
    ref=differential_evolution(lambda x:reference_loss(x,r),[(1e-8,max(10*v,1)),(1e-5,.6),(.01,.995)],seed=2026,popsize=10,maxiter=110,tol=1e-7,polish=True,workers=1)
    ours_loss=reference_loss(ours,r); ref_loss=reference_loss(ref.x,r)
    gap=ours_loss-ref_loss
    return dict(market=Path(path).stem,n=len(r),ours_omega=ours[0],ours_alpha=ours[1],ours_beta=ours[2],ref_omega=ref.x[0],ref_alpha=ref.x[1],ref_beta=ref.x[2],ours_loss=ours_loss,ref_loss=ref_loss,loss_gap=gap,ref_success=bool(ref.success))

if __name__=='__main__':
    import json
    results=[compare(p) for p in (sys.argv[1:] or ['data/sp500.csv','data/tasi.csv'])]
    print(json.dumps(results,indent=2))
    if any((x['loss_gap']>0.002 or not x['ref_success']) for x in results):sys.exit(2)
