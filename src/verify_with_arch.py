"""Optional external estimator validation: python src/verify_with_arch.py.
Both estimators use percent returns, identical zero-mean Gaussian GARCH(1,1).
Differences in backcast, optimizer, and parameter constraints are documented.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from full_garch_validation import fit_garch

def objective(z,r):
    w,a,b=np.asarray(z,dtype=float)
    if w<=0 or a<0 or b<0 or a+b>=.995:return float('inf')
    h=np.empty(len(r));h[0]=max(float(np.var(r)),1e-5)
    for t in range(1,len(r)):h[t]=w+a*r[t-1]**2+b*h[t-1]
    return float(np.mean(np.log(h)+r*r/h))

def check(path):
    try:from arch import arch_model
    except ImportError as exc:raise SystemExit('NOT VERIFIED: install arch then rerun') from exc
    df=pd.read_csv(path);df['Date']=pd.to_datetime(df.Date)
    df=df.sort_values('Date')
    r=100*np.diff(np.log(df.Close.to_numpy(dtype=float)))
    r=r[:int(.7*len(r))]
    ours,_=fit_garch(r/100)
    fit=arch_model(r,mean='Zero',vol='GARCH',p=1,o=0,q=1,dist='normal',rescale=False).fit(disp='off')
    theirs=np.array([fit.params['omega'],fit.params['alpha[1]'],fit.params['beta[1]']])
    print(path,'ours',ours,'arch',theirs,'our QLIKE',objective(ours,r),'arch QLIKE under our initialization',objective(theirs,r))
    print('ARCH optimizer converged:',fit.convergence_flag==0)
    return objective(ours,r),objective(theirs,r)
if __name__=='__main__':
    for path in sys.argv[1:] or ['data/sp500.csv','data/tasi.csv']:check(path)
