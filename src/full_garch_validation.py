"""Full out-of-sample GARCH benchmark, one pre-test fit per expanding fold.
Fixed fitted parameters are filtered sequentially using only information at each forecast origin.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm,chi2
from advanced_validation import fit_predict, losses, bootstrap_diff, kupiec
from benchmark import prepare,EPS

def fit_garch(r):
    r=np.asarray(r,dtype=float)*100
    v=max(float(np.var(r)),1e-5)
    def filtering(theta,seq):
        w,a,b=theta; h=np.empty(len(seq));h[0]=v
        for t in range(1,len(seq)):
            h[t]=w+a*seq[t-1]**2+b*h[t-1]
        return h
    def obj(z):
        w,a,b=z
        if a+b>=.995: return 1e9
        h=filtering(z,r)
        return float(np.mean(np.log(h)+r*r/h))
    fit=minimize(obj,[v*.04,.08,.88],method='SLSQP',bounds=[(1e-8,max(v*10,1)),(1e-5,.6),(.01,.995)],constraints=[{'type':'ineq','fun':lambda z:.995-z[1]-z[2]}],options={'maxiter':250,'ftol':1e-9})
    if not fit.success or not np.isfinite(fit.fun): raise RuntimeError('GARCH optimizer failed: '+fit.message)
    return fit.x,filtering(fit.x,r)[-1]

def run(market,path,out):
    x=prepare(path)
    raw=pd.read_csv(path);raw['Date']=pd.to_datetime(raw.Date);raw=raw.sort_values('Date').reset_index(drop=True)
    r=np.log(raw.Close.to_numpy()).astype(float);r=np.diff(r,prepend=np.nan)
    lookup={d:i for i,d in enumerate(raw.Date)}
    n=len(x);bounds=[int(n*z) for z in [.70,.80,.90,1.0]]
    allrows=[]; parameters=[]
    for fold,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:]),1):
        train,test=x.iloc[:lo],x.iloc[lo:hi]
        preds=fit_predict(train,test)
        first=lookup[train.date.iloc[-1]]
        theta,h=fit_garch(r[1:first+1]); h=theta[0]+theta[1]*(100*r[first])**2+theta[2]*h; parameters.append({'market':market,'fold':fold,'omega':theta[0],'alpha':theta[1],'beta':theta[2]})
        g=[];prev_idx=first
        for d in test.date:
            idx=lookup[d]
            for k in range(prev_idx+1,idx+1):
                h=theta[0]+theta[1]*(100*r[k])**2+theta[2]*h
            g.append(max(h/10000,EPS))
            prev_idx=idx
        preds['GARCH']=np.asarray(g)
        for j,d in enumerate(test.date):
            idx=lookup[d];target=float(test.y.iloc[j]);signed=float(r[idx+1])
            for name,forecast in preds.items():
                allrows.append({'market':market,'fold':fold,'date':d,'model':name,'y':target,'pred_var':float(forecast[j]),'next_return':signed})
    df=pd.DataFrame(allrows);df.to_csv(out/f'{market}_predictions.csv',index=False)
    pd.DataFrame(parameters).to_csv(out/f'{market}_garch_parameters.csv',index=False)
    summary=[];comp=[]
    for (fold,model),g in df.groupby(['fold','model']):
        q,m=losses(g.y,g.pred_var)
        exceptions=g.next_return.to_numpy()<norm.ppf(.01)*np.sqrt(g.pred_var.to_numpy())
        exc,nn,p=kupiec(exceptions)
        summary.append({'market':market,'fold':fold,'model':model,'n':len(g),'QLIKE':float(q.mean()),'MSE':float(m.mean()),'VaR_1pct_exceptions':exc,'Kupiec_p':p})
    for fold in [1,2,3]:
        f=df[df.fold==fold]; base=f[f.model=='EWMA'].set_index('date')
        for name in ['RV20','RF','HGB_log','GARCH']:
            g=f[f.model==name].set_index('date').join(base[['pred_var']],rsuffix='_base')
            q,_=losses(g.y,g.pred_var); qb,_=losses(g.y,g.pred_var_base)
            d,l,u=bootstrap_diff(q,qb,B=2000,block=10,seed=2026+fold)
            comp.append({'market':market,'fold':fold,'model':name,'QLIKE_difference_vs_EWMA':d,'CI95_low':l,'CI95_high':u})
    return pd.DataFrame(summary),pd.DataFrame(comp)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--sp500',default='data/sp500.csv');p.add_argument('--tasi',default='data/tasi.csv');p.add_argument('--out',default='outputs_v6');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ss=[];cc=[]
    for market,path in [('SP500',a.sp500),('TASI',a.tasi)]:
        s,c=run(market,path,out);ss.append(s);cc.append(c)
    scores=pd.concat(ss,ignore_index=True);scores.to_csv(out/'scores.csv',index=False)
    pd.concat(cc,ignore_index=True).to_csv(out/'bootstrap.csv',index=False)
    print(scores.to_string(index=False))
