"""Expanding-window OOS validation with strictly positive variance forecasts."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from benchmark import prepare, FEATURES, EPS

SEED=2026

def garch_forecast(returns, horizon=1):
    """Gaussian GARCH(1,1), estimated on returns available through forecast origin.
    Output is next-session conditional variance. No claim of intraday realized variance.
    """
    r=np.asarray(returns,dtype=float)*100
    if len(r)<100: raise ValueError('insufficient returns')
    def objective(z):
        omega,alpha,beta=z
        if alpha+beta>=.999: return 1e12
        h=np.empty(len(r));h[0]=max(np.var(r),1e-6)
        for t in range(1,len(r)):h[t]=omega+alpha*r[t-1]**2+beta*h[t-1]
        return float(.5*np.sum(np.log(h)+r*r/h))
    v=max(np.var(r),1e-5)
    res=minimize(objective,[v*.04,.08,.88],method='SLSQP',bounds=[(1e-8,max(v*10,1)),(1e-5,.6),(.01,.999)],constraints=[{'type':'ineq','fun':lambda z:.999-z[1]-z[2]}],options={'maxiter':130,'ftol':1e-7})
    if not res.success: return np.nan
    omega,alpha,beta=res.x
    h=v
    for x in r[:-1]:h=omega+alpha*x*x+beta*h
    return max((omega+alpha*r[-1]**2+beta*h)/10000,EPS)

def fit_predict(train,test):
    X=train[FEATURES];Xt=test[FEATURES];y=train.y.to_numpy()
    out={'RV20':test.rv20.to_numpy(),'EWMA':test.ewma.to_numpy()}
    rf=RandomForestRegressor(n_estimators=120,max_depth=6,min_samples_leaf=12,n_jobs=-1,random_state=SEED)
    rf.fit(X,y);out['RF']=np.maximum(rf.predict(Xt),EPS)
    # log-link prevents negative predictions and reduces extreme squared-return influence
    hgb=HistGradientBoostingRegressor(max_iter=130,max_leaf_nodes=9,l2_regularization=2,random_state=SEED)
    hgb.fit(X,np.log(np.maximum(y,EPS)))
    out['HGB_log']=np.maximum(np.exp(hgb.predict(Xt)),EPS)
    return out

def losses(y,p):
    y=np.asarray(y);p=np.maximum(np.asarray(p),EPS)
    return np.log(p)+y/p,(y-p)**2

def bootstrap_diff(a,b,B=1200,block=10,seed=SEED):
    d=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(seed);n=len(d)
    vals=np.empty(B)
    for i in range(B):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        ix=np.concatenate([(np.arange(s,s+block)%n) for s in starts])[:n]
        vals[i]=d[ix].mean()
    return float(d.mean()),*map(float,np.quantile(vals,[.025,.975]))

def kupiec(exceed,p0=.01):
    n=len(exceed);x=int(np.sum(exceed));ph=np.clip(x/n,1e-12,1-1e-12)
    ll0=x*np.log(p0)+(n-x)*np.log(1-p0)
    ll1=x*np.log(ph)+(n-x)*np.log(1-ph)
    return x,n,float(chi2.sf(2*(ll1-ll0),1))

def run(market,path,out):
    # prepare's date is origin t; target is squared return t+1
    x=prepare(path)
    raw=pd.read_csv(path);raw['Date']=pd.to_datetime(raw.Date);raw=raw.sort_values('Date')
    ret=pd.Series(np.log(raw.Close.to_numpy()).astype(float)).diff().to_numpy()
    date_to_idx={pd.Timestamp(d):i for i,d in enumerate(raw.Date)}
    # 3 expanding folds; each test block ~10% of history; refit once per fold
    n=len(x);bounds=[int(n*z) for z in [.70,.80,.90,1.00]]
    rows=[]
    for fold,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:]),1):
        train=x.iloc[:lo];test=x.iloc[lo:hi];pred=fit_predict(train,test)
        # GARCH re-estimated every 20 forecast origins, updated conditional variance
        # Strictly use historical returns up to origin, never the following session
        g=[]
        for origin in test.date:
            idx=date_to_idx[pd.Timestamp(origin)]
            # expensive full refit per point; do a sparse diagnostic on first 30 points only
            if len(g)<30:g.append(garch_forecast(ret[1:idx+1]))
            else:g.append(np.nan)
        pred['GARCH_diagnostic']=np.asarray(g)
        # actual signed next return for VaR backtest
        signed=np.array([ret[date_to_idx[pd.Timestamp(d)]+1] for d in test.date])
        for model,p in pred.items():
            for i,(dt,y,pp,r) in enumerate(zip(test.date,test.y,p,signed)):
                if np.isfinite(pp):rows.append({'market':market,'fold':fold,'date':dt,'model':model,'y':y,'pred_var':pp,'next_return':r})
    result=pd.DataFrame(rows);result.to_csv(out/f'{market}_walkforward_predictions.csv',index=False)
    summary=[]
    for (fold,model),g in result.groupby(['fold','model']):
        q,m=losses(g.y,g.pred_var)
        exceptions=g.next_return.to_numpy() < norm.ppf(.01)*np.sqrt(g.pred_var.to_numpy())
        exc,nobs,pval=kupiec(exceptions)
        summary.append({'market':market,'fold':fold,'model':model,'n':len(g),'QLIKE':q.mean(),'MSE':m.mean(),'VaR_1pct_exceptions':exc,'VaR_expected':nobs*.01,'Kupiec_p':pval})
    summary=pd.DataFrame(summary);summary.to_csv(out/f'{market}_walkforward_scores.csv',index=False)
    comparisons=[]
    for fold in [1,2,3]:
        f=result[result.fold==fold]
        baseline=f[f.model=='EWMA'].set_index('date')
        for model in ['RV20','RF','HGB_log']:
            comp=f[f.model==model].set_index('date').join(baseline[['pred_var']],rsuffix='_base')
            q,_=losses(comp.y,comp.pred_var);qb,_=losses(comp.y,comp.pred_var_base)
            diff,low,high=bootstrap_diff(q,qb,seed=SEED+fold)
            comparisons.append({'market':market,'fold':fold,'model':model,'QLIKE_diff_vs_EWMA':diff,'bootstrap_CI_low':low,'bootstrap_CI_high':high,'CI_excludes_zero':bool(low>0 or high<0)})
    pd.DataFrame(comparisons).to_csv(out/f'{market}_bootstrap.csv',index=False)
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--sp500',default='data/sp500.csv');p.add_argument('--tasi',default='data/tasi.csv');p.add_argument('--out',default='outputs_v5');a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    for market,path in [('SP500',a.sp500),('TASI',a.tasi)]:
        s=run(market,path,out);print(market);print(s.to_string(index=False))
