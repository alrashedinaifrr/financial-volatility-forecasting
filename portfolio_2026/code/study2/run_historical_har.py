"""Reproduce archived five-model forecast protocol from supplied historical index closes;
add HAR-SR2 on identical forecast origins. No observations after 2026-01-22 used.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from scipy.optimize import minimize, nnls

ROOT=Path(__file__).resolve().parent
EPS=1e-9
FEATURES=['abs1','abs2','abs5','rv5','rv20','rv60','neg','ewma','logrv20']
PUBLISHED={
 'SP500':[[-8.6829,-8.6578,-8.6033,-8.6647,-5.4486],[-8.5915,-8.6069,-8.5986,-8.4787,-5.4383],[-8.2092,-8.3361,-8.3458,-8.1485,-6.4115]],
 'TASI':[[-8.8909,-8.8989,-8.8856,-8.8656,-7.3704],[-8.9573,-8.9523,-8.8955,-8.8735,-6.9971],[-8.1680,-8.4717,-8.4670,-8.1045,-6.2092]]}
REF=['EWMA','GARCH','RF','RV20','HGB_log']

def prepare(path):
 d=pd.read_csv(path);d['Date']=pd.to_datetime(d.Date);d=d.sort_values('Date').reset_index(drop=True)
 r=np.log(d.Close).diff();sq=r*r
 x=pd.DataFrame({'date':d.Date,'target_date':d.Date.shift(-1),'y':sq.shift(-1)})
 for lag in [0,1,4]:x[f'abs{lag+1}']=r.abs().shift(lag)
 for window in [5,20,60]:x[f'rv{window}']=sq.rolling(window).mean()
 x['neg']=(r<0).astype(float);x['ewma']=sq.ewm(alpha=.06,adjust=False).mean()
 x['logrv20']=np.log(x.rv20.clip(lower=EPS))
 x['har_daily']=sq;x['har_weekly']=sq.rolling(5).mean();x['har_monthly']=sq.rolling(22).mean()
 return d,x.dropna().reset_index(drop=True)

def fit_garch(ret):
 r=np.asarray(ret,dtype=float)*100
 v=max(float(np.var(r)),1e-5)
 def filtering(theta,seq):
  w,a,b=theta;h=np.empty(len(seq));h[0]=v
  for t in range(1,len(seq)):h[t]=w+a*seq[t-1]**2+b*h[t-1]
  return h
 def obj(z):
  w,a,b=z
  if a+b>=.995:return 1e9
  h=filtering(z,r)
  return float(np.mean(np.log(h)+r*r/h))
 fit=minimize(obj,[v*.04,.08,.88],method='SLSQP',bounds=[(1e-8,max(v*10,1)),(1e-5,.6),(.01,.995)],
              constraints=[{'type':'ineq','fun':lambda z:.995-z[1]-z[2]}], options={'maxiter':250,'ftol':1e-9})
 if not fit.success or not np.isfinite(fit.fun):raise RuntimeError(f'GARCH optimization failed: {fit.message}')
 return fit.x,filtering(fit.x,r)[-1]

def run(market,path):
 raw,x=prepare(path); n=len(x);bounds=[int(n*z) for z in [.70,.80,.90,1.0]]
 orig_idx={v:i for i,v in enumerate(raw.Date)};ret=np.log(raw.Close.to_numpy()).astype(float);ret=np.diff(ret,prepend=np.nan)
 recs=[];configs=[]
 for fold,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:]),1):
  t=time.time()
  train=x.iloc[:lo-1];test=x.iloc[lo:hi]
  assert train.target_date.max()<test.date.min()
  X=train[FEATURES];Xt=test[FEATURES]
  rf=RandomForestRegressor(n_estimators=120,max_depth=6,min_samples_leaf=12,n_jobs=-1,random_state=2026)
  rf.fit(X,train.y)
  hgb=HistGradientBoostingRegressor(max_iter=130,max_leaf_nodes=9,l2_regularization=2,random_state=2026)
  hgb.fit(X,np.log(np.maximum(train.y,EPS)))
  p={'RV20':test.rv20.to_numpy(), 'EWMA':test.ewma.to_numpy(),
     'RF':np.maximum(rf.predict(Xt),EPS),
     'HGB_log':np.maximum(np.exp(hgb.predict(Xt)),EPS)}
  first=orig_idx[train.date.iloc[-1]];theta,h=fit_garch(ret[1:first+1]);h=theta[0]+theta[1]*(100*ret[first])**2+theta[2]*h
  g=[];prev=first
  for date in test.date:
   idx=orig_idx[date]
   for k in range(prev+1,idx+1):h=theta[0]+theta[1]*(100*ret[k])**2+theta[2]*h
   g.append(max(h/10000,EPS));prev=idx
  p['GARCH']=np.asarray(g)
  cols=['har_daily','har_weekly','har_monthly']
  F=np.c_[np.ones(len(train)),train[cols].to_numpy(float)];Ft=np.c_[np.ones(len(test)),test[cols].to_numpy(float)]
  beta=np.linalg.lstsq(F,train.y.to_numpy(float),rcond=None)[0]
  har_raw=Ft@beta; p['HAR_SR2']=np.maximum(har_raw,EPS)
  # Positive-coefficient robustness: intercept and three lag coefficients constrained nonnegative
  beta_nnls=nnls(F,train.y.to_numpy(float))[0]
  p['HAR_NNLS']=np.maximum(Ft@beta_nnls,EPS)
  for model,pred in p.items():
   pred=np.maximum(pred,EPS)
   for origin,target_date,y,forecast in zip(test.date,test.target_date,test.y,pred):
    recs.append({'market':market,'fold':str(fold),'date':str(origin.date()),'target_date':str(target_date.date()),'model':model,'target_sq_return':float(y),'pred_var':float(forecast)})
  scores={name:float(np.mean(np.log(np.maximum(pred,EPS))+test.y.to_numpy()/np.maximum(pred,EPS))) for name,pred in p.items()}
  reference={k:PUBLISHED[market][fold-1][i] for i,k in enumerate(REF)}
  differences={k:scores[k]-v for k,v in reference.items()}
  configs.append({'market':market,'fold':fold,'train_n':len(train),'test_n':len(test),'train_last_origin':str(train.date.iloc[-1].date()),
                  'test_first_origin':str(test.date.iloc[0].date()),'test_last_origin':str(test.date.iloc[-1].date()),
                  'HAR_coefficients':beta.tolist(),'HAR_NNLS_coefficients':beta_nnls.tolist(),
                  'HAR_nonpositive_unfloored':int((har_raw<=0).sum()),'garch_parameters':theta.tolist(),
                  'scores':scores,'published_diffs':differences})
  print(market,fold,'n',len(train),len(test),'HAR floor',int((har_raw<=0).sum()),'delta',differences,'seconds',round(time.time()-t,1),flush=True)
 return recs,configs

def main():
 (ROOT/'output').mkdir(parents=True,exist_ok=True)
 rows=[];metadata=[]
 for market,name in [('SP500','sp500.csv'),('TASI','tasi.csv')]:
  a,b=run(market,ROOT/'input'/name);rows+=a;metadata+=b
 pd.DataFrame(rows).to_csv(ROOT/'output'/'archive_predictions_with_HAR.csv',index=False)
 (ROOT/'output'/'archive_replication.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
 print('done rows=',len(rows))
if __name__=='__main__':main()
