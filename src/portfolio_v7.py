from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
dst=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(dst/'src'))
from benchmark import prepare, FEATURES, EPS
from advanced_validation import fit_predict, losses, bootstrap_diff
from full_garch_validation import fit_garch
out=dst/'results_v9';out.mkdir(exist_ok=True)
summary=[];prediction_rows=[];transfers=[]
for market in ['SP500','TASI']:
 raw=pd.read_csv(dst/'data'/f'{market.lower()}.csv');raw['Date']=pd.to_datetime(raw.Date);raw=raw.sort_values('Date').reset_index(drop=True)
 x=prepare(dst/'data'/f'{market.lower()}.csv');r=np.log(raw.Close.to_numpy());r=np.diff(r,prepend=np.nan);lookup={d:i for i,d in enumerate(raw.Date)}
 bounds=[int(len(x)*z) for z in [.7,.8,.9,1.0]]
 for fold,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:]),1):
  train=x.iloc[:lo-1];test=x.iloc[lo:hi];pred=fit_predict(train,test)
  first=lookup[train.date.iloc[-1]];theta,h=fit_garch(r[1:first+1]);h=theta[0]+theta[1]*(100*r[first])**2+theta[2]*h;g=[];prev_idx=first
  for d in test.date:
   idx=lookup[d]
   for k in range(prev_idx+1,idx+1):h=theta[0]+theta[1]*(100*r[k])**2+theta[2]*h
   g.append(max(h/10000,EPS));prev_idx=idx
  pred['GARCH']=np.array(g)
  for name,p in pred.items():
   q,m=losses(test.y,p);summary.append({'market':market,'fold':fold,'model':name,'n':len(test),'QLIKE':float(np.mean(q)),'MSE':float(np.mean(m))})
   for dt,y,pr in zip(test.date,test.y,p):prediction_rows.append({'market':market,'fold':fold,'date':dt,'model':name,'target_sq_return':y,'pred_var':pr})
summary=pd.DataFrame(summary);summary.to_csv(out/'fold_scores.csv',index=False)
pd.DataFrame(prediction_rows).to_csv(out/'predictions.csv',index=False)
# Cross-market transfer: identical destination evaluation origins; source train ends before first destination origin and excludes unavailable last target.
for source,destination in [('SP500','TASI'),('TASI','SP500')]:
 a=prepare(dst/'data'/f'{source.lower()}.csv');b=prepare(dst/'data'/f'{destination.lower()}.csv');cut=b.date.iloc[int(.8*len(b))]
 a=a[a.date < cut].iloc[:-1];bt=b[b.date>=cut]
 # Compare source-trained with destination-trained on identical dates and with the same hyperparameters
 local=b[b.date<cut].iloc[:-1]
 for training,training_market in [(a,source),(local,destination)]:
  model=RandomForestRegressor(n_estimators=120,max_depth=6,min_samples_leaf=12,n_jobs=-1,random_state=2026)
  model.fit(training[FEATURES],training.y)
  p=np.maximum(model.predict(bt[FEATURES]),EPS);q,m=losses(bt.y,p)
  transfers.append({'source':training_market,'destination':destination,'evaluation_start':str(bt.date.iloc[0].date()),'evaluation_end':str(bt.date.iloc[-1].date()),'n':len(bt),'QLIKE':float(q.mean()),'MSE':float(m.mean())})
transfers=pd.DataFrame(transfers);transfers.to_csv(out/'aligned_transfer.csv',index=False)
# CIs for RF - EWMA, each fold; bootstrap descriptive, no multiple-test adjustment
preds=pd.DataFrame(prediction_rows);ci=[]
for (market,fold),g in preds.groupby(['market','fold']):
 pivot=g.pivot(index='date',columns='model',values='pred_var');y=g.drop_duplicates('date').set_index('date').loc[pivot.index,'target_sq_return']
 qa,_=losses(y,pivot['RF']);qb,_=losses(y,pivot['EWMA']);d,l,u=bootstrap_diff(qa,qb,B=1500,block=10,seed=2026+fold)
 ci.append({'market':market,'fold':fold,'RF_minus_EWMA':d,'CI95_low':l,'CI95_high':u})
pd.DataFrame(ci).to_csv(out/'rf_vs_ewma_bootstrap.csv',index=False)
# visualizations
for market in ['SP500','TASI']:
 t=summary[summary.market==market].pivot(index='fold',columns='model',values='QLIKE')
 fig,ax=plt.subplots(figsize=(9,5));t[['EWMA','GARCH','RF','HGB_log','RV20']].plot(ax=ax,marker='o');ax.set_title(f'{market}: expanding-window out-of-sample QLIKE');ax.set_xlabel('Evaluation fold');ax.set_ylabel('Mean QLIKE (lower is better)');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'{market}_fold_qlike.png',dpi=170);plt.close(fig)
