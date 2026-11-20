"""Audited chronological addition of GARCH(1,1), random forest, RV20 to 2026.

Uses the exact historical implementations in `run_historical_har.py` and exactly the
same training cutoff, feature definitions and forecast origins in
`run_2026_corrected.py`. Does not replace any published six-model result.
"""
from __future__ import annotations
import sys,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT=Path(__file__).resolve().parents[2]
from run_2026_corrected import prepare, CUTOFF,LAST_TARGET,FEATURES
from run_historical_har import fit_garch

BASE=ROOT/'results/later_2026_predictions_including_HAR.csv'
PRICES=ROOT/'data/finance/complete'
OUT=ROOT/'results';OUT.mkdir(parents=True,exist_ok=True)
original=pd.read_csv(BASE,dtype={'date':str,'target_date':str,'model':str})
added=[];audit=[]
for market,name in [('SP500','sp500.csv'),('TASI','tasi.csv')]:
    raw,panel=prepare(PRICES/name)
    pre=panel[panel.target_date<=CUTOFF].copy()
    train=pre.iloc[:-1]
    test=panel[(panel.date>=CUTOFF)&(panel.target_date<=LAST_TARGET)].copy()
    old=original[original.market==market]
    assert len(test)=={'SP500':195,'TASI':188}[market]
    assert len(train)=={'SP500':2466,'TASI':2451}[market]
    assert set(test.date.dt.strftime('%Y-%m-%d'))==set(old.date)
    assert np.max(np.abs(test.y.to_numpy()-old[old.model=='EWMA'].sort_values('date').target_sq_return.to_numpy()))<1e-15
    assert train.target_date.max()<test.date.min()
    rf=RandomForestRegressor(n_estimators=120,max_depth=6,min_samples_leaf=12,n_jobs=-1,random_state=2026)
    rf.fit(train[FEATURES],train.y.to_numpy(float))
    rf_p=np.maximum(rf.predict(test[FEATURES]),1e-9)
    rv20_p=np.maximum(test.rv20.to_numpy(float),1e-9)
    # Exactly the same GARCH initialization and recursion as original historical folds.
    index={v:i for i,v in enumerate(raw.Date)}
    returns=np.diff(np.log(raw.Close.to_numpy(float)),prepend=np.nan)
    first=index[train.date.iloc[-1]]
    theta,h=fit_garch(returns[1:first+1])
    h=theta[0]+theta[1]*(100*returns[first])**2+theta[2]*h
    prev=first;gg=[]
    for date in test.date:
        idx=index[date]
        for k in range(prev+1,idx+1):
            h=theta[0]+theta[1]*(100*returns[k])**2+theta[2]*h
        gg.append(max(h/10000,1e-9));prev=idx
    preds={'RF':rf_p,'GARCH':np.asarray(gg),'RV20':rv20_p}
    scores={}
    for model,p in preds.items():
        assert len(p)==len(test) and np.isfinite(p).all() and np.all(p>0)
        scores[model]=float(np.mean(np.log(p)+test.y.to_numpy()/p))
        expected_targets=old[old.model=='EWMA'].set_index('date').target_sq_return
        for date,target,y,var in zip(test.date,test.target_date,test.y,p):
            y=float(expected_targets.loc[date.strftime('%Y-%m-%d')])
            added.append({'market':market,'fold':'2026','date':date.strftime('%Y-%m-%d'),
                'target_date':target.strftime('%Y-%m-%d'),'model':model,
                'target_sq_return':float(y),'pred_var':float(var)})
    audit.append({'market':market,'first_2026_origin':str(test.date.iloc[0].date()),
        'last_2026_target':str(test.target_date.iloc[-1].date()),'train_n':len(train),
        'evaluation_n':len(test),'rf_n_estimators':120,'rf_depth':6,'rf_min_samples_leaf':12,
        'rf_seed':2026,'garch_omega_alpha_beta':theta.tolist(),
        'garch_training_last_origin':str(train.date.iloc[-1].date()),
        'forecast_scores':scores})
    print(market,scores,flush=True)

allrows=pd.concat([original,pd.DataFrame(added)],ignore_index=True)
assert not allrows.duplicated(['market','fold','date','model']).any()
models=sorted(allrows.model.unique())
assert len(models)==9,models
for market,g in allrows.groupby('market'):
    pivot=g.pivot(index='date',columns='model',values='pred_var')
    assert len(pivot)=={'SP500':195,'TASI':188}[market] and pivot.shape[1]==9 and not pivot.isna().any().any()
    y=g.groupby('date').target_sq_return.nunique()
    assert y.max()==1
allrows.to_csv(OUT/'2026_nine_model_predictions.csv',index=False)
(OUT/'2026_added_model_audit.json').write_text(json.dumps(audit,indent=2))
pd.DataFrame([{'market':r['market'],'model':k,'QLIKE':v} for r in audit for k,v in r['forecast_scores'].items()]).to_csv(OUT/'2026_added_model_scores.csv',index=False)
print('done',len(allrows),'rows',models,flush=True)
