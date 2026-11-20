"""Regenerate fixed-specification 2026 evaluation plus HAR-SR2 on exact archived prices.

Inputs: complete 2016-2026 Date,Close CSVs, one per market. This script does NOT
fetch/reconstruct historical prices, nor infer origin losses from published means.
The original source CSVs are required for exact reproduction.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from study2_extension import run_inference, EPS

SEED=2026
FEATURES=['abs1','abs2','abs5','rv5','rv20','rv60','neg','ewma','logrv20']
CUTOFF=pd.Timestamp('2026-01-22')
LAST_TARGET=pd.Timestamp('2026-10-30')
# The exact reports were computed from archival market closes with these lengths.
EXPECTED={'SP500':{'historical_prices':2528,'historical_origins':2467,'train':2466,'later_origins':195},
          'TASI': {'historical_prices':2513,'historical_origins':2452,'train':2451,'later_origins':188}}
PUBLISHED_2026={'SP500':{'Log target':-5.1597,'Rescaled log':-8.4697,'Poisson':-8.5879,
                        'Direct mean':-8.5784,'EWMA':-8.5816},
                'TASI':{'Log target':-6.6184,'Rescaled log':-8.6471,'Poisson':-8.7084,
                        'Direct mean':9336.0379,'EWMA':-8.7774}}

def file_sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as stream:
        for x in iter(lambda:stream.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def prepare(csv_path):
    raw=pd.read_csv(csv_path)
    if not {'Date','Close'}.issubset(raw.columns):raise ValueError('Date and Close columns required')
    raw['Date']=pd.to_datetime(raw.Date,errors='raise')
    raw['Close']=pd.to_numeric(raw.Close,errors='raise')
    if raw.Date.isna().any() or raw.Date.duplicated().any() or raw.Close.isna().any() or (raw.Close<=0).any() or not np.isfinite(raw.Close).all():
        raise ValueError('Invalid market prices; require unique dates and positive finite closes')
    raw=raw.sort_values('Date').reset_index(drop=True)
    ret=np.log(raw.Close).diff();sq=ret.pow(2)
    x=pd.DataFrame({'date':raw.Date,'target_date':raw.Date.shift(-1),'y':sq.shift(-1)})
    for lag in [0,1,4]:x[f'abs{lag+1}']=ret.abs().shift(lag)
    for days in [5,20,60]:x[f'rv{days}']=sq.rolling(days).mean()
    x['neg']=(ret<0).astype(float)
    x['ewma']=sq.ewm(alpha=.06,adjust=False).mean()
    x['logrv20']=np.log(x.rv20.clip(lower=EPS))
    x['har_daily']=sq
    x['har_weekly']=sq.rolling(5).mean()
    x['har_monthly']=sq.rolling(22).mean()
    x=x.dropna().reset_index(drop=True)
    return raw,x

def learner(loss='squared_error'):
    return HistGradientBoostingRegressor(loss=loss,max_iter=130,max_leaf_nodes=9,
                    l2_regularization=2,random_state=SEED,early_stopping=False)

def fit_log(training):
    est=learner();est.fit(training[FEATURES],np.log(np.maximum(training.y.to_numpy(float),EPS)))
    return est

def log_predict(est,frame):
    return np.maximum(np.exp(est.predict(frame[FEATURES])),EPS)

def estimate_multiplier(training):
    n=len(training)
    cuts=(int(.6*n),int(.8*n),n)
    actual=[];fit=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        inner_train=training.iloc[:lo-1]  # keep one origin out before evaluation block
        inner_val=training.iloc[lo:hi]
        assert inner_train.target_date.max()<inner_val.date.min(), 'Inner split leakage'
        pred=log_predict(fit_log(inner_train),inner_val)
        actual.append(inner_val.y.to_numpy(float));fit.append(pred)
    y=np.concatenate(actual);h=np.concatenate(fit)
    if len(y)<100 or not np.isfinite(y/h).all():raise ValueError('Invalid forward multiplier inputs')
    c=float(np.mean(y/h))
    if c<=0 or not np.isfinite(c):raise ValueError('Invalid log retransformation coefficient')
    return c

def fit_har(training,test):
    cols=['har_daily','har_weekly','har_monthly']
    beta=np.linalg.lstsq(np.c_[np.ones(len(training)),training[cols]],training.y.to_numpy(float),rcond=None)[0]
    prediction=np.maximum(np.c_[np.ones(len(test)),test[cols]]@beta,EPS)
    return prediction,beta

def process_market(market,path,allow_mismatch=False):
    raw,panel=prepare(path)
    historical=raw.loc[raw.Date<=CUTOFF]
    pre=panel.loc[panel.target_date<=CUTOFF].copy()
    train=pre.iloc[:-1]  # exclude origin 2026-01-21, for target 2026-01-22
    test=panel.loc[(panel.date>=CUTOFF)&(panel.target_date<=LAST_TARGET)].copy()
    assert len(test)>0, 'No 2026 evaluation dates found'
    assert train.target_date.max()<test.date.min(), 'Training uses future target'
    expected=EXPECTED[market]
    count={'historical_prices':len(historical),'historical_origins':len(pre),'train':len(train),'later_origins':len(test)}
    # Note pre includes archive origin Jan21/target Jan22, and should have 2467/2452 rows.
    # Source identity must be checked, not inferred from similar approximate dates.
    exact=count==expected
    if not exact and not allow_mismatch:
        raise ValueError(f'{market} archived dataset mismatch: {count} vs {expected}. '
             'Use exact source CSVs; --allow-mismatch runs an explicitly non-replication sensitivity.')
    c=estimate_multiplier(train)
    log=log_predict(fit_log(train),test)
    direct=learner('squared_error');direct.fit(train[FEATURES],train.y.to_numpy(float))
    # Unscaled target: reproduces published historical and 2026 Poisson QLIKE.
    # Scaling Y by 10,000 changes regularized Poisson boosting fits.
    pois=learner('poisson');pois.fit(train[FEATURES],train.y.to_numpy(float))
    har,beta=fit_har(train,test)
    preds={'Log target':log,'Rescaled log':np.maximum(c*log,EPS),
           'Poisson':np.maximum(pois.predict(test[FEATURES]),EPS),
           'Direct mean':np.maximum(direct.predict(test[FEATURES]),EPS),
           'EWMA':np.maximum(test.ewma.to_numpy(float),EPS),
           'HAR-SR2':har}
    out=[]
    for model,pred in preds.items():
        for date,td,y,p in zip(test.date,test.target_date,test.y,pred):
            out.append({'market':market,'fold':'2026','date':date.strftime('%Y-%m-%d'),
                  'target_date':td.strftime('%Y-%m-%d'),
                  'model':model,'target_sq_return':float(y),'pred_var':float(p)})
    results=pd.DataFrame(out)
    scores={m:float(np.mean(np.log(preds[m])+test.y.to_numpy(float)/preds[m])) for m in preds}
    deviations={m:float(scores[m]-v) for m,v in PUBLISHED_2026[market].items()}
    if exact and any(abs(d)>0.0001 for d in deviations.values()):
        # Deliberately do NOT silently claim the new source exactly reproduces 2026.
        exact=False
    return results,{'market':market,'input_file':str(path),'sha256':file_sha256(path),
                    'counts':count,'n_unique_2026_targets':test.target_date.nunique(),
                    'start_origin':str(test.date.iloc[0].date()),'end_target':str(test.target_date.iloc[-1].date()),
                    'fixed_training_cutoff':str(train.date.iloc[-1].date()),
                    'multiplier':c,'har_coefficients':beta.tolist(),'poisson_training_target_scale':1.0,
                    'observed_QLIKE':scores,'deviation_from_reported_QLIKE':deviations,
                    'published_Qlike_replication_check_passed':exact}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sp500-prices',required=True)
    ap.add_argument('--tasi-prices',required=True)
    ap.add_argument('--out-dir',default='results_study2_2026')
    ap.add_argument('--allow-mismatch',action='store_true')
    ap.add_argument('--replicates',type=int,default=4000)
    ap.add_argument('--block',type=int,default=10)
    a=ap.parse_args();o=Path(a.out_dir);o.mkdir(parents=True,exist_ok=True)
    results=[];metadata=[]
    for name,path in [('SP500',a.sp500_prices),('TASI',a.tasi_prices)]:
        x,meta=process_market(name,path,a.allow_mismatch)
        results.append(x);metadata.append(meta)
    pd.concat(results,ignore_index=True).to_csv(o/'later_2026_predictions_including_HAR.csv',index=False)
    (o/'source_replication_checks.json').write_text(json.dumps(metadata,indent=2))
    # The inferential runner uses the same dates, same outcomes for all six models.
    allrows=pd.concat(results,ignore_index=True)
    scores,mcs=run_inference(allrows,o/'inference',replicates=a.replicates,block=a.block)
    # Automatic sensitivity to serial-dependence block size.
    for alternative in [5,10,20]:
        if alternative!=a.block:
            run_inference(allrows,o/f'inference_block_{alternative}',replicates=a.replicates,block=alternative)
    # The TASI direct-mean model is floor-pathological in the historical report;
    # keep the full 6-model MCS, and provide sensitivity to its inclusion.
    subset=allrows.loc[allrows.model!='Direct mean']
    run_inference(subset,o/'inference_excluding_direct_mean',replicates=a.replicates,block=a.block)
    print(json.dumps(metadata,indent=2))
    print(scores.to_string(index=False))
    print(mcs.to_string(index=False))
    if not all(m['published_Qlike_replication_check_passed'] for m in metadata):
        print('WARNING: New data do not exactly reproduce source; mark these as sensitivity results.')
if __name__=='__main__':main()
