"""Study II: HAR squared-return benchmark; paired 2026 uncertainty; MCS.

Inputs:
  --historical-forecasts ... : original repo's results_final/predictions.csv (long)
  --later-forecasts ... : 2026 origin-level forecasts in the same schema;
      market,fold,date,model,target_sq_return,pred_var. A 'fold' can be '2026'.
  --sp500-prices ... --tasi-prices ... : Date,Close historical+later prices,
      optionally permits HAR forecasts on aligned origin dates.

The 2026 published *means* are not suitable for DM, block bootstrap or MCS;
per-origin observations are required. Never reverse-engineer daily losses from
summary tables. The HAR-style benchmark uses daily squared returns rather than
intraday realized variance, and must not be called HAR-RV.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import norm

EPS=1e-9
ORIGINAL=['RV20','EWMA','GARCH','RF','HGB_log']
LATER=['Log target','Rescaled log','Poisson','Direct mean','EWMA']


def har_panel(price_file,history_end=None):
    """Forecast origin t uses data through t, predicting (log P[t+1]/P[t])^2."""
    raw=pd.read_csv(price_file)
    if not {'Date','Close'}.issubset(raw):raise ValueError('Missing Date,Close')
    raw['Date']=pd.to_datetime(raw['Date'],errors='raise')
    raw['Close']=pd.to_numeric(raw['Close'],errors='raise')
    if raw.Date.isna().any() or raw.Date.duplicated().any() or not (raw.Close>0).all():
        raise ValueError('Invalid prices or dates')
    raw=raw.sort_values('Date').reset_index(drop=True)
    if history_end is not None:
        raw=raw.loc[raw.Date<=pd.Timestamp(history_end)].reset_index(drop=True)
    s=np.log(raw.Close).diff().pow(2)
    panel=pd.DataFrame({'date':raw.Date,'y':s.shift(-1),'rv1':s,
                        'rv5':s.rolling(5).mean(),'rv22':s.rolling(22).mean(),
                        'rv60_for_alignment':s.rolling(60).mean()})
    panel['target_date']=raw.Date.shift(-1)
    return panel.dropna().reset_index(drop=True)


def har_fit_predict(train,test):
    """OLS-HAR with positivity floor applied only to issued predictions."""
    cols=['rv1','rv5','rv22']
    if len(train)<100:raise ValueError('At least 100 HAR training origins required')
    X=np.c_[np.ones(len(train)),train[cols].to_numpy(float)]
    y=train['y'].to_numpy(float)
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    Xtest=np.c_[np.ones(len(test)),test[cols].to_numpy(float)]
    forecast=np.maximum(Xtest@beta,EPS)
    if not np.isfinite(forecast).all():raise ValueError('Invalid HAR forecast')
    return forecast, beta


def har_forecast_rows(prices,market,cutoff='2026-01-22'):
    """Historical folds + later evaluation; preserve 1-origin boundary embargo.

    Historical fractions are based on the same 60-lag panel as benchmark.py.
    Later fit excludes Jan 21 2026 origin as original study does; first later
    forecast originates at the Jan 22 close and predicts next target session.
    """
    pre=har_panel(prices,cutoff)
    bounds=[int(len(pre)*x) for x in [.70,.80,.90,1.0]]
    frames=[];coeff=[]
    for fold,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:]),1):
        tr=pre.iloc[:lo-1];te=pre.iloc[lo:hi]
        pred,b=har_fit_predict(tr,te)
        frames.append(pd.DataFrame({'market':market,'fold':str(fold),
            'date':te.date.dt.strftime('%Y-%m-%d'),'model':'HAR_SR2',
            'target_sq_return':te.y,'pred_var':pred}))
        coeff.append({'market':market,'fold':str(fold),'train_n':len(tr),'test_n':len(te),
                      'intercept':float(b[0]),'rv1_coef':float(b[1]),
                      'rv5_coef':float(b[2]),'rv22_coef':float(b[3])})
    later=har_panel(prices)
    cut=pd.Timestamp(cutoff)
    # Date prior to forecast's first origin is skipped to match the original
    # paper's explicit exclusion of Jan 21 2026. Historical training uses
    # origin <= Jan 20 2026.
    tr=later.loc[later.date < (cut-pd.Timedelta(days=1))]
    te=later.loc[later.date >= cut]
    if len(te):
        pred,b=har_fit_predict(tr,te)
        frames.append(pd.DataFrame({'market':market,'fold':'2026',
            'date':te.date.dt.strftime('%Y-%m-%d'),'model':'HAR_SR2',
            'target_sq_return':te.y,'pred_var':pred}))
        coeff.append({'market':market,'fold':'2026','train_n':len(tr),'test_n':len(te),
                      'intercept':float(b[0]),'rv1_coef':float(b[1]),
                      'rv5_coef':float(b[2]),'rv22_coef':float(b[3])})
    return pd.concat(frames,ignore_index=True),pd.DataFrame(coeff)


def qlike(y,h,eps=EPS):
    y=np.asarray(y,float);h=np.asarray(h,float)
    if (y<0).any() or not np.isfinite(y).all() or not np.isfinite(h).all():
        raise ValueError('Invalid proxy or forecast')
    if (h<=0).any():raise ValueError('Forecast must be strictly positive')
    h=np.maximum(h,eps)
    return np.log(h)+y/h


def long_to_panel(df):
    """Require one forecast per market,fold,origin,model and matched targets."""
    need={'market','date','model','target_sq_return','pred_var'}
    if need-set(df):raise ValueError('Missing columns: '+str(need-set(df)))
    frame=df.copy();frame['fold']=frame['fold'].astype(str) if 'fold' in frame else '2026'
    keys=['market','fold','date','model']
    if frame.duplicated(keys).any():raise ValueError('Duplicate model/forecast origin')
    sizes=frame.groupby(['market','fold','date'])['target_sq_return'].nunique()
    if (sizes>1).any():raise ValueError('Same origin must have same realized target')
    if (frame.pred_var<=0).any() or (frame.target_sq_return<0).any():
        raise ValueError('Nonpositive forecasts or invalid target')
    return frame


def aligned_matrix(df, market,fold,models):
    v=df[(df.market==market)&(df.fold==str(fold))]
    pivot=v.pivot(index='date',columns='model',values='pred_var')
    missing=set(models)-set(pivot)
    if missing:raise ValueError(f'Missing models for {market}/{fold}: {missing}')
    pivot=pivot[models].dropna()
    target=v.drop_duplicates('date').set_index('date')['target_sq_return'].loc[pivot.index]
    # Alignment is an explicit, auditable intersection, no different dates
    # for different models within the same comparison.
    if target.isna().any():raise ValueError('Missing evaluation targets')
    return target.to_numpy(float),pivot.to_numpy(float),list(pivot.index)


def circular_block_indices(n,block,rng):
    """Index bootstrap: time-unit blocks preserve within-series dependence."""
    starts=rng.integers(0,n,size=int(np.ceil(n/block)))
    return (starts[:,None]+np.arange(block)[None,:]).ravel()[:n]%n


def pairwise_inference(losses,models,replicates=4000,block=10,seed=2026,lag=10):
    """Pointwise paired bootstrap CI + asymptotic DM/HAC, corrected across pairs.

    DM is used as a descriptive equal predictive accuracy test for one-step
    losses. Bartlett Newey-West HAC and fixed integer lag are recorded; the
    finite-sample normal approximation may be poor with volatile QLIKE tails.
    """
    losses=np.asarray(losses,float)
    if losses.ndim!=2 or len(losses)<max(40,2*block):raise ValueError('Insufficient losses')
    n,k=losses.shape;rng=np.random.default_rng(seed)
    boot=np.empty((replicates,k))
    for i in range(replicates):
        idx=circular_block_indices(n,block,rng)
        boot[i]=losses[idx].mean(axis=0)
    rows=[]
    for i,j in combinations(range(k),2):
        d=losses[:,i]-losses[:,j];mean=d.mean(); centered=d-mean
        gamma0=np.dot(centered,centered)/n
        L=min(lag,n-1);lrv=gamma0
        for h in range(1,L+1):
            gam=np.dot(centered[h:],centered[:-h])/n
            lrv+=2*(1-h/(L+1))*gam
        se=np.sqrt(max(lrv,0)/n)
        z=mean/se if se>0 else float('inf')*np.sign(mean)
        pv=2*norm.sf(abs(z))
        bootdiff=boot[:,i]-boot[:,j];ci=np.quantile(bootdiff,[.025,.975]);
        rows.append({'model_a':models[i],'model_b':models[j],
                     'mean_qlike_a_minus_b':float(mean),'n':n,
                     'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),
                     'DM_HAC_z':float(z),'DM_p_two_sided':float(pv),
                     'HAC_lags':int(L),'bootstrap_block':int(block)})
    # Holm step-down for the family represented by this call
    p=np.array([x['DM_p_two_sided'] for x in rows]);order=np.argsort(p)
    corrected=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]).clip(0,1)
    for position,idx in enumerate(order):rows[idx]['DM_p_Holm_within_market']=float(corrected[position])
    return pd.DataFrame(rows)


def model_confidence_set(losses,models,alpha=.05,replicates=4000,block=10,seed=2026):
    """Sequential MCS with T_max, centering bootstrap loss differentials.

    At each stage d_it=L_it - (1/m)sum_j L_jt. T_max is maximum studentized
    mean d_i. Null statistics center resampled means on original d_i;
    elimination is worst positive T_i. This is a *within-market/window*
    confidence set; comparison across blocks is not a simultaneous claim.
    """
    X=np.asarray(losses,float)
    if X.ndim!=2 or X.shape[1]!=len(models) or X.shape[0]<max(40,2*block):
        raise ValueError('Invalid MCS loss panel')
    selected=np.arange(len(models));history=[];rng=np.random.default_rng(seed)
    n=len(X)
    while len(selected)>1:
        panel=X[:,selected]
        D=panel-panel.mean(axis=1,keepdims=True)
        db=D.mean(axis=0)
        boot=np.empty((replicates,len(selected)))
        for b in range(replicates):
            ind=circular_block_indices(n,block,rng)
            boot[b]=D[ind].mean(axis=0)
        se=boot.std(axis=0,ddof=1)
        if np.any(se<1e-14):
            # Identical forecasters cannot be distinguished; no exclusion.
            history.append({'models':','.join(models[i] for i in selected),
                            'p_value':1.0,'excluded':'','tmax':0.0})
            break
        ti=db/se;observed=float(ti.max())
        centered=(boot-db)/se
        tstar=centered.max(axis=1)
        pv=float((1+(tstar>=observed).sum())/(1+replicates))
        worst=int(np.argmax(ti))
        excluded=models[selected[worst]] if pv<alpha else ''
        history.append({'models':','.join(models[i] for i in selected),
                        'p_value':pv,'excluded':excluded,'tmax':observed})
        if not excluded:break
        selected=np.delete(selected,worst)
    return [models[i] for i in selected],pd.DataFrame(history)


def run_inference(forecasts,outdir,replicates=4000,block=10):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    frame=long_to_panel(forecasts)
    allscores=[];allpairs=[];allmcs=[];stages=[]
    for (market,fold),sub in frame.groupby(['market','fold']):
        models=sorted(sub['model'].unique())
        if len(models)<2:continue
        y,h,dates=aligned_matrix(sub,market,fold,models)
        L=np.column_stack([qlike(y,h[:,j]) for j in range(h.shape[1])]);
        for j,model in enumerate(models):
            allscores.append({'market':market,'fold':fold,'model':model,'n':len(y),
                              'QLIKE':float(L[:,j].mean()),
                              'MSE':float(np.mean((y-h[:,j])**2))})
        pair=pairwise_inference(L,models,replicates,block,seed=2026,lag=10)
        pair.insert(0,'fold',fold);pair.insert(0,'market',market);allpairs.append(pair)
        retained,hist=model_confidence_set(L,models,replicates=replicates,block=block,seed=2026)
        allmcs.append({'market':market,'fold':fold,'n':len(y),'models_evaluated':','.join(models),
                       'mcs_retained':','.join(retained),'alpha':.05,'replicates':replicates,
                       'block':block})
        hist.insert(0,'fold',fold);hist.insert(0,'market',market);stages.append(hist)
    if not allscores:raise ValueError('No comparable model groups')
    pd.DataFrame(allscores).to_csv(out/'study2_model_scores.csv',index=False)
    pd.concat(allpairs).to_csv(out/'study2_pairwise_qlike_DM_bootstrap.csv',index=False)
    pd.DataFrame(allmcs).to_csv(out/'study2_model_confidence_sets.csv',index=False)
    pd.concat(stages).to_csv(out/'study2_MCS_elimination_stages.csv',index=False)
    (out/'study2_run_metadata.json').write_text(json.dumps({
        'confidence_set':'T_max sequential MCS, circular moving-block bootstrap',
        'multiplicity':'Holm correction within each market/evaluation block for DM pairs',
        'block':block,'replicates':replicates,'random_seed':2026,
        'HAC_lags_DM':10,'target':'next-session squared log returns',
        'caveats':['exploratory comparisons','2026 is one retrospective period',
                   'squared returns are noisy volatility proxies']},indent=2))
    return pd.DataFrame(allscores),pd.DataFrame(allmcs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--historical-forecasts');ap.add_argument('--later-forecasts')
    ap.add_argument('--sp500-prices');ap.add_argument('--tasi-prices')
    ap.add_argument('--out-dir',default='outputs/volatility');ap.add_argument('--replicates',type=int,default=4000)
    ap.add_argument('--block',type=int,default=10)
    args=ap.parse_args();out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for x in [args.historical_forecasts,args.later_forecasts]:
        if x:frames.append(pd.read_csv(x,dtype={'fold':str}))
    if args.sp500_prices or args.tasi_prices:
        if not(args.sp500_prices and args.tasi_prices):ap.error('Both market price files required for HAR')
        hars=[];coefs=[]
        for market,path in [('SP500',args.sp500_prices),('TASI',args.tasi_prices)]:
            hf,cf=har_forecast_rows(path,market);hars.append(hf);coefs.append(cf)
        har=pd.concat(hars,ignore_index=True);har.to_csv(out/'HAR_SR2_origin_forecasts.csv',index=False)
        pd.concat(coefs).to_csv(out/'HAR_SR2_coefficients.csv',index=False)
        frames.append(har)
    if not frames:ap.error('Supply per-origin forecasts or both raw market price CSVs')
    combined=pd.concat(frames,ignore_index=True)
    # If no originating QLIKE forecasts have been supplied, we output HAR
    # predictions but cannot perform meaningful pairwise inference.
    gr=combined.groupby(['market','fold'])['model'].nunique()
    if (gr>=2).any():
        combined=combined[combined.set_index(['market','fold']).index.isin(gr[gr>=2].index)]
        scores,mcs=run_inference(combined,out,args.replicates,args.block)
        print(scores.to_string(index=False));print(mcs.to_string(index=False))
    else:print('HAR forecast rows written; awaiting matched model predictions for inference.')
if __name__=='__main__':main()
