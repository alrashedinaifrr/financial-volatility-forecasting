"""Cross-market next-session variance forecasting. No fabricated observations."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

EPS=1e-9
FEATURES=['abs1','abs2','abs5','rv5','rv20','rv60','neg','ewma','logrv20']

def prepare(path):
    df=pd.read_csv(path)
    if not {'Date','Close'}.issubset(df): raise ValueError('CSV must contain Date, Close')
    df['Date']=pd.to_datetime(df.Date,errors='raise')
    df['Close']=pd.to_numeric(df.Close,errors='raise')
    if df.Date.duplicated().any() or df.Close.isna().any() or not np.isfinite(df.Close).all() or (df.Close<=0).any(): raise ValueError('Invalid date or price')
    df=df.sort_values('Date').reset_index(drop=True)
    r=np.log(df.Close).diff(); sq=r*r
    x=pd.DataFrame({'date':df.Date,'y':sq.shift(-1)})
    for lag in [0,1,4]: x[f'abs{lag+1}']=r.abs().shift(lag)
    for w in [5,20,60]: x[f'rv{w}']=sq.rolling(w).mean()
    x['neg']=(r<0).astype(float)
    x['ewma']=sq.ewm(alpha=.06,adjust=False).mean()
    x['logrv20']=np.log(x.rv20.clip(lower=EPS))
    x=x.dropna().reset_index(drop=True)
    if len(x)<650: raise ValueError('Need >=650 complete supervised observations')
    return x

def score(y,p):
    p=np.maximum(np.asarray(p),EPS); y=np.asarray(y)
    return {'MSE':float(np.mean((y-p)**2)),'QLIKE':float(np.mean(np.log(p)+y/p))}

def models():
    return {'RF':RandomForestRegressor(n_estimators=250,max_depth=6,min_samples_leaf=12,n_jobs=-1,random_state=42),
            'HGB':HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=9,l2_regularization=1,random_state=42)}

def forecast(train,test):
    pred={'RV20':test.rv20.to_numpy(),'EWMA':test.ewma.to_numpy()}
    for name,model in models().items():
        model.fit(train[FEATURES],train.y)
        pred[name]=np.maximum(model.predict(test[FEATURES]),EPS)
    return pred

def eval_market(df,market,out):
    n=len(df); train_end=int(.6*n); val_end=int(.8*n)
    tr,val,test=df.iloc[:train_end],df.iloc[train_end:val_end],df.iloc[val_end:]
    pv=forecast(tr,val)
    scores={k:score(val.y,p) for k,p in pv.items()}
    winner=min(scores,key=lambda k:scores[k]['QLIKE'])
    pt=forecast(df.iloc[:val_end],test)
    rows=[{'market':market,'split':'validation','model':k,**score(val.y,p)} for k,p in pv.items()]
    rows +=[{'market':market,'split':'test','model':k,**score(test.y,p)} for k,p in pt.items()]
    pd.DataFrame({'market':market,'date':test.date,'realized_squared_return':test.y,**pt}).to_csv(out/f'{market}_test_predictions.csv',index=False)
    return rows,{'winner_validation':winner,'n':n,'train_end':str(tr.date.iloc[-1].date()),'validation_end':str(val.date.iloc[-1].date()),'test_start':str(test.date.iloc[0].date()),'test_end':str(test.date.iloc[-1].date())}

def transfer(src,dst,src_name,dst_name,out):
    # Strict temporal protocol: train on source data available BEFORE target test begins.
    cut=dst.date.iloc[int(.8*len(dst))]
    source_train=src[src.date<cut]
    target_test=dst.iloc[int(.8*len(dst)):]
    if len(source_train)<350: raise ValueError('Insufficient source observations before target test start')
    pred=forecast(source_train,target_test)
    rows=[{'train_market':src_name,'test_market':dst_name,'model':k,**score(target_test.y,p)} for k,p in pred.items()]
    pd.DataFrame({'date':target_test.date,'actual':target_test.y,**pred}).to_csv(out/f'transfer_{src_name}_to_{dst_name}.csv',index=False)
    return rows

def main():
    a=argparse.ArgumentParser();a.add_argument('--sp500',required=True);a.add_argument('--tasi',required=True);a.add_argument('--out',default='outputs');args=a.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data={'SP500':prepare(args.sp500),'TASI':prepare(args.tasi)}
    results=[];metadata={}
    for k,df in data.items():
        r,m=eval_market(df,k,out);results+=r;metadata[k]=m
    pd.DataFrame(results).to_csv(out/'benchmark_scores.csv',index=False)
    transfers=transfer(data['SP500'],data['TASI'],'SP500','TASI',out)+transfer(data['TASI'],data['SP500'],'TASI','SP500',out)
    pd.DataFrame(transfers).to_csv(out/'transfer_scores.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(results).to_string(index=False));print('\nCROSS MARKET\n',pd.DataFrame(transfers).to_string(index=False))
if __name__=='__main__': main()
