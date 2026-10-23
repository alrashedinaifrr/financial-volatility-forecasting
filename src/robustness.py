"""Out-of-sample uncertainty estimates for benchmark prediction files.
Block bootstrap preserves short-range serial dependence; not a proof of superiority.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def qlike_loss(y,p):
    p=np.maximum(np.asarray(p,dtype=float),1e-9)
    return np.log(p)+np.asarray(y,dtype=float)/p

def block_bootstrap_mean_ci(d,block=10,reps=2000,seed=42):
    d=np.asarray(d,dtype=float); n=len(d)
    if n<2*block:raise ValueError('Too few observations for requested block length')
    rng=np.random.default_rng(seed)
    means=np.empty(reps)
    for b in range(reps):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        indices=np.concatenate([(np.arange(s,s+block)%n) for s in starts])[:n]
        means[b]=d[indices].mean()
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def evaluate_file(path,block=10,reps=2000):
    df=pd.read_csv(path)
    y=df['realized_squared_return'].to_numpy()
    reference=qlike_loss(y,df['EWMA'])
    rows=[]
    for model in ['RV20','RF','HGB']:
        if model not in df:continue
        difference=qlike_loss(y,df[model])-reference
        low,high=block_bootstrap_mean_ci(difference,block,reps)
        rows.append({'model':model,'reference':'EWMA','n':len(y),'mean_qlike_difference':difference.mean(),
                     'ci95_lower':low,'ci95_upper':high,'excludes_zero':bool(low>0 or high<0)})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--outputs',default='outputs');p.add_argument('--block',type=int,default=10);p.add_argument('--reps',type=int,default=2000)
    a=p.parse_args();out=Path(a.outputs);frames=[]
    for market in ['SP500','TASI']:
        path=out/f'{market}_test_predictions.csv'
        if not path.exists():raise FileNotFoundError(path)
        res=evaluate_file(path,a.block,a.reps);res.insert(0,'market',market);frames.append(res)
    result=pd.concat(frames,ignore_index=True);result.to_csv(out/'qlike_block_bootstrap.csv',index=False)
    print(result.to_string(index=False))
if __name__=='__main__':main()
