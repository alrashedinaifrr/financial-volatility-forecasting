"""2026 nine-model exploratory comparison; 72 paired tests and MCS sensitivity.

The expanded nine-model candidate set was subsequently verified with official
R MCS 0.2.1 (24/24 successful runs); archived output is in results/official_R_nine_model/.
"""
from __future__ import annotations
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from study2_extension import run_inference
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';IN=OUT/'2026_nine_model_predictions.csv'
d=pd.read_csv(IN,dtype={'fold':str,'date':str,'target_date':str})
assert d.model.nunique()==9
for block in (5,10,20):
    print('running block=',block,flush=True)
    directory=OUT/f'nine_model_inference_block_{block}'
    score,mcs=run_inference(d,directory,replicates=3999,block=block)
    assert len(score)==18
    assert len(pd.read_csv(directory/'study2_pairwise_qlike_DM_bootstrap.csv'))==72
    print('MCS',mcs.to_string(index=False),flush=True)
    if block==10:
        pw=pd.read_csv(directory/'study2_pairwise_qlike_DM_bootstrap.csv')
        pcol='DM_p_two_sided'; values=pw[pcol].to_numpy()
        order=np.argsort(values);adj=np.empty(len(values))
        adj[order]=np.maximum.accumulate((len(values)-np.arange(len(values)))*values[order]).clip(0,1)
        pw['DM_p_Holm_across_both_markets_72']=adj
        pw.to_csv(OUT/'2026_nine_model_all_72_pairwise_DM.csv',index=False)
        score.to_csv(OUT/'2026_nine_model_all_scores.csv',index=False)
    # Exclude direct-mean floor pathology to check model universe stability
    subset=d[d.model!='Direct mean']
    sm,mc=run_inference(subset,OUT/f'nine_model_without_direct_mean_block_{block}',replicates=3999,block=block)
    print('without direct mean',mc.to_string(index=False),flush=True)
print('DONE',flush=True)
