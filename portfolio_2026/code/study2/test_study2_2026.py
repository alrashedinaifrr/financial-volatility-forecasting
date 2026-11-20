from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from study2_extension import (qlike, model_confidence_set, pairwise_inference,
                              circular_block_indices, run_inference)
from run_2026_from_prices import (prepare,fit_log,estimate_multiplier,process_market,
                                  PUBLISHED_2026)


def synthetic_prices(tmp_path,name='SP500',seed=2):
    rng=np.random.default_rng(seed)
    before=pd.bdate_range(end='2026-01-22',periods=850)
    after=pd.bdate_range(start='2026-01-23',periods=52)
    dates=before.append(after)
    vol=np.clip(.01 + .003*np.sin(np.arange(len(dates))/40),.002,.05)
    r=rng.normal(0,vol)
    prices=np.exp(np.log(140)+np.cumsum(r))
    file=tmp_path/f'{name}.csv'
    pd.DataFrame({'Date':dates,'Close':prices}).to_csv(file,index=False)
    return file


def test_qlike_loss_and_labels():
    y=np.array([0,.02,.1]); h=np.array([1e-9,.04,.2])
    np.testing.assert_allclose(qlike(y,h),np.log(h)+y/h)


def test_rolling_features_are_causal(tmp_path):
    file=synthetic_prices(tmp_path)
    _,p1=prepare(file)
    raw=pd.read_csv(file);cut=600;raw.loc[cut:,'Close']*=1.15
    raw.to_csv(file,index=False)
    _,p2=prepare(file)
    keep=['abs1','abs2','abs5','rv5','rv20','rv60','neg','ewma','logrv20','har_daily','har_weekly','har_monthly']
    np.testing.assert_allclose(p1.loc[:450,keep],p2.loc[:450,keep],rtol=1e-12,atol=1e-12)


def test_2026_origin_consistency_and_qualifications(tmp_path):
    path=synthetic_prices(tmp_path)
    with pytest.raises(ValueError,match='mismatch'):
        process_market('SP500',path,False)
    rows,metadata=process_market('SP500',path,True)
    assert set(rows.model)=={'Log target','Rescaled log','Poisson','Direct mean','EWMA','HAR-SR2'}
    assert rows.groupby('date').model.nunique().eq(6).all()
    assert len(rows)==52*6
    assert rows.groupby('date').target_sq_return.nunique().eq(1).all()
    assert (rows.pred_var>0).all()
    assert not metadata['published_Qlike_replication_check_passed']
    assert metadata['fixed_training_cutoff']=='2026-01-20'
    assert metadata['multiplier']>0
    # Run paired bootstrap and sequential MCS with the HAR model included
    scores,mcs=run_inference(rows,tmp_path/'results',replicates=100,block=5)
    assert len(scores)==6 and len(mcs)==1
    assert 'HAR-SR2' in mcs.models_evaluated.iloc[0]


def test_2026_bartlett_and_mcs_synthetic():
    rng=np.random.default_rng(2026)
    n=150
    common=rng.standard_normal(n)
    losses=np.column_stack([common,common+rng.normal(0,.02,n),common+.9])
    tab=pairwise_inference(losses,['a','b','bad'],replicates=450,block=10,lag=10)
    assert len(tab)==3
    assert (tab.ci95_low<=tab.ci95_high).all()
    assert np.isfinite(tab.DM_p_two_sided).all()
    keep,stages=model_confidence_set(losses,['a','b','bad'],replicates=450,block=10)
    assert 'bad' not in keep
    assert len(stages)>=2


def test_bootstrap_keeps_common_days():
    rng=np.random.default_rng(5)
    ix=circular_block_indices(25,7,rng)
    assert ix.shape==(25,) and (ix>=0).all() and (ix<25).all()
