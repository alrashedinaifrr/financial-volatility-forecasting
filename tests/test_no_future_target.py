import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark import prepare

def test_target_is_next_return_and_features_ignore_future(tmp_path):
    dates=pd.bdate_range('2020-01-01',periods=800)
    prices=100*np.exp(np.cumsum(np.sin(np.arange(800)/17)*.003+.0001))
    f=tmp_path/'prices.csv'; pd.DataFrame({'Date':dates,'Close':prices}).to_csv(f,index=False)
    a=prepare(f)
    changed=prices.copy(); changed[-1]*=1.2
    pd.DataFrame({'Date':dates,'Close':changed}).to_csv(f,index=False)
    b=prepare(f)
    common=a.iloc[:-1].merge(b.iloc[:-1],on='date',suffixes=('_a','_b'))
    for name in ['abs1','abs2','abs5','rv5','rv20','rv60','neg','ewma','logrv20']:
        np.testing.assert_allclose(common[name+'_a'],common[name+'_b'])
    assert a.y.iloc[-1]!=b.y.iloc[-1]
