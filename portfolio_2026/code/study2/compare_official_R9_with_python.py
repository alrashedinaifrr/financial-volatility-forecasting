#!/usr/bin/env python3
"""Compare actual official R MCS 9-model results against archived Python sensitivity.
Requires completed R-generated ZIP. Does not simulate or impute missing official R output.
"""
import argparse, zipfile, io, pathlib, csv, json, sys
import pandas as pd

MODEL_NAMES={'Direct mean','EWMA','GARCH','HAR-SR2','Log target','Poisson','RF','RV20','Rescaled log'}

def read_r_zip(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, 'Corrupt ZIP'
        opts=[n for n in z.namelist() if n.endswith('/run_status.csv') or n=='run_status.csv']
        if len(opts)!=1:raise ValueError('Expected exactly one official-R run_status.csv; found '+str(opts))
        status=pd.read_csv(io.BytesIO(z.read(opts[0])))
        if len(status)!=24 or status['status'].ne('OK').any():
            raise ValueError('Official R package run incomplete: expected 24 OK cases')
        versions=[n for n in z.namelist() if n.endswith('/sessionInfo.txt') or n=='sessionInfo.txt']
        if len(versions)!=1:raise ValueError('Missing official R sessionInfo.txt')
        info=z.read(versions[0]).decode('utf-8')
        if 'MCS version:' not in info:raise ValueError('Missing official MCS package version')
    return status,info

def parse_set(value):
    if pd.isna(value) or not str(value).strip():return frozenset()
    return frozenset(t.strip() for t in str(value).split(';') if t.strip())

def compare(status,base):
    actual=[]
    for _,r in status.iterrows():
        incl=parse_set(r['included']);excl=parse_set(r['excluded'])
        expected=MODEL_NAMES if r['candidate_set']=='all_nine' else MODEL_NAMES-{'Direct mean'}
        if incl & excl or incl | excl != expected:raise ValueError('Invalid R membership for '+r['id'])
        out={'market':r['market'],'candidate_set':r['candidate_set'],'statistic':r['statistic'],
             'k':int(r['k']),'R_included':'; '.join(sorted(incl)),'R_excluded':'; '.join(sorted(excl)),
             'R_B':int(r['B']), 'R_status':r['status']}
        if r['statistic']=='Tmax':
            folder=('nine_model_inference_block_' if r['candidate_set']=='all_nine' else
                    'nine_model_without_direct_mean_block_')+str(r['k'])
            source=base/'results'/folder/'study2_model_confidence_sets.csv'
            py=pd.read_csv(source)
            rr=py[py.market==r['market']]
            if len(rr)!=1:raise ValueError('Missing custom Python sensitivity for '+r['id'])
            retained=frozenset(t.strip() for t in rr.iloc[0].mcs_retained.split(',') if t.strip())
            out.update(Python_included='; '.join(sorted(retained)), agreement=incl==retained,
                       Python_B=int(rr.iloc[0].replicates))
        else:
            out.update(Python_included='',agreement=None,Python_B=None)
        actual.append(out)
    df=pd.DataFrame(actual)
    if len(df)!=24:raise ValueError('Unexpected number of cases')
    return df

def main():
    p=argparse.ArgumentParser();p.add_argument('--r-zip',type=pathlib.Path,required=True)
    p.add_argument('--project',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    st,session=read_r_zip(a.r_zip)
    result=compare(st,a.project)
    a.output.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output/'R9_official_vs_custom_Python.csv',index=False)
    (a.output/'R_sessionInfo.txt').write_text(session)
    summary={'official_R_cases_successful':len(st), 'Tmax_comparable_cases':int(result.agreement.notna().sum()),
             'Tmax_membership_agreements':int(result.agreement.fillna(False).sum()),
             'Tmax_membership_disagreements':int((result.agreement==False).sum()),
             'TR_official_only_cases':int((result.statistic=='TR').sum()),
             'warning':'Difference in membership is not automatically a bug; bootstrap methods and seeds differ.'}
    (a.output/'R9_comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
