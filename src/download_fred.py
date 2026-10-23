"""Download FRED S&P500 series when executed in an internet-enabled environment."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='data/sp500.csv')
    p.add_argument('--end',default='2026-10-22')
    a=p.parse_args()
    url='https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500'
    df=pd.read_csv(url)
    date_col='observation_date' if 'observation_date' in df.columns else 'DATE'
    df=df.rename(columns={date_col:'Date','SP500':'Close'})[['Date','Close']]
    df['Date']=pd.to_datetime(df['Date']);df['Close']=pd.to_numeric(df['Close'],errors='coerce')
    df=df.dropna();df=df[df.Date<=pd.Timestamp(a.end)]
    if len(df)<650:raise RuntimeError('Insufficient observations from FRED')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);df.to_csv(out,index=False)
    print(f'Saved {len(df)} observations to {out}; source {url}')
if __name__=='__main__':main()
