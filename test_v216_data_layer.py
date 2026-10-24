import types
from pathlib import Path
import numpy as np
import pandas as pd

from lab_config import Config
from lab_io import Store, read_json
from lab_sources import Client
from lab_bulk import vectorized_turnover, finlab_credentials_present
from lab_universe import turnover_snapshots, signal_calendar
from demo_data import create


def test_finlab_credentials_require_all_three():
    assert not finlab_credentials_present({})
    assert not finlab_credentials_present({'FINLAB_REFRESH_TOKEN':'a','FINLAB_SESSION_ID':'b'})
    assert finlab_credentials_present({'FINLAB_REFRESH_TOKEN':'a','FINLAB_SESSION_ID':'b','FINLAB_API_KEY':'c'})


def test_dynamic_finmind_quota_uses_reported_plan(tmp_path):
    cfg=Config(start='2020-01-01',end='2026-01-01',finmind_quota_utilization=.9)
    store=Store(tmp_path/'cache')
    c=Client(store,cfg,lambda **kw:None,lambda:None,token='secret')
    class Resp:
        ok=True
        def json(self):return {'api_request_limit':6000,'user_count':123}
    c.session=types.SimpleNamespace(get=lambda *a,**k:Resp())
    assert c.finmind_limit(force=True)==5400
    info=store.meta('finmind_quota_info')
    assert info['reported_limit']==6000 and info['safe_limit']==5400


def test_vectorized_turnover_matches_per_stock_math():
    cal=pd.bdate_range('2020-01-01',periods=500)
    cfg=Config(start=str(cal[0].date()),end=str(cal[-1].date()),pools=(2,),top_k=1,
               min_train_months=12,outer_test_months=3,min_outer_folds=1,holdout_months=6,
               candidate_budget=20,recent_amount_floor=1,active_ratio60=.5)
    signals=signal_calendar(cal,cfg)
    amount=pd.DataFrame({'1000':np.arange(len(cal))+1_000_000.,'1001':np.arange(len(cal))+2_000_000.},index=cal)
    mapping={'1000':'1000.TW','1001':'1001.TWO'}
    got=vectorized_turnover(amount,signals,cfg,mapping).set_index(['date','ticker']).sort_index()
    rows=[]
    for code,ticker in mapping.items():
        raw=pd.DataFrame({'date':cal,'amount':amount[code].to_numpy(),'volume':np.ones(len(cal))*1000})
        x=turnover_snapshots(raw,cal,signals,cfg);x['ticker']=ticker;rows.append(x)
    expected=pd.concat(rows).set_index(['date','ticker']).sort_index()
    for col in ('amount252','amount20','amount60','active60'):
        assert np.allclose(got[col].to_numpy(),expected[col].to_numpy(),equal_nan=True)
    assert got.eligible.tolist()==expected.eligible.tolist()


def test_bulk_path_skips_per_symbol_provider_requests(monkeypatch,tmp_path):
    import lab_data, lab_jobs
    source=tmp_path/'source';base=create(source,stocks=44,sessions=1200)
    prices=pd.read_csv(source/'prices.csv.gz',parse_dates=['date'])
    catalog=read_json(source/'catalog.json');calendar=pd.read_csv(source/'calendar.csv')['date'].tolist()
    indexed={t:f for t,f in prices.groupby('ticker')}
    start,end=str(prices.date.min().date()),str(prices.date.max().date())
    cfg=Config(start=start,end=end,pools=(10,20),top_k=5,source_mode='auto_current_roster',
               history_provider='AUTO',min_current_roster=40,min_train_months=12,outer_test_months=3,
               min_outer_folds=2,holdout_months=6,candidate_budget=20,block_bootstrap_samples=100,capital=100_000)

    class FakeClient:
        raw_calls=0;adj_calls=0
        def __init__(self,*a,**k):self.token=''
        def catalog(self):return catalog
        def raw(self,*a,**k):self.raw_calls+=1;raise AssertionError('per-symbol raw should not be used')
        def adjusted(self,*a,**k):self.adj_calls+=1;raise AssertionError('Yahoo adjusted should not be used')

    class FakeBulk:
        populated=[]
        def __init__(self,cfg,progress,check):self.cfg=cfg;self.progress=progress;self.check=check
        def calendar(self):return calendar
        def turnover(self,signals,cat,cache_dir):
            wide=prices[prices.ticker!='0050.TW'].pivot(index='date',columns='ticker',values='amount')
            wide.columns=[c.split('.')[0] for c in wide.columns]
            mapping={r['ticker'].split('.')[0]:r['ticker'] for r in cat}
            return vectorized_turnover(wide,signals,self.cfg,mapping),{'provider':'FAKE_BULK'}
        def populate_selected_prices(self,store,tickers):
            self.populated.extend(tickers)
            for t in tickers:
                f=indexed[t]
                raw=f[['date','raw_open','raw_high','raw_low','raw_close','volume','amount']].rename(columns={'raw_open':'open','raw_high':'high','raw_low':'low','raw_close':'close'})
                adj=f[['date','adj_open','adj_high','adj_low','adj_close']].rename(columns={'adj_open':'open','adj_high':'high','adj_low':'low','adj_close':'close'})
                store.put('raw',t,raw,start,end,'FAKE_BULK')
                store.put('adj',t,adj,start,end,'FAKE_BULK_ADJ')
            return {'provider':'FAKE_BULK','symbols':len(tickers)}

    monkeypatch.setattr(lab_data,'Client',FakeClient)
    monkeypatch.setattr(lab_data,'FinLabBulk',FakeBulk)
    monkeypatch.setattr(lab_data,'finlab_ready',lambda:True)
    run=lab_jobs.run_pipeline(tmp_path/'job',cfg)
    state=read_json(run/'state.json')
    assert state['status']=='COMPLETE'
    assert FakeClient.raw_calls==0 and FakeClient.adj_calls==0
    scope=read_json(run/'source_scope.json')
    assert scope['history_provider']=='FINLAB_BULK'
    assert scope['monthly_union_count']<=len(catalog)
