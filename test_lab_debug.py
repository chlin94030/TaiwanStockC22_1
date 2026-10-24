"""V21.1 visible startup / safety regression tests. No live API calls."""
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from threading import Lock
import time
import pytest
from lab_config import load_config
from lab_runtime import estimated_hours, hours_message, seconds_since, progress_caption
from lab_io import JobLock,read_json,SourceBlocked
from lab_jobs import start,run_pipeline
from lab_sources import Client


def test_first_run_must_disclose_hours():
    assert estimated_hours(2001) > 7.5
    assert estimated_hours(2001,True) > 3.5
    assert '小時' in hours_message(2001)
    assert 'Token' not in hours_message(2001)


def test_cache_progress_not_counted_before_done():
    assert '100' in progress_caption({'phase':'AMOUNT','done':1,'total':101})
    assert progress_caption({'phase':'POOL','done':1,'total':101})==''
    assert seconds_since('not-a-date') is None
    assert seconds_since(datetime.now(timezone.utc).isoformat()) < 10


def test_queued_pointer_committed_before_background_finish(tmp_path):
    from demo_data import create
    import lab_jobs
    cfg=create(tmp_path/'input',stocks=44,sessions=1200)
    root=tmp_path/'work'
    run=start(root,cfg,input_dir=tmp_path/'input')
    item=read_json(root/'latest.json')
    assert item['run']==str(run)
    assert read_json(run/'state.json')['status'] in ('QUEUED','RUNNING','COMPLETE')
    worker=lab_jobs._THREADS.get(str(root.resolve()))
    assert worker
    worker.join(timeout=40)
    assert not worker.is_alive(), 'Synthetic integration should finish promptly'
    assert read_json(run/'state.json')['status']=='COMPLETE'
    original=read_json(run/'results.json')
    again=start(root,cfg,input_dir=tmp_path/'input')
    assert again==run
    assert read_json(run/'results.json')==original


def test_workspace_lock_conflict_is_visible(tmp_path):
    from demo_data import create
    cfg=create(tmp_path/'dataset',stocks=44,sessions=1200)
    root=tmp_path/'job'
    with JobLock(root/'worker.lock'):
        run=run_pipeline(root,cfg,input_dir=tmp_path/'dataset')
    state=read_json(run/'state.json')
    assert state['status']=='BLOCKED'
    assert state['cached_work_retained'] is True
    assert 'workspace' in state['message']


def test_429_long_quota_does_not_silently_sleep(monkeypatch,tmp_path):
    class FakeStore:
        def __init__(self):self.params={}
        def meta(self,key,default=None):return self.params.get(key,default)
        def set_meta(self,key,value):self.params[key]=value
    class Resp:
        status_code=429
        headers={'Retry-After':'3600'}
        def json(self):return {'status':429,'msg':'Request limit exceeded'}
    class FakeSession:
        def get(self,*args,**kwargs):return Resp()
    cfg=load_config(Path(__file__).parent/'config.json')
    x=Client(FakeStore(),cfg,lambda **kw:None,lambda:None)
    x.session=FakeSession()
    with pytest.raises(SourceBlocked,match='配額'):
        x.json_get('https://api.finmindtrade.com/api/v4/data',
                   params={'dataset':'TaiwanStockTradingDate'},finmind=False)


def test_mocked_whole_market_auto_download_resumes_exact_cache(monkeypatch,tmp_path):
    """Actual auto branch with simulated provider, not investment performance."""
    import pandas as pd
    import lab_data
    from demo_data import create
    from lab_config import Config
    from lab_io import SourceBlocked,Store
    source=tmp_path/'mock_source'
    _=create(source,stocks=44,sessions=1200)
    prices=pd.read_csv(source/'prices.csv.gz',parse_dates=['date'])
    calendar=pd.read_csv(source/'calendar.csv')['date'].tolist()
    catalog=read_json(source/'catalog.json')
    indexed={t:f for t,f in prices.groupby('ticker')}
    days=(pd.Timestamp(calendar[0]).date().isoformat(),pd.Timestamp(calendar[-1]).date().isoformat())
    cfg=Config(start=days[0],end=days[1],pools=(10,20),source_mode='auto_current_roster',history_provider='FINMIND',
               min_current_roster=40,min_train_months=12,outer_test_months=3,
               min_outer_folds=2,holdout_months=6,candidate_budget=20,
               block_bootstrap_samples=100,capital=100_000)
    cfg.validate()

    class MockClient:
        attempted=[]
        fail_symbol=None
        fail=True
        def __init__(self,store,cfg,progress,check,token=None):
            self.store=store
        def catalog(self):return catalog
        def calendar(self):return calendar
        def raw(self,ticker,start=None):
            self.attempted.append(ticker)
            if self.fail and ticker==self.fail_symbol:
                raise SourceBlocked('Simulated provider outage')
            f=indexed[ticker]
            f=f[f.date>=pd.Timestamp(start or days[0])]
            return f[['date','raw_open','raw_high','raw_low','raw_close','volume','amount']].rename(columns={
                    'raw_open':'open','raw_high':'high','raw_low':'low','raw_close':'close'})
        def adjusted(self,ticker):
            return indexed[ticker][['date','adj_open','adj_high','adj_low','adj_close']].rename(columns={
                'adj_open':'open','adj_high':'high','adj_low':'low','adj_close':'close'})
        def wait(self,*args,**kwargs):pass

    MockClient.fail_symbol=catalog[5]['ticker']
    monkeypatch.setattr(lab_data,'Client',MockClient)
    root=tmp_path/'job'
    run=run_pipeline(root,cfg)
    state=read_json(run/'state.json')
    assert state['status']=='BLOCKED'
    store=Store(root/'cache')
    assert store.cached('raw:'+catalog[0]['ticker'],*days)
    assert not store.cached('raw:'+catalog[5]['ticker'],*days)
    attempted_first=len(MockClient.attempted)
    MockClient.fail=False
    run=run_pipeline(root,cfg)
    state=read_json(run/'state.json')
    assert state['status']=='COMPLETE',state
    assert catalog[0]['ticker'] not in MockClient.attempted[attempted_first:]
    assert read_json(run/'results.json')['models']


def test_streamlit_module_renders_without_blocking_data_fetch(monkeypatch, tmp_path):
    """UI-contract smoke test; it is NOT a real Streamlit Cloud browser test."""
    import sys
    import types
    import runpy
    import lab_jobs

    calls=[]
    class FakeST(types.ModuleType):
        def __init__(self):
            super().__init__('streamlit')
            self.secrets={}
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def __getattr__(self,name):
            if name=='fragment':return lambda **kw:lambda f:f
            if name=='columns':return lambda n:[self]*n
            if name=='expander':return lambda *a,**kw:self
            if name=='radio':return lambda *a,**kw:a[1][0]
            if name=='button':return lambda *a,**kw:False
            if name=='file_uploader':return lambda *a,**kw:None
            if name=='checkbox':return lambda *a,**kw:False
            def accept(*args,**kwargs):
                calls.append(name)
                return None
            return accept
    monkeypatch.setitem(sys.modules,'streamlit',FakeST())
    monkeypatch.setattr(lab_jobs,'start',lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('UI should not start downloads')))
    runpy.run_path(str(Path(__file__).parent/'app.py'),run_name='__main__')
    assert 'title' in calls and 'warning' in calls
