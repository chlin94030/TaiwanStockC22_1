"""V21.2 regression: reproducible parser defect + durable data-only resume.
No live source, repository, or financial performance claims are made here.
"""
from pathlib import Path
from dataclasses import replace
import json,sqlite3,time,zipfile,hashlib
import numpy as np
import pandas as pd
import pytest
from lab_config import Config
from lab_io import Store,DataError,JobLock,read_json,atomic_json,PauseRequested
from lab_sources import parse_raw
from lab_snapshot import export_snapshot,import_snapshot,validate_db,allowed
from lab_features import raw_adjusted_frame
import lab_jobs


def payload(open_=120, high=110, amount=123456.0):
    return {'status':200,'data':[{'date':'2019-01-02','stock_id':'1240','open':open_,
            'max':high,'min':90,'close':105,'Trading_Volume':1000,'Trading_money':amount}]}


def test_original_high_mismatch_reproduces_and_amount_fix_preserves_values():
    with pytest.raises(DataError,match='OHLC high mismatch'):parse_raw(payload(),'1240.TWO')
    f=parse_raw(payload(),'1240.TWO',price_policy='quarantine')
    assert f.amount.iloc[0]==123456.0 and f.volume.iloc[0]==1000
    assert f[['open','high','low','close']].isna().all().all()
    issue=f.attrs['quality_issues'][0]
    assert issue['observed']['high']==110 and issue['observed']['open']==120
    assert issue['kind']=='OHLC_RANGE_MISMATCH'


def test_quarantined_row_cannot_get_executable_price_back_from_yahoo():
    r=parse_raw(payload(),'1240.TWO',price_policy='quarantine')
    a=pd.DataFrame({'date':r.date,'open':100.,'high':110.,'low':90.,'close':105.})
    out=raw_adjusted_frame(r,a,pd.DatetimeIndex(r.date))
    assert out.adj_open.isna().all() and out.adj_close.isna().all()
    assert out.amount.iloc[0]==123456.0


@pytest.mark.parametrize('amount',[-1,float('inf'),None])
def test_bad_amount_still_fails(amount):
    with pytest.raises(DataError):parse_raw(payload(amount=amount),'1240.TWO',price_policy='quarantine')


def test_valid_price_not_modified():
    p=payload(open_=100)
    strict=parse_raw(p,'1240.TWO');safe=parse_raw(p,'1240.TWO',price_policy='quarantine')
    pd.testing.assert_frame_equal(strict,safe)
    assert safe.attrs['quality_issues']==[]


def seed(root):
    cfg=Config(start='2018-01-01',end='2026-10-20',history_provider='FINMIND').resolved()
    store=Store(root/'cache')
    f=parse_raw(payload(),'1240.TWO',price_policy='quarantine')
    store.put('raw','1240.TWO',f,cfg.start,cfg.end,'FAKE_FIXTURE')
    store.set_meta('finmind_requests',[time.time()])
    store.set_meta('finmind_retry_until',time.time()+40)
    rid=lab_jobs.identifier(cfg);run=root/'runs'/rid;run.mkdir(parents=True)
    atomic_json(root/'latest.json',{'job_id':rid,'run':str(run),'config':cfg.to_dict(),'input_dir':None})
    atomic_json(run/'state.json',{'status':'PAUSED','done':1,'total':1985,'phase':'AMOUNT'})
    atomic_json(run/'protocol.json',{'config':cfg.to_dict(),'version':__import__('lab_config').VERSION})
    (run/'error.txt').write_text('fake-secret-DO-NOT-EXPORT')
    (root/'secrets.toml').write_text('FINMIND_TOKEN="fake-secret"')
    return store,cfg,run


def test_snapshot_online_backup_and_cross_host_merge(tmp_path):
    src=tmp_path/'serverA';store,cfg,run=seed(src)
    output=tmp_path/'snapshot.zip'
    meta=export_snapshot(src,output)
    assert meta['sha256']==hashlib.sha256(output.read_bytes()).hexdigest()
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert not any('secret' in name or 'error.txt' in name for name in z.namelist())
    dest=tmp_path/'serverB';other=Store(dest/'cache')
    old=parse_raw(payload(open_=100),'1240.TWO');other.put('raw','9999.TW',old,cfg.start,cfg.end,'EXISTING')
    info=import_snapshot(output,dest)
    after=Store(dest/'cache')
    assert after.cached('raw:1240.TWO',cfg.start,cfg.end)
    assert after.cached('raw:9999.TW',cfg.start,cfg.end)
    assert after.get('raw','1240.TWO').amount.iloc[0]==123456
    assert after.get('raw','1240.TWO').open.isna().all()
    assert after.quality_summary()[0]['rows']==1
    pointer=read_json(dest/'latest.json')
    assert str(dest) in pointer['run'] and str(src) not in pointer['run']
    assert info['existing_unmatched_symbols_preserved']
    assert after.meta('finmind_retry_until')>time.time()


def test_incremental_raw_update_does_not_erase_old_quality_audit(tmp_path):
    store,cfg,run=seed(tmp_path/'a')
    f=parse_raw(payload(open_=100),'1240.TWO');f['date']=pd.to_datetime(['2019-01-03'])
    store.put('raw','1240.TWO',f,cfg.start,cfg.end,'FIXTURE')
    assert store.quality_summary()[0]['rows']==1


def test_snapshot_rejects_corruption_before_modifying_store(tmp_path):
    src=tmp_path/'src';seed(src);out=tmp_path/'ok.zip';export_snapshot(src,out)
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(out) as z,zipfile.ZipFile(bad,'w') as b:
        for item in z.infolist():
            data=z.read(item.filename)
            if item.filename=='latest.json':data+=b' '
            b.writestr(item.filename,data)
    with pytest.raises(DataError,match='checksum'):import_snapshot(bad,tmp_path/'dest')
    assert not (tmp_path/'dest'/'cache'/'market.sqlite').exists()


@pytest.mark.parametrize('name',['../escape','/tmp/escape','runs/abcd/script.py','.streamlit/secrets.toml','x\\y'])
def test_unsafe_snapshot_paths_rejected(name):assert not allowed(name)


def test_sqlite_executable_objects_rejected(tmp_path):
    store=Store(tmp_path/'cache')
    with sqlite3.connect(store.db) as c:c.execute('CREATE VIEW unexpected AS SELECT 1')
    with pytest.raises(DataError,match='unexpected'):validate_db(store.db)


def test_export_does_not_race_active_worker(tmp_path):
    root=tmp_path/'src';seed(root)
    with JobLock(root/'worker.lock'):
        with pytest.raises(DataError,match='workspace'):export_snapshot(root,tmp_path/'out.zip')


def test_deadline_saves_pause_not_error(monkeypatch,tmp_path):
    cfg=Config(start='2018-01-01',end='2026-10-20').resolved()
    def fake(store,cfg,run,progress,check,*a):
        store.put('raw','1240.TWO',parse_raw(payload(open_=100),'1240.TWO'),cfg.start,cfg.end,'FAKE')
        check();raise AssertionError('deadline should pause')
    monkeypatch.setattr(lab_jobs,'build_dataset',fake)
    run=lab_jobs.run_pipeline(tmp_path,cfg,deadline=time.monotonic()-1)
    assert read_json(run/'state.json')['status']=='PAUSED'
    assert Store(tmp_path/'cache').cached('raw:1240.TWO',cfg.start,cfg.end)


@pytest.mark.parametrize('status',['PAUSED','BLOCKED','ERROR','COMPLETE'])
def test_reconnect_does_not_restart_terminal_or_user_pause(monkeypatch,tmp_path,status):
    _,cfg,run=seed(tmp_path)
    atomic_json(run/'state.json',{'status':status})
    monkeypatch.setattr(lab_jobs,'start',lambda *a:(_ for _ in ()).throw(AssertionError('must not restart')))
    assert lab_jobs.resume_if_interrupted(tmp_path) is None


def test_reconnect_recovers_only_interrupted_active_work(monkeypatch,tmp_path):
    _,cfg,run=seed(tmp_path);atomic_json(run/'state.json',{'status':'RUNNING'})
    called=[]
    monkeypatch.setattr(lab_jobs,'start',lambda root,cfg,token,idir:called.append(cfg.end) or run)
    assert lab_jobs.resume_if_interrupted(tmp_path)==run and called==['2026-10-20']


def test_reconnect_does_not_duplicate_cross_process_job(monkeypatch,tmp_path):
    _,cfg,run=seed(tmp_path);atomic_json(run/'state.json',{'status':'RUNNING'})
    monkeypatch.setattr(lab_jobs,'start',lambda *a:(_ for _ in ()).throw(AssertionError('duplicate')))
    with JobLock(tmp_path/'worker.lock'):assert lab_jobs.resume_if_interrupted(tmp_path) is None


def test_background_driver_freezes_end_date_and_exports(monkeypatch,tmp_path):
    import run_background
    root=tmp_path/'root';_,cfg,run=seed(root)
    seen=[]
    def fake(root,cfg,**kwargs):
        seen.append(cfg.end);return run
    monkeypatch.setattr(run_background,'run_until_stop',fake)
    code=run_background.main(['--root',str(root),'--output',str(tmp_path/'out'),'--minutes','0.1'])
    assert code==0 and seen==['2026-10-20']
    assert (tmp_path/'out'/'V21_DATABASE.zip').is_file()


def test_background_already_complete_does_not_research(monkeypatch,tmp_path):
    import run_background
    root=tmp_path/'root';_,cfg,run=seed(root);atomic_json(run/'state.json',{'status':'COMPLETE'})
    monkeypatch.setattr(run_background,'run_until_stop',lambda *a,**k:(_ for _ in ()).throw(AssertionError('no re-run')))
    assert run_background.main(['--root',str(root),'--output',str(tmp_path/'out')])==0


def test_workflow_bounded_single_writer_and_no_credential_literals():
    s=(Path(__file__).parent/'.github/workflows/v21-background.yml').read_text()
    assert 'max-parallel: 1' in s and 'cancel-in-progress: false' in s
    assert '--minutes 30' in s and 'timeout-minutes: 50' in s
    assert 'retention-days: 14' in s
    assert 'schedule:' not in s and 'push:' not in s
    assert 'FINMIND_TOKEN: ${{ secrets.FINMIND_TOKEN }}' in s
    assert 'secrets.toml' not in s


def test_one_bad_amount_does_not_cancel_other_downloads(monkeypatch,tmp_path):
    import lab_data
    cfg=Config(start='2018-01-01',end='2026-10-20',history_provider='FINMIND').resolved()
    catalog=[{'ticker':'1236.TW'},{'ticker':'1240.TWO'},{'ticker':'1241.TW'}]
    called=[]
    class Fake:
        def __init__(self,*a):pass
        def catalog(self):return catalog
        def calendar(self):return pd.bdate_range('2018-01-01',periods=300).strftime('%Y-%m-%d').tolist()
        def raw(self,ticker,start=None):
            called.append(ticker)
            if ticker=='1240.TWO':raise DataError('Actual amount invalid')
            return parse_raw(payload(open_=100),'1240.TWO')
    monkeypatch.setattr(lab_data,'Client',Fake)
    run=lab_jobs.run_pipeline(tmp_path,cfg)
    assert called==['1236.TW','1240.TWO','1241.TW','0050.TW']
    assert read_json(run/'state.json')['status']=='BLOCKED'
    assert 'unresolved' in read_json(run/'state.json')['message']
    store=Store(tmp_path/'cache')
    assert store.cached('raw:1236.TW',cfg.start,cfg.end)
    assert store.cached('raw:1241.TW',cfg.start,cfg.end)
    assert not store.cached('raw:1240.TWO',cfg.start,cfg.end)
    called.clear();lab_jobs.run_pipeline(tmp_path,cfg)
    assert called==['1240.TWO']


def test_api_artifact_roundtrip_uses_inner_verified_snapshot(monkeypatch,tmp_path):
    import lab_remote
    src=tmp_path/'src';seed(src);snapshot=tmp_path/'snap.zip';export_snapshot(src,snapshot)
    artifact=tmp_path/'artifact.zip'
    with zipfile.ZipFile(artifact,'w') as z:z.write(snapshot,'V21_DATABASE.zip')
    monkeypatch.setattr(lab_remote.GitHub,'artifacts',lambda self:[{'id':123,'name':'v21-checkpoint-fixture'}])
    import shutil
    monkeypatch.setattr(lab_remote.GitHub,'archive',lambda self,suffix,dest:shutil.copy2(artifact,dest))
    result=lab_remote.restore_latest('my/repo','fake-private-token',tmp_path/'dest')
    assert result['restored'] and result['artifact_id']==123
    assert 'fake-private-token' not in json.dumps(result)
    assert Store(tmp_path/'dest'/'cache').cached('raw:1240.TWO','2018-01-01','2026-10-20')


def test_snapshot_rejects_zip_traversal_before_extract(tmp_path):
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:z.writestr('../escape.txt','malicious')
    with pytest.raises(DataError,match='Unsafe'):import_snapshot(bad,tmp_path/'dest')
    assert not (tmp_path/'escape.txt').exists()


def test_quota_defer_wait_is_persisted(monkeypatch,tmp_path):
    from lab_sources import Client
    from lab_io import SourceDeferred
    cfg=Config(start='2018-01-01',end='2026-10-20').resolved();store=Store(tmp_path/'cache')
    class Resp:
        status_code=429
        headers={'Retry-After':'600'}
        def json(self):return {'status':429,'msg':'Rate limit'}
    class Session:
        def get(self,*a,**kw):return Resp()
    c=Client(store,cfg,lambda **k:None,lambda:None);c.session=Session()
    with pytest.raises(SourceDeferred):c.json_get('https://example.com',finmind=False)
    assert store.meta('finmind_retry_until')>time.time()+590


def test_no_secret_or_executable_in_snapshot(tmp_path):
    root=tmp_path/'a';_,cfg,run=seed(root)
    (run/'malicious.py').write_text('raise Exception("no")')
    out=tmp_path/'out.zip';export_snapshot(root,out)
    with zipfile.ZipFile(out) as z:
        assert all(not x.endswith('.py') for x in z.namelist())
        assert all(b'fake-secret' not in z.read(x) for x in z.namelist() if not x.endswith('.sqlite'))
