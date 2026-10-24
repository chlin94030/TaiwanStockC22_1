"""Causal, cache, freeze, and deterministic execution regression tests."""
from pathlib import Path
from dataclasses import replace
import copy
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from lab_config import Config
from lab_io import read_json,atomic_json,DataError,sha,Store
from lab_backtest import Case
from studio_behavior import behavior_features,benchmark_context,regime
from studio_engine import StudioCase,validate_profiles,evaluate,cached_rows
from studio_pool import adopt_dataset,lock_pool,load_pool,load_dataset,discover_prepared,report_pool_suggestions
from studio_runner import run_experiment,DEFAULT_SPEC,configure
from studio_cache import prepare_fixed_features,load_labels

HERE=Path(__file__).parent

@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    from demo_data import create
    from lab_jobs import run_pipeline
    base=tmp_path_factory.mktemp('fixedpool')
    cfg=replace(create(base/'input',stocks=44,sessions=1450),research_action='prepare_data')
    run=run_pipeline(base/'work',cfg,input_dir=base/'input')
    assert read_json(run/'state.json')['status']=='COMPLETE'
    assert not (run/'results.json').exists()
    adopt_dataset(base/'work',run)
    lock_pool(base/'work',{'5':'ADV40','20':'ADV40','60':'ADV40'})
    return base/'work',run,cfg


def fixture_case():
    n=20;d=pd.Timestamp('2023-01-03');factors=('liquidity20','high252')
    c=Case(d,np.array([f'{i:04}.TW' for i in range(n)]),np.column_stack([np.linspace(1,.05,n),np.linspace(.05,1,n)]),
        np.ones(n)*1e9,np.ones(n)*.05,np.ones(n,dtype=bool),np.ones(n)*-.03,.02,.03,0.,d+pd.Timedelta(1,'d'),d+pd.Timedelta(40,'d'),factors)
    return StudioCase(c,np.array(['A']*10+['B']*10),np.ones(n)*50,np.zeros(n),np.ones(n),'NEUTRAL_PROXY')


def cand(policy='none'):
    return validate_profiles({'schema':1,'candidates':[{'name':'test','weights':{'liquidity20':1},'policy':policy}]},['liquidity20'])[0]


def test_prepare_does_not_run_search(workspace):
    root,run,cfg=workspace
    assert read_json(run/'DATA_READY.json')['strategy_search_executed'] is False
    assert not (run/'final_freeze_60.json').exists()
    assert discover_prepared(root)[0]['run']==str(run)


def test_lock_is_not_static_tickers(workspace):
    root,run,cfg=workspace
    lock=load_pool(root)
    assert lock['members_change_monthly_using_T_only_history'] is True
    assert lock['optimality_proven'] is False
    p=pd.read_csv(run/'monthly_pools.csv.gz')
    # We keep the original historical memberships, never replace them by final-month names.
    assert lock['monthly_membership_sha']==sha(run/'monthly_pools.csv.gz')
    assert len(p.date.unique())>12


def test_lock_checksum(workspace):
    root,run,cfg=workspace;lock=load_pool(root)
    path=root/'studio'/'pool_locks'/(lock['lock_id']+'.json');raw=path.read_bytes()
    try:
        q=dict(lock);q['mapping']['60']='ADV20';atomic_json(path,q)
        with pytest.raises(DataError,match='modified'):load_pool(root)
    finally:path.write_bytes(raw)


def test_suggestions_only_training(workspace):
    root,run,cfg=workspace;path=run/'results.json'
    atomic_json(path,{'models':{'60':{'freeze':{'choice':{'pool':'ADV20'},'train_end':'2020-01-01'},
                                       'final_tail':{'best_pool':'ADV40','profit':999}}}})
    try:assert report_pool_suggestions(root)['60']['pool']=='ADV20'
    finally:path.unlink()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.1,1.1])
def test_bad_weights(value):
    with pytest.raises(DataError):validate_profiles({'schema':1,'candidates':[{'name':'x','weights':{'x':value}}]},['x'])


def test_duplicate_profiles():
    x={'name':'x','weights':{'x':1}}
    with pytest.raises(DataError,match='Duplicate'):validate_profiles({'schema':1,'candidates':[x,{**x,'name':'y'}]},['x'])


def test_unknown_factor():
    with pytest.raises(DataError):validate_profiles({'schema':1,'candidates':[{'name':'x','weights':{'unknown':1}}]},['x'])


def test_future_outcome_never_changes_selection():
    a=fixture_case();cfg=Config(end='2024-01-01');r=evaluate(a,cand(),cfg)
    a.base.net[:]=999;assert evaluate(a,cand(),cfg)['selected']==r['selected']
    a.base.net[0]=np.nan;q=evaluate(a,cand(),cfg)
    assert q['selected']==r['selected'] and np.isnan(q['net'])
    assert q['state']=='SELECTED_EXIT_UNKNOWN' and not q['decision_uses_future_outcomes']


def test_no_backfill():
    a=fixture_case();a.rsi[0]=90;r=evaluate(a,cand('discipline'),Config(end='2024-01-01'))
    assert len(r['core_selected'])==10
    assert len(r['selected'])==3 # Current sector cap; do not refill outside the locked top ten.
    assert r['cash_slots']==7
    assert all(x in r['core_selected'] for x in r['selected'])


def test_half_exposure_does_not_hide_benchmark_gap():
    a=fixture_case();a.regime='PANIC_UNCONFIRMED';cfg=Config(end='2024-01-01')
    r=evaluate(a,cand('risk_half'),cfg)
    assert r['selected_slot_weight']==.05
    assert r['excess_0050']==pytest.approx(r['net']-.02)
    assert r['exposure_matched_excess']==pytest.approx(r['net']-.01)
    assert r['filled_weight']<=.5


def test_unknown_regime_blocks_risk_policy():
    a=fixture_case();a.regime='UNKNOWN'
    assert evaluate(a,cand('risk_half'),Config(end='2024-01-01'))['state']=='INPUT_COVERAGE_BLOCKED'


def test_contrarian_requires_stabilization():
    a=fixture_case();a.recovery[:]=0
    r=evaluate(a,cand('recovery_only'),Config(end='2024-01-01'))
    assert r['selected']==[] and r['net']==0


def frame():
    rng=np.random.default_rng(14);idx=pd.bdate_range('2020-01-01',periods=400)
    c=100*np.cumprod(1+rng.normal(0,.012,len(idx)))
    return pd.DataFrame({'adj_close':c,'adj_high':c*1.01,'adj_low':c*.99,'amount':np.ones(len(idx))*1e8},index=idx)


def test_behavior_prefix_invariance():
    f=frame();a=behavior_features(f);f.iloc[300:,:]*=1.8;b=behavior_features(f)
    pd.testing.assert_frame_equal(a.iloc[:300],b.iloc[:300])


def test_context_prefix_invariance():
    f=frame();a=benchmark_context(f);f.iloc[300:,:]*=2;b=benchmark_context(f)
    pd.testing.assert_frame_equal(a.iloc[:300],b.iloc[:300])


def test_not_enough_history_is_missing():
    r=behavior_features(frame())
    assert r['recovery_ready'].iloc[:78].isna().all()
    assert r['ks_patience'].iloc[:60].isna().all()


@pytest.mark.parametrize('row,breadth,expected',[
    ({},.5,'UNKNOWN'),
    ({'market_drawdown60':-.2,'market_bias20':-.1,'market_above200':0,'market_ret5':-.05,'market_ret20':-.1,'market_vol_ratio':2},.2,'PANIC_UNCONFIRMED'),
    ({'market_drawdown60':-.15,'market_bias20':0,'market_above200':0,'market_ret5':.03,'market_ret20':-.1,'market_vol_ratio':2},.4,'RECOVERY_PROXY'),
    ({'market_drawdown60':0,'market_bias20':.15,'market_above200':1,'market_ret5':.02,'market_ret20':.2,'market_vol_ratio':1},.4,'EUPHORIA_DIVERGENCE'),
])
def test_regimes(row,breadth,expected):assert regime(row,breadth)==expected


def test_one_time_features_and_cost_cache(workspace,monkeypatch):
    root,run,cfg=workspace
    import requests
    monkeypatch.setattr(requests.sessions.Session,'request',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden')))
    f,info,hit=prepare_fixed_features(root)
    _,_,hit2=prepare_fixed_features(root);assert hit2
    labels,li=load_labels(root,f,cfg);assert li['hit']
    _,li=load_labels(root,f,replace(cfg,slippage=.002));assert not li['hit']
    _,li=load_labels(root,f,replace(cfg,slippage=.002));assert li['hit']
    assert info['http_requests']==0


def test_strategy_cache_and_integrity(tmp_path):
    cc=[fixture_case()];cfg=Config(end='2024-01-01');p=cand()
    a,h=cached_rows(tmp_path,cc,p,cfg,{'source':'fixture'},[0]);assert not h
    b,h=cached_rows(tmp_path,cc,p,cfg,{'source':'fixture'},[0]);assert h
    assert a[0]['net']==b[0]['net']
    file=next(x for x in (tmp_path/'studio'/'strategy_cache').glob('*.json') if '.sha.' not in x.name)
    file.write_text('[]')
    with pytest.raises(DataError,match='checksum'):cached_rows(tmp_path,cc,p,cfg,{'source':'fixture'},[0])


def test_fast_weight_edit_reuses_others(workspace,monkeypatch):
    root,run,cfg=workspace
    import lab_universe,requests
    monkeypatch.setattr(requests.sessions.Session,'request',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden')))
    profiles=read_json(HERE/'studio_profiles.json');spec={**DEFAULT_SPEC,'fast_bootstrap_samples':100,'final_bootstrap_samples':200}
    r1=run_experiment(root,profiles,spec)
    one=read_json(r1/'results.json');assert one['cache']['pool_searches']==0
    assert not (r1/'final_records.csv').exists()
    assert all(not v['final_tail_inspected'] for v in one['models'].values())
    changed=copy.deepcopy(profiles);changed['candidates'][0]['weights']={'liquidity20':.75,'high252':.25}
    r2=run_experiment(root,changed,spec)
    two=read_json(r2/'results.json')
    assert two['cache']['feature_hit'] and two['cache']['strategy_hits']==27
    assert two['cache']['strategy_misses']==3
    state=(r2/'state.json').read_bytes();r3=run_experiment(root,changed,spec)
    assert r2==r3 and (r3/'state.json').read_bytes()==state
    for h in two['models']:
        frozen=read_json(r2/'FROZEN_STRATEGIES.json')[h]
        assert pd.Timestamp(frozen['train_max_exit'])<pd.Timestamp(frozen['tail_start'])
    with zipfile.ZipFile(r2/'STRATEGY_RESULTS.zip') as z:assert z.testzip() is None


def test_final_release_once(workspace):
    root,run,cfg=workspace;profiles=read_json(HERE/'studio_profiles.json')
    spec={**DEFAULT_SPEC,'fast_bootstrap_samples':100,'final_bootstrap_samples':200}
    final=run_experiment(root,profiles,spec,mode='confirm')
    assert read_json(final/'state.json')['status']=='COMPLETE'
    assert (final/'final_records.csv').exists()
    changed=copy.deepcopy(profiles);changed['candidates'][0]['weights']={'liquidity20':.65,'high252':.35}
    with pytest.raises(DataError,match='already released'):run_experiment(root,changed,spec,mode='confirm')
    # Repeated development remains allowed but never releases the tail again.
    r=run_experiment(root,changed,spec,mode='fast')
    assert read_json(r/'state.json')['status']=='COMPLETE'
    assert not (r/'final_records.csv').exists()


@pytest.mark.parametrize('cost',[float('nan'),float('inf'),-.1,2.0])
def test_cost_validation(workspace,cost):
    root,_,_=workspace;_,meta=load_dataset(root)
    with pytest.raises(DataError):configure(meta,{'cost_overrides':{'slippage':cost}})


def test_snapshot_studio_allowlist():
    from lab_snapshot import allowed
    assert allowed('studio/pool_locks/'+'a'*24+'.json')
    assert allowed('studio/experiments/'+'b'*24+'/REPORT.txt')
    assert allowed('studio/confirmation_ledger.json')
    assert not allowed('studio/../../secret')
    assert not allowed('studio/datasets/'+'a'*24+'/secrets.toml')
    assert not allowed('studio/experiments/'+'b'*24+'/error.txt')


def test_snapshot_carries_fixed_lock(workspace,tmp_path):
    # Export format is intentionally for auto-mode DB only; create a source copy.
    from lab_snapshot import export_snapshot,import_snapshot
    import shutil
    root,run,cfg=workspace
    auto=tmp_path/'source';auto.mkdir();shutil.copytree(root/'studio',auto/'studio')
    shutil.copytree(run/'import_cache',auto/'cache')
    file=tmp_path/'snapshot.zip';export_snapshot(auto,file)
    target=tmp_path/'target';import_snapshot(file,target)
    assert read_json(target/'studio'/'active_pool.json')==read_json(auto/'studio'/'active_pool.json')
    assert read_json(target/'studio'/'confirmation_ledger.json')==read_json(auto/'studio'/'confirmation_ledger.json')
    assert (target/'studio'/'latest.json').exists()


def test_synthetic_not_marketed_as_performance(workspace):
    root,_,_=workspace
    p=read_json(root/'studio'/'latest.json');result=read_json(Path(p['run'])/'results.json')
    assert all(v['verdict']=='SYNTHETIC_ENGINE_TEST_NOT_PERFORMANCE' for v in result['models'].values())

class FakeUI:
    """Headless call contract, explicitly not a real Streamlit/browser test."""
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def expander(self,*a,**k):return self
    def form(self,*a,**k):return self
    def selectbox(self,label,options,index=0,format_func=None,**kw):
        if format_func:format_func(options[index])
        return options[index]
    def text_area(self,*a,value='',**k):return value
    def text_input(self,*a,value='',**k):return value
    def checkbox(self,*a,**k):return False
    def button(self,*a,**k):return False
    def form_submit_button(self,*a,**k):return False
    def fragment(self,*a,**k):return lambda f:f
    def __getattr__(self,name):return lambda *a,**k:None


def test_default_ui_no_existing_data(tmp_path):
    from studio_ui import render
    render(FakeUI(),tmp_path)


def test_default_ui_prepared_and_frozen(workspace):
    from studio_ui import render
    render(FakeUI(),workspace[0])


def test_label_cache_future_not_evaluated_in_fast(workspace,monkeypatch):
    import studio_engine
    root,run,cfg=workspace
    profiles=read_json(HERE/'studio_profiles.json')
    profiles['candidates']=profiles['candidates'][:1]
    # Every tail outcome may be stored, but fast mode must never evaluate a
    # selected strategy on that tail. Check at the evaluator boundary.
    _,meta=load_dataset(root)
    dates=pd.DatetimeIndex(sorted(pd.read_csv(run/'monthly_pools.csv.gz').date.unique()))
    boundary=dates[-cfg.holdout_months]
    old=studio_engine.evaluate
    def checked(sc,p,c):
        assert sc.base.date<boundary
        return old(sc,p,c)
    monkeypatch.setattr(studio_engine,'evaluate',checked)
    r=run_experiment(root,profiles,{**DEFAULT_SPEC,'top_k':11,'fast_bootstrap_samples':100})
    assert read_json(r/'state.json')['status']=='COMPLETE'


def test_pause_is_cooperative_and_results_resume(workspace):
    from lab_io import PauseRequested
    root,run,cfg=workspace;p=read_json(HERE/'studio_profiles.json');p['candidates']=p['candidates'][:2]
    def stop():raise PauseRequested('test pause')
    spec={**DEFAULT_SPEC,'top_k':12,'fast_bootstrap_samples':100}
    r=run_experiment(root,p,spec,check=stop)
    assert read_json(r/'state.json')['status']=='PAUSED'
    resumed=run_experiment(root,p,spec)
    assert resumed==r and read_json(resumed/'state.json')['status']=='COMPLETE'


def test_changed_date_does_not_buy_a_fresh_tail():
    from studio_runner import assert_confirmation_unused
    ledger={'old':{'experiment_id':'old','holding_windows':[['2024-01-01','2024-06-01']]}}
    with pytest.raises(DataError,match='overlapping'):assert_confirmation_unused(ledger,'new','different',[['2024-05-01','2024-08-01']])
    assert_confirmation_unused(ledger,'new','genuine_new',[['2024-07-01','2024-08-01']])
    assert_confirmation_unused(ledger,'old','old',[['2024-01-01','2024-06-01']])
