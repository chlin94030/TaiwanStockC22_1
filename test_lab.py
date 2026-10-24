from dataclasses import replace
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from lab_config import Config,FACTORS
from lab_io import DataError,Store,atomic_json,read_json,sha
from lab_sources import parse_raw,parse_roster,parse_yahoo_history
from lab_universe import turnover_snapshots,make_pools,validate_membership
from lab_features import features,percentile_reference,return_record,attach_evidence
from lab_candidates import candidate_registry
from lab_backtest import Case,evaluate
from lab_search import matured_training_indices,choose_inner,verdict
from lab_stats import disjoint_indices,uncertainty
from lab_data import extract_dataset

@pytest.fixture
def cfg():return Config(end='2026-10-20',pools=(10,20),top_k=5,block_bootstrap_samples=100)

@pytest.fixture
def f():
    dates=pd.bdate_range('2020-01-01',periods=400)
    a=np.arange(400)*.05+100
    return pd.DataFrame({'raw_open':a,'raw_high':a+1,'raw_low':a-1,'raw_close':a,
        'adj_open':a,'adj_high':a+1,'adj_low':a-1,'adj_close':a,'amount':1e8,'volume':1e6},index=dates)


def test_config_defaults_do_not_imply_optimum(cfg):
    cfg.validate();assert cfg.holdout_months==24;assert cfg.pools==(10,20)

def test_historical_scope_never_uses_current_auto(cfg):
    with pytest.raises(ValueError):replace(cfg,requested_scope='HISTORICAL_MEMBERSHIP').validate()

def test_bad_config_rejected(cfg):
    with pytest.raises(ValueError):replace(cfg,lookback=180).validate()
    with pytest.raises(ValueError):replace(cfg,fee_sell=-.1).validate()

def test_registry_finite_deterministic(cfg):
    a=candidate_registry(cfg);b=candidate_registry(cfg)
    assert a==b and len(a)==64 and len({c['id'] for c in a})==64
    assert all(abs(sum(c['weights'].values())-1)<1e-9 for c in a)
    assert all(set(c['weights']).issubset(FACTORS) for c in a)

def test_registry_no_outcomes_input(cfg):
    assert candidate_registry(cfg)[0]['weights']=={'high252':.2,'liquidity20':.8}

def test_actual_amount_not_price_volume():
    p={'status':200,'data':[{'date':'2024-01-02','stock_id':'2330','open':100,'max':110,'min':90,'close':105,'Trading_Volume':1000,'Trading_money':102345}]}
    q=parse_raw(p,'2330.TW');assert q.amount.iloc[0]==102345 and q.amount.iloc[0]!=q.close.iloc[0]*q.volume.iloc[0]
    del p['data'][0]['Trading_money']
    with pytest.raises(DataError):parse_raw(p,'2330.TW')

def test_symbol_mismatch():
    p={'status':200,'data':[{'date':'2024-01-02','stock_id':'2317','open':1,'max':2,'min':1,'close':1,'Trading_Volume':10,'Trading_money':12}]}
    with pytest.raises(DataError):parse_raw(p,'2330.TW')

def test_adjusted_data_never_raw_fallback():
    f=pd.DataFrame({'Open':[100.],'High':[110.],'Low':[90.],'Close':[100.]},index=pd.to_datetime(['2024-01-02']))
    with pytest.raises(DataError):parse_yahoo_history(f)
    f['Adj Close']=50.;out=parse_yahoo_history(f);assert out.open.iloc[0]==50

def test_factor_future_mutation(f):
    a=features(f);g=f.copy();g.loc[g.index[330]:,['adj_close','adj_open','adj_high','adj_low']]*=10
    b=features(g)
    pd.testing.assert_frame_equal(a.iloc[:330],b.iloc[:330])

def test_amount_year_not_180(f,cfg):
    r=f.rename(columns=lambda c:c.removeprefix('raw_')).loc[:,['open','high','low','close','amount','volume']].reset_index(names='date')
    s=turnover_snapshots(r,f.index,[f.index[180],f.index[251]],cfg)
    assert np.isnan(s.amount252.iloc[0]);assert s.amount252.iloc[1]==1e8

def test_real_zero_vs_unknown_amount(f,cfg):
    r=pd.DataFrame({'date':f.index,'open':100,'high':101,'low':99,'close':100,'amount':1e8,'volume':1e6})
    r.loc[200,'amount']=0
    a=turnover_snapshots(r,f.index,[f.index[251]],cfg)
    assert a.amount252.iloc[0]<1e8
    r.loc[200,'amount']=np.nan
    assert np.isnan(turnover_snapshots(r,f.index,[f.index[251]],cfg).amount252.iloc[0])

def turnover_frame():
    return pd.DataFrame([{'date':pd.Timestamp(d),'ticker':str(1000+i)+'.TW','amount252':1000-i,
                         'amount20':1e8,'eligible':True} for d in ('2024-01-31','2024-02-29') for i in range(25)])

def test_universe_strict_size(cfg):
    with pytest.raises(DataError):make_pools(turnover_frame(),pd.to_datetime(['2024-01-31']),replace(cfg,pools=(100,500)))

def test_monthly_topn_nested(cfg):
    p,_=make_pools(turnover_frame(),pd.to_datetime(['2024-01-31','2024-02-29']),cfg)
    for d,f in p.groupby('date'):assert set(f[f.pool=='ADV10'].ticker)<=set(f[f.pool=='ADV20'].ticker)

def test_future_turnover_does_not_change_past_pool(cfg):
    t=turnover_frame();a,_=make_pools(t,pd.to_datetime(['2024-01-31']),cfg)
    t.loc[t.date==pd.Timestamp('2024-02-29'),'amount252']*=-1
    b,_=make_pools(t,pd.to_datetime(['2024-01-31']),cfg);pd.testing.assert_frame_equal(a,b)

def test_capital_not_substitute_for_market_cap(cfg):
    with pytest.raises(DataError):make_pools(turnover_frame(),pd.to_datetime(['2024-01-31']),replace(cfg,include_market_cap_pools=True))

def test_future_membership_blocked():
    m=pd.DataFrame({'date':['2024-01-31'],'ticker':['1000.TW'],'available_at':['2024-02-02'],'source':['test']})
    with pytest.raises(DataError):validate_membership(m)

def test_ranking_reference_not_pool_specific(cfg):
    f=pd.DataFrame({'date':[pd.Timestamp('2024-01-31')]*30,'ticker':[str(i) for i in range(30)]})
    for col in FACTORS:f[col]=np.arange(30)
    out=percentile_reference(f,cfg)
    assert out.rank_high252.iloc[9]==10/30

def test_future_evidence_unavailable(cfg):
    f=pd.DataFrame({'date':pd.to_datetime(['2024-01-31','2024-02-29']),'ticker':['1000.TW']*2})
    e=pd.DataFrame({'ticker':['1000.TW'],'available_at':['2024-02-05T00:00:00Z'],'factor':['revenue_yoy'],'value':[20.],'source':['published']})
    a=attach_evidence(f,e);assert np.isnan(a.revenue_yoy.iloc[0]);assert a.revenue_yoy.iloc[1]==20

def test_trade_cost_arithmetic(f,cfg):
    q=return_record(f,f.index[300],5,cfg)
    expected=(f.adj_close.iloc[305]/f.adj_open.iloc[301])*(1-cfg.fee_sell-cfg.stock_tax-cfg.slippage)/(1+cfg.fee_buy+cfg.slippage)-1
    assert q['net']==pytest.approx(expected)

def test_missing_exit_does_not_become_benchmark(f,cfg):
    f.loc[f.index[305],'adj_close']=np.nan
    q=return_record(f,f.index[300],5,cfg);assert np.isnan(q['net']) and q['state']=='EXIT_PRICE_UNKNOWN'

def test_suspended_benchmark_not_zero(f,cfg):
    f.loc[f.index[301],'volume']=0
    assert np.isnan(return_record(f,f.index[300],5,cfg,True)['net'])
    assert return_record(f,f.index[300],5,cfg,False)['net']==0

def test_gap_cap_cash(f,cfg):
    f.loc[f.index[301],'adj_open']=f.adj_close.iloc[300]*1.09
    q=return_record(f,f.index[300],5,cfg);assert q['net']==0 and not q['filled']

def case():
    n=20;ranks=np.column_stack([np.arange(1,n+1)/n]*len(FACTORS))
    return Case(pd.Timestamp('2024-01-31'),np.array([str(1000+i)+'.TW' for i in range(n)]),
        ranks,np.repeat(1e9,n),np.arange(n)/100,np.ones(n,bool),np.repeat(-.1,n),.02,.05,0,
        pd.Timestamp('2024-02-01'),pd.Timestamp('2024-02-08'),FACTORS)

def test_rank_never_depends_on_future_return(cfg):
    c=case();a=evaluate(c,{'high252':1},cfg);c.net[:]=np.nan
    b=evaluate(c,{'high252':1},cfg);assert a['selected']==b['selected'];assert np.isnan(b['net'])

def test_unknown_loss_floor_separate_from_return(cfg):
    c=case();c.net[19]=np.nan;q=evaluate(c,{'high252':1},cfg)
    assert np.isnan(q['net']);assert q['missing_slots']==1;assert np.isfinite(q['loss_floor'])

def test_capacity_cash_slots_no_backfill_after_outcome(cfg):
    c=case();c.amount20[:]=1;c.amount20[-2:]=1e9
    q=evaluate(c,{'high252':1},cfg);assert q['cash_slots']==3;assert q['net']==pytest.approx((.18+.19)/5)

def test_feature_missing_blocks_not_cash(cfg):
    c=case();c.ranks[:,0]=np.nan
    q=evaluate(c,{'liquidity20':1},cfg);assert q['state']=='FEATURE_COVERAGE_BLOCKED';assert np.isnan(q['net'])

def test_purge_uses_label_exit(cfg):
    cal=pd.bdate_range('2020-01-01',periods=1000)
    dates=cal[300:800:20];exits=pd.DatetimeIndex([cal[cal.get_loc(d)+60] for d in dates])
    boundary=dates[15];idx=matured_training_indices(dates,exits,boundary,cal,5)
    assert all(exits[idx]<cal[cal.get_loc(boundary)-5]);assert len(idx)<15

def test_holdout_does_not_choose_winner(cfg):
    cfg=replace(cfg,min_train_months=12)
    cs=candidate_registry(cfg);N=36;C=len(cs);rng=np.random.default_rng(5)
    net=rng.normal(.01,.02,(N,C));g={'net':net.copy(),'floor':net.copy(),'fills':np.ones((N,C)),'benchmark':np.zeros(N)}
    best,_=choose_inner({('ADV10',20):g},cs,20,np.arange(24),cfg)
    g['net'][24:]=1000;g['floor'][24:]=-1000
    other,_=choose_inner({('ADV10',20):g},cs,20,np.arange(24),cfg)
    assert best==other

def test_disjoint_global_selection():
    d=pd.to_datetime(['2024-01-01','2024-01-15','2024-02-01','2024-03-01'])
    e=pd.to_datetime(['2024-02-15','2024-02-29','2024-03-15','2024-04-15'])
    assert disjoint_indices(d,e)==[0,3]

def test_missing_dates_disclosed(cfg):
    d=pd.date_range('2020-01-01',periods=12,freq='MS');a=np.ones(12)*.01;a[3]=np.nan
    q=uncertainty(a,d,d+pd.Timedelta(days=5),cfg)
    assert q['complete']==11 and len(q['missing_dates'])==1

def test_safe_zip_rejects_path_traversal(tmp_path):
    p=tmp_path/'evil.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('../app.py','bad')
    with pytest.raises(DataError):extract_dataset(p,tmp_path/'out')

def test_store_null_and_cache(tmp_path):
    s=Store(tmp_path/'db');f=pd.DataFrame({'date':pd.to_datetime(['2024-01-01']),'open':[1.],'high':[2.],'low':[1.],'close':[2.],'volume':[5.],'amount':[np.nan]})
    s.put('raw','1000.TW',f,'2024-01-01','2024-01-05','TEST')
    assert s.cached('raw:1000.TW','2024-01-02','2024-01-03')
    assert not s.cached('raw:1000.TW','2023-01-01','2024-01-03')
    assert pd.isna(s.get('raw','1000.TW').amount.iloc[0])

def test_json_nan_is_null(tmp_path):
    p=tmp_path/'f.json';atomic_json(p,{'x':np.nan,'y':np.int64(3)})
    assert read_json(p)=={'x':None,'y':3}


def test_pipeline_and_completed_resume(tmp_path):
    from demo_data import create
    from lab_jobs import run_pipeline
    cfg=create(tmp_path/'input',stocks=44,sessions=1200)
    run=run_pipeline(tmp_path/'job',cfg,input_dir=tmp_path/'input')
    state=read_json(run/'state.json')
    assert state['status']=='COMPLETE',state
    results=read_json(run/'results.json')
    assert len(results['models'])==3
    assert all(v['verdict']=='SYNTHETIC_ENGINE_TEST_NOT_PERFORMANCE' for v in results['models'].values())
    before=sha(run/'results.json')
    again=run_pipeline(tmp_path/'job',cfg,input_dir=tmp_path/'input')
    assert again==run and sha(run/'results.json')==before
    with zipfile.ZipFile(run/'LAB_RESULTS.zip') as z:assert z.testzip() is None
    for v in results['models'].values():
        for fold in v['outer_folds']:
            if 'max_train_label_exit' in fold:assert fold['max_train_label_exit']<fold['test_start']

def test_unknown_entry_volume_not_filled(f,cfg):
    f.loc[f.index[301],'volume']=np.nan
    q=return_record(f,f.index[300],5,cfg)
    assert np.isnan(q['net']) and not q['filled'] and q['state']=='ENTRY_VOLUME_UNKNOWN'

def test_bad_weights_rejected(cfg):
    with pytest.raises(DataError):evaluate(case(),{'invented':1},cfg)
    with pytest.raises(DataError):evaluate(case(),{'high252':.5},cfg)

def test_optional_registry_obeys_budget(cfg):
    c=replace(cfg,source_mode='import',enable_optional_factors=True)
    c.validate();assert len(candidate_registry(c))==c.candidate_budget
    with pytest.raises(ValueError):replace(c,candidate_budget=20).validate()

def test_pit_cap_no_future_lookup(cfg):
    t=turnover_frame();caps=t[['date','ticker']].copy();caps['market_cap']=np.arange(len(caps))+100
    caps['available_at']='2025-01-01';caps['source']='TEST'
    with pytest.raises(DataError):make_pools(t,pd.to_datetime(['2024-01-31']),replace(cfg,include_market_cap_pools=True),caps=caps)

def test_raw_incremental_does_not_delete_old_rows(tmp_path):
    s=Store(tmp_path/'db')
    f=pd.DataFrame({'date':['2024-01-01'],'open':[1.],'high':[2.],'low':[1.],'close':[2.],'volume':[5.],'amount':[9.]})
    s.put('raw','1000.TW',f,'2024-01-01','2024-01-01','TEST')
    f.date='2024-01-02';s.put('raw','1000.TW',f,'2024-01-01','2024-01-02','TEST')
    assert len(s.get('raw','1000.TW'))==2;assert s.covered_range('raw:1000.TW')==('2024-01-01','2024-01-02')

@pytest.mark.parametrize('h',[5,20,60])
@pytest.mark.parametrize('benchmark',[False,True])
def test_vectorized_labels_match_scalar(f,cfg,h,benchmark):
    from lab_features import label_records
    f.loc[f.index[301],'volume']=0
    f.loc[f.index[321],'adj_open']=f.adj_close.iloc[320]*1.1
    f.loc[f.index[340],'raw_close']=np.nan
    dates=f.index[290:341:5]
    fast=label_records(f,dates,h,cfg,benchmark)
    for r in fast:
        q=return_record(f,r['date'],h,cfg,benchmark)
        assert r['state']==q['state']
        assert r['filled']==q['filled']
        assert (np.isnan(r['net']) and np.isnan(q['net'])) or r['net']==pytest.approx(q['net'])
        assert (np.isnan(r['path_mdd']) and np.isnan(q['path_mdd'])) or r['path_mdd']==pytest.approx(q['path_mdd'])

def test_batched_search_matches_scalar(cfg):
    from lab_backtest import evaluate_batch
    c=case();c.net[19]=np.nan;c.ranks[5,2]=np.nan;c.amount20[18]=0
    cs=candidate_registry(cfg);bat=evaluate_batch(c,cs,cfg)
    for j,item in enumerate(cs):
        a=evaluate(c,item['weights'],cfg)
        for key in ('net','loss_floor','filled_weight','feature_coverage'):
            assert (np.isnan(a[key]) and np.isnan(bat[key][j])) or a[key]==pytest.approx(bat[key][j])
