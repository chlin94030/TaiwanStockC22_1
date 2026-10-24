"""Engineering/adversarial fixtures. No market-performance claims."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from lab_config import Config,FACTORS,VERSION
from lab_candidates import candidate_registry
from lab_features import features,raw_adjusted_frame,return_record
from lab_price_quality import analytic_hlc_from_audit
from lab_approval import reference_choice,is_reference
from lab_backtest import Case,evaluate,evaluate_batch,build_cases
from lab_search import choose_inner,record_choice,summarize_rows
from lab_robust import score_training
from lab_upgrade import run_config
from lab_io import Store,atomic_json,read_json


def config(**kw):
    return Config(end='2026-10-20',min_train_months=12,block_bootstrap_samples=30,**kw)


def stock_frame(logreturns):
    close=100*np.exp(np.cumsum(logreturns));idx=pd.bdate_range('2018-01-01',periods=len(close))
    return pd.DataFrame({'raw_open':close,'raw_high':close*1.01,'raw_low':close*.99,'raw_close':close,
                        'adj_open':close,'adj_high':close*1.01,'adj_low':close*.99,'adj_close':close,
                        'amount':1e8,'volume':1e6},index=idx)


def makecase(benchmark=.03):
    n=30
    ranks=np.column_stack([np.arange(n)/n for _ in FACTORS])
    return Case(pd.Timestamp('2024-01-31'),np.array([str(1000+i)+'.TW' for i in range(n)]),
                ranks,np.repeat(1e9,n),np.linspace(.01,.1,n),np.ones(n,bool),np.repeat(-.02,n),
                benchmark,.04,0.,pd.Timestamp('2024-02-01'),pd.Timestamp('2024-02-08'),FACTORS)


def test_default_ten_slots_new_model_not_old_five():
    c=config();assert c.top_k==10 and len(FACTORS)==18
    r=evaluate(makecase(),{'liquidity20':1},c)
    assert len(r['selected'])==10 and r['net']==pytest.approx(makecase().net[-10:].mean())


def test_reference_zero_stock_cost_and_not_earned_alpha():
    c=config();r=record_choice(reference_choice(),[makecase()],[0],c)
    s=summarize_rows(r,c)
    assert r[0]['filled_weight']==1 and r[0]['stock_filled_weight']==0
    assert s['extra_cost_stress_excess']==0 and s['stock_capital_fraction']==0
    assert s['versus_0050']['mean']==0 and s['versus_same_pool']['complete']==0


def test_active_stress_still_subtracts_real_stock_exposure():
    c=config();choice={'candidate_id':'active','weights':{'liquidity20':1},'pool':'ADV100'}
    r=record_choice(choice,[makecase()],[0],c);s=summarize_rows(r,c)
    assert s['extra_cost_stress_excess']==pytest.approx(r[0]['excess_0050']-.005)


def test_missing_reference_remains_unknown_under_stress():
    c=config();r=record_choice(reference_choice(),[makecase(np.nan)],[0],c)
    assert summarize_rows(r,c)['extra_cost_stress_excess'] is None


def test_old_winner_cannot_win_with_negative_recent_training():
    c=config(recent_train_months=12)
    cs=candidate_registry(c);a=np.repeat(np.r_[np.repeat(.15,24),np.repeat(-.02,12)][:,None],len(cs),axis=1)
    g={'net':a,'floor':a.copy(),'fills':np.ones_like(a),'benchmark':np.zeros(36)}
    winner,board=choose_inner({('ADV100',20):g},cs,20,np.arange(36),c)
    assert is_reference(winner) and all(r['recent_train_stress_excess']<0 for r in board)


def test_recent_maximin_ignores_unused_tail():
    c=config();cs=candidate_registry(c);a=np.ones((48,len(cs)))*.04
    g={'net':a.copy(),'floor':a.copy(),'fills':a*0+1,'benchmark':np.zeros(48)}
    w,b=choose_inner({('ADV100',20):g},cs,20,np.arange(24),c)
    g['net'][24:]=-1e5;g['floor'][24:]=1e7;g['benchmark'][24:]=999
    ww,bb=choose_inner({('ADV100',20):g},cs,20,np.arange(24),c)
    assert w==ww and b==bb


def test_training_indices_not_random_shuffled():
    c=config();a=np.ones((24,1))*.04;g={'net':a,'floor':a,'fills':a*0+1,'benchmark':np.zeros(24)}
    from lab_io import DataError
    with pytest.raises(DataError,match='chronological'):
        score_training(g,0,np.arange(24)[::-1],{'liquidity20':1},c)


def test_pool_equivalence_hash_only_uses_training():
    c=config();cs=candidate_registry(c);a=np.ones((30,len(cs)))*.05
    g={'net':a.copy(),'floor':a.copy(),'fills':np.ones_like(a),'benchmark':np.zeros(30)}
    g2=copy.deepcopy(g);g2['net'][24:]=99
    _,board=choose_inner({('ADV100',20):g,('ADV500',20):g2},cs,20,np.arange(24),c)
    matches=[r for r in board if r['candidate_index']==0]
    assert matches[0]['training_outcome_equivalence']==matches[1]['training_outcome_equivalence']


def test_residual_features_remove_estimated_beta_not_common_constant():
    rng=np.random.default_rng(5);market=rng.normal(.0002,.01,400)
    b=stock_frame(market);f=stock_frame(2*market+.001)
    o=features(f,b)
    assert o.benchmark_beta120.iloc[-1]==pytest.approx(2,abs=1e-10)
    assert o.residual_momentum20.iloc[-1]==pytest.approx(.02,abs=1e-10)
    assert o.residual_momentum60.iloc[-1]==pytest.approx(.06,abs=1e-10)
    assert o.residual_lowvol60.iloc[-1]==pytest.approx(0,abs=1e-10)


def test_stock_and_benchmark_future_mutation_cannot_change_past_features():
    rng=np.random.default_rng(9);b=stock_frame(rng.normal(0,.01,400));f=stock_frame(rng.normal(.001,.02,400))
    a=features(f,b);f2=f.copy();b2=b.copy()
    f2.loc[f.index[330]:,f.columns.str.startswith('adj_')]*=100
    b2.loc[b.index[330]:,b.columns.str.startswith('adj_')]*=.01
    out=features(f2,b2)
    pd.testing.assert_frame_equal(a.iloc[:330],out.iloc[:330])


def test_current_extreme_return_cannot_change_same_day_estimated_beta():
    rng=np.random.default_rng(2);m=rng.normal(0,.01,400);b=stock_frame(m);f=stock_frame(m*1.5)
    before=features(f,b);f.loc[f.index[-1],'adj_close']*=3
    after=features(f,b)
    assert before.benchmark_beta120.iloc[-1]==pytest.approx(after.benchmark_beta120.iloc[-1])


def test_no_market_history_does_not_fake_residual_signal():
    f=stock_frame(np.ones(400)*.001);out=features(f)
    assert out.residual_momentum20.isna().all()
    assert features(f,stock_frame(np.zeros(400))).residual_momentum60.isna().all()


def test_residual_window_does_not_fill_missing_market_price():
    rng=np.random.default_rng(42);b=stock_frame(rng.normal(0,.01,400));f=stock_frame(rng.normal(0,.01,400))
    b.loc[b.index[-5],'adj_close']=np.nan
    assert pd.isna(features(f,b).residual_momentum20.iloc[-1])


def test_bad_open_can_not_erase_independently_valid_close():
    d=pd.to_datetime(['2024-01-31'])
    raw=pd.DataFrame({'date':d,'open':[120.],'high':[110.],'low':[90.],'close':[105.],'amount':[1e8],'volume':[1e6]})
    adj=pd.DataFrame({'date':d,'open':[100.],'high':[110.],'low':[90.],'close':[105.]})
    f=raw_adjusted_frame(raw,adj,d)
    assert pd.isna(f.raw_open.iloc[0]) and pd.isna(f.adj_open.iloc[0])
    assert f.adj_close.iloc[0]==105 and f.adj_high.iloc[0]==110


def test_close_outside_high_low_stays_unknown():
    d=pd.to_datetime(['2024-01-31']);raw=pd.DataFrame({'date':d,'open':[100.],'high':[110.],'low':[90.],'close':[200.],'amount':[1e8],'volume':[1e6]})
    adj=raw[['date','open','high','low','close']].copy()
    assert pd.isna(raw_adjusted_frame(raw,adj,d).adj_close.iloc[0])


def test_ledger_recovery_does_not_invent_execution_price():
    d=pd.to_datetime(['2024-01-31']);raw=pd.DataFrame({'date':d,'open':[np.nan],'high':[np.nan],'low':[np.nan],'close':[np.nan],'amount':[1e8],'volume':[1e6]})
    issue={'date':'2024-01-31','kind':'OHLC_RANGE_MISMATCH','observed':{'open':120.,'high':110.,'low':90.,'close':105.}}
    a,stat=analytic_hlc_from_audit(raw,[issue])
    adj=pd.DataFrame({'date':d,'open':[100.],'high':[110.],'low':[90.],'close':[105.]})
    f=raw_adjusted_frame(raw,adj,d,a)
    assert f.adj_close.iloc[0]==105 and stat['analytical_hlc_restored']==1
    assert pd.isna(f.raw_close.iloc[0]) and pd.isna(f.raw_open.iloc[0])
    assert raw.close.isna().all() and stat['execution_raw_modified'] is False


def test_conflicting_ledger_never_overwrites_new_raw_values():
    d=pd.to_datetime(['2024-01-31']);raw=pd.DataFrame({'date':d,'open':[np.nan],'high':[110.],'low':[90.],'close':[106.]})
    issue={'date':'2024-01-31','kind':'OHLC_RANGE_MISMATCH','observed':{'high':110.,'low':90.,'close':105.}}
    a,s=analytic_hlc_from_audit(raw,[issue]);assert a.close.iloc[0]==106 and s['analytical_hlc_restored']==0


def test_bad_open_in_high_history_still_requires_full_252_valid_highs():
    f=stock_frame(np.ones(400)*.001);f.loc[f.index[390],'adj_high']=np.nan
    assert pd.isna(features(f).high252.iloc[-1])


def test_registry_is_bounded_in_small_fixtures_too():
    assert len(candidate_registry(config(candidate_budget=20)))==20
    assert len(candidate_registry(config()))==64
    assert any('residual_momentum60' in c['weights'] for c in candidate_registry(config()))


def test_scalar_and_batch_parity_at_ten_slots():
    c=config();case=makecase();cs=candidate_registry(c)
    v=evaluate_batch(case,cs,c)
    for j,x in enumerate(cs):
        assert v['net'][j]==pytest.approx(evaluate(case,x['weights'],c)['net'])


def test_versions_reuse_raw_era_but_never_inherit_old_five_slot_config():
    c=config();old=replace(c,top_k=5,end='2026-10-19').to_dict()
    new=run_config(c,{'config':old},{'version':'21.3-evidence-guard'})
    assert new.end=='2026-10-19' and new.top_k==10 and new.selector==c.selector
    same=run_config(c,{'config':old},{'version':VERSION})
    assert same.end=='2026-10-19' and same.top_k==5


def test_different_source_mode_never_adopts_old_settings():
    c=config(source_mode='import');old=replace(c,source_mode='auto_current_roster',end='2026-10-19').to_dict()
    assert run_config(c,{'config':old},{'version':'21.3'})==c


def test_finite_global_price_missing_never_gets_zero_return():
    c=config();case=makecase();case.net[-1]=np.nan
    q=evaluate(case,{'liquidity20':1},c)
    assert np.isnan(q['net']) and q['missing_slots']==1 and q['loss_floor']<0


def test_snapshot_whitelist_includes_new_diagnostics_not_code():
    from lab_snapshot import allowed
    rid='a'*20
    assert allowed(f'runs/{rid}/price_quality_audit.csv')
    assert allowed(f'runs/{rid}/PRETAIL_GATES.json')
    assert not allowed(f'runs/{rid}/lab_robust.py')


def consensus_fixture(top_k=2):
    c=config(top_k=top_k,near_best_tolerance=.03)
    cs=[{'id':'a','name':'a','weights':{'liquidity20':1.},'complexity':1},
        {'id':'b','name':'b','weights':{'high252':1.},'complexity':1}]
    cc=[];gr=[]
    for i in range(24):
        ranks=np.zeros((4,len(FACTORS)));ranks[:,0]=[1.,.9,.2,.1];ranks[:,1]=[.2,.1,1.,.9]
        a=.04 if i//8!=1 else .08;b=.12-a
        date=pd.Timestamp('2018-01-31')+pd.DateOffset(months=i)
        case=Case(date,np.array(['1000.TW','1001.TW','1002.TW','1003.TW']),ranks,
                  np.repeat(1e9,4),np.array([a,a,b,b]),np.ones(4,bool),np.repeat(-.02,4),0.,.06,0.,
                  date+pd.Timedelta(days=1),date+pd.Timedelta(days=5),FACTORS)
        cc.append(case);gr.append(evaluate_batch(case,cs,c))
    g={name:np.array([r[name] for r in gr]) for name in ('net','loss_floor','filled_weight','feature_coverage')}
    g={'net':g['net'],'floor':g['loss_floor'],'fills':g['filled_weight'],
       'feature_coverage':g['feature_coverage'],'benchmark':np.zeros(24)}
    return c,cs,cc,g


def test_training_consensus_is_recomputed_actual_topk_not_return_average():
    c,cs,cc,g=consensus_fixture(2)
    w,board=choose_inner({('ADV100',5):g},cs,5,np.arange(24),c,{('ADV100',5):cc})
    assert w['candidate_index']==-2 and len(board)==3
    q=evaluate(cc[0],w['weights'],c)
    assert q['selected']==['1000.TW','1002.TW']
    assert q['net']==pytest.approx(.06)
    assert w['train_stress_excess']==pytest.approx(.055)
    assert w['selection_method']=='TRAINING_ONLY_CONSENSUS_RERANK'


def test_average_weights_do_not_claim_average_portfolio_returns():
    c,cs,cc,g=consensus_fixture(1)
    w,board=choose_inner({('ADV100',5):g},cs,5,np.arange(24),c,{('ADV100',5):cc})
    assert w['candidate_index']>=0
    ensemble=[r for r in board if r['candidate_index']==-2][0]
    # With one funded slot, averaged factor ranks choose the tie-breaking ticker,
    # NOT 50/50 ownership of the old a/b portfolios. Score is recomputed as such.
    actual=evaluate(cc[0],ensemble['weights'],c)['net']
    assert actual==pytest.approx(.04) and actual!=pytest.approx(.06)


def test_consensus_budget_and_future_outcome_mutation_invariance():
    c,cs,cc,g=consensus_fixture(2)
    w,board=choose_inner({('ADV100',5):g},cs,5,np.arange(24),c,{('ADV100',5):cc})
    cc.append(copy.deepcopy(cc[-1]));cc[-1].net[:]=100000
    gg={k:np.concatenate([v,v[-1:]],axis=0) for k,v in g.items()}
    gg['net'][-1]=1e9;gg['floor'][-1]=-1e9
    ww,bb=choose_inner({('ADV100',5):gg},cs,5,np.arange(24),c,{('ADV100',5):cc})
    assert w==ww and len(bb)==len(cs)+1
    assert len(w['weights'])<=c.max_active_factors


def test_dynamic_consensus_pool_equivalence_is_not_negative_array_index():
    from lab_approval import equivalent_training_pools
    c,cs,cc,g=consensus_fixture(2)
    w,_=choose_inner({('ADV100',5):g},cs,5,np.arange(24),c,{('ADV100',5):cc})
    eq=equivalent_training_pools({('ADV100',5):g,('ADV200',5):copy.deepcopy(g)},w,np.arange(24),
                                cases={('ADV100',5):cc,('ADV200',5):copy.deepcopy(cc)},cfg=c)
    assert eq==['ADV100','ADV200']


def test_new_version_does_not_auto_resume_unfinished_old_protocol(tmp_path,monkeypatch):
    import lab_jobs
    root=tmp_path/'work';run=root/'runs'/'old';run.mkdir(parents=True)
    atomic_json(root/'latest.json',{'run':str(run),'config':config(top_k=5).to_dict()})
    atomic_json(run/'protocol.json',{'version':'21.3-evidence-guard'})
    atomic_json(run/'state.json',{'status':'RUNNING'})
    monkeypatch.setattr(lab_jobs,'start',lambda *args:(_ for _ in ()).throw(AssertionError('Must explicitly start new protocol')))
    assert lab_jobs.resume_if_interrupted(root) is None


def test_upgrade_reuses_roster_only_with_exact_start_end(tmp_path):
    from lab_upgrade import reuse_collection_context
    c=config();old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    atomic_json(old/'roster.json',[{'ticker':'2330.TW','name':'original'}])
    atomic_json(old/'calendar.json',['2018-01-02','2018-01-03'])
    atomic_json(old/'results.json',{'do_not_copy':'old_alpha'})
    latest={'config':c.to_dict(),'run':str(old)}
    out=reuse_collection_context(latest,new,c)
    assert read_json(new/'roster.json')[0]['name']=='original' and out['reused_features_or_outcomes'] is False
    assert not (new/'results.json').exists()
    newer=tmp_path/'newer';newer.mkdir()
    assert reuse_collection_context(latest,newer,replace(c,end='2026-10-22'))=={}
    assert not (newer/'roster.json').exists()


def test_upgrade_rejects_invalid_roster_context(tmp_path):
    from lab_upgrade import reuse_collection_context
    c=config();old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    atomic_json(old/'roster.json',[{'ticker':'0050.TW'}])
    atomic_json(old/'calendar.json',['2018-01-02'])
    assert reuse_collection_context({'config':c.to_dict(),'run':str(old)},new,c)=={}
