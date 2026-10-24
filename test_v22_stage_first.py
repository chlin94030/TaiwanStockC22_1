from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd
import pytest
from lab_backtest import Case
from lab_config import Config
from v22_regime import classify_stage,ACTIONS
from v22_pool import build_v22_pools
from v22_engine import V22Case,validate_profiles,evaluate
from v22_runner import _export,V22_VERSION


def ctx(**kw):
    base=dict(market_drawdown60=-.02,market_bias20=.02,market_above200=1,market_ret5=.01,market_ret20=.03,market_ret60=.06,
        market_amount_ratio5_60=1.,market_vol_ratio=1.,breadth=.6,breadth_mom20=.6,breadth_mom60=.6,median_demand_balance=.1,median_amount_acceleration=0.)
    base.update(kw);return base

@pytest.mark.parametrize('expected,row',[('A1',ctx(market_drawdown60=-.10,market_bias20=-.01,market_above200=0,market_ret5=.03,market_ret20=-.01,market_ret60=-.05,market_amount_ratio5_60=.65,market_vol_ratio=.9,breadth=.48,breadth_mom20=.62,breadth_mom60=.4,median_demand_balance=.4,median_amount_acceleration=-.25)),
('A2',ctx(market_drawdown60=-.02,market_bias20=.04,market_above200=1,market_ret5=.02,market_ret20=.08,market_ret60=.15,market_amount_ratio5_60=1.,market_vol_ratio=1.,breadth=.72,breadth_mom20=.7,breadth_mom60=.68,median_demand_balance=.3,median_amount_acceleration=.05)),
('A3',ctx(market_drawdown60=0,market_bias20=.15,market_above200=1,market_ret5=.03,market_ret20=.12,market_ret60=.25,market_amount_ratio5_60=2.,market_vol_ratio=1.5,breadth=.85,breadth_mom20=.82,breadth_mom60=.78,median_demand_balance=.4,median_amount_acceleration=.6)),
('B1',ctx(market_drawdown60=-.05,market_bias20=.02,market_above200=1,market_ret5=-.04,market_ret20=-.05,market_ret60=.05,market_amount_ratio5_60=.7,market_vol_ratio=1.1,breadth=.48,breadth_mom20=.35,breadth_mom60=.55,median_demand_balance=-.1,median_amount_acceleration=-.2)),
('B2',ctx(market_drawdown60=-.12,market_bias20=-.08,market_above200=0,market_ret5=-.03,market_ret20=-.10,market_ret60=-.18,market_amount_ratio5_60=1.3,market_vol_ratio=1.3,breadth=.25,breadth_mom20=.22,breadth_mom60=.25,median_demand_balance=-.4,median_amount_acceleration=.3)),
('B3',ctx(market_drawdown60=-.22,market_bias20=-.12,market_above200=0,market_ret5=.03,market_ret20=-.12,market_ret60=-.25,market_amount_ratio5_60=2.1,market_vol_ratio=1.8,breadth=.18,breadth_mom20=.2,breadth_mom60=.15,median_demand_balance=-.5,median_amount_acceleration=.6))])
def test_six_stage_examples(expected,row):
    assert classify_stage(row)['stage']==expected


def test_unknown_is_cash():
    r=classify_stage({});assert r['stage']=='UNKNOWN' and r['exposure']==0 and r['action']==ACTIONS['UNKNOWN']


def test_pool_build_declared_sizes():
    rows=[]
    for d in pd.to_datetime(['2022-01-31','2022-02-28']):
        for i in range(500):
            rows.append({'date':d,'ticker':f'{1000+i}.TW','amount252':1e9-i*1e6,'amount20':(1e9-i*1e6)*(0.5+(i%50)/50),
                'pool':'ADV500','declared_size':500})
    out=build_v22_pools(pd.DataFrame(rows))
    assert set(out.pool)=={'ADV200','ADV300','KOSTO250','KOSTO300'}
    for (d,p),g in out.groupby(['date','pool']):assert g.ticker.nunique()==int(''.join(x for x in p if x.isdigit()))


def profile():
    obj={'schema':2,'candidates':[{'name':'x','stage_weights':{'A1':{'ks_double_bottom':1.0},'B3':{'ks_double_bottom':1.0}},
        'stage_exposure':{'A1':1.,'A2':0.,'A3':0.,'B1':0.,'B2':0.,'B3':.6,'UNKNOWN':0.},'discipline':True}]}
    return validate_profiles(obj,['ks_double_bottom'])[0]


def case(stage='A1',confirm=1.):
    n=20;d=pd.Timestamp('2024-01-31');factors=('ks_double_bottom',)
    ranks=np.linspace(1,.05,n).reshape(-1,1)
    c=Case(d,np.array([f'{1000+i}.TW' for i in range(n)]),ranks,np.ones(n)*1e9,np.linspace(.1,-.05,n),np.ones(n,dtype=bool),
        np.ones(n)*-.03,.02,.01,0.,d+pd.Timedelta(days=1),d+pd.Timedelta(days=30),factors)
    return V22Case(c,np.array(['S'+str(i%5) for i in range(n)]),np.ones(n)*50,np.zeros(n),np.zeros(n),np.ones(n)*confirm,np.ones(n)*confirm,stage,.5,ACTIONS[stage])


def spec():
    return {'pool_rule':'KOSTO300','top_k':10,'capital':1e6,'stage_confidence_floor':.03,'sector_cap':.3,'max_rsi':84.,'max_bias20':.14,'b3_confirmation_rank':.45}


def test_stage_gate_dominates_stock_score():
    cfg=Config(end='2025-01-01',top_k=10)
    assert evaluate(case('A3'),profile(),cfg,spec())['selected']==[]
    assert evaluate(case('B2'),profile(),cfg,spec())['selected']==[]
    a=evaluate(case('A1'),profile(),cfg,spec());assert len(a['selected'])>0 and a['stage_exposure']==1


def test_b3_needs_confirmation():
    cfg=Config(end='2025-01-01',top_k=10)
    assert evaluate(case('B3',.2),profile(),cfg,spec())['selected']==[]
    assert len(evaluate(case('B3',.8),profile(),cfg,spec())['selected'])>0


def test_future_returns_do_not_change_selection():
    cfg=Config(end='2025-01-01',top_k=10);c=case('A1');a=evaluate(c,profile(),cfg,spec());c.base.net[:]=999;b=evaluate(c,profile(),cfg,spec())
    assert a['selected']==b['selected']


def test_export_contains_full_adjustment_report(tmp_path):
    run=tmp_path;pd.DataFrame([{'x':1}]).to_csv(run/'candidate_ranking.csv',index=False)
    result={'models':{'5':{'pool':'KOSTO300','choice':{'name':'x'},'choice_score':.01,'verdict':'DEVELOPMENT_ONLY_NOT_ACCEPTANCE','final_tail_inspected':False,'stage_performance':[]}}}
    _export(run,result,[run/'candidate_ranking.csv'])
    with zipfile.ZipFile(run/'V22_TEST_REPORT.zip') as z:
        assert z.testzip() is None and {'REPORT.txt','REPORT.md','results.json','candidate_ranking.csv','EXPORT_SHA256.json'}<=set(z.namelist())
