"""V22 stage-first stock-selection engine.
Market stage decides whether/how much to invest; stock factors decide which names.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lab_backtest import Case
from lab_io import DataError, atomic_json, read_json, sha
from studio_pool import digest
from studio_cache import algorithm_hash

@dataclass
class V22Case:
    base: Case
    sectors: np.ndarray
    rsi: np.ndarray
    bias: np.ndarray
    double_top_rank: np.ndarray
    capitulation_rank: np.ndarray
    double_bottom_rank: np.ndarray
    stage: str
    confidence: float
    action: str


def validate_profiles(obj,available):
    if not isinstance(obj,dict) or obj.get('schema')!=2:raise DataError('V22 expects profiles schema 2')
    rows=obj.get('candidates',[])
    if not 1<=len(rows)<=32:raise DataError('Use 1..32 explicit V22 candidates')
    out=[];seen=set();names=set();stages={'A1','A2','A3','B1','B2','B3','UNKNOWN'}
    for c in rows:
        if set(c)-{'name','stage_weights','stage_exposure','discipline'}:raise DataError('Unknown V22 strategy setting')
        name=c.get('name');sw=c.get('stage_weights');se=c.get('stage_exposure');disc=bool(c.get('discipline',True))
        if not isinstance(name,str) or not name or name in names:raise DataError('Candidate names must be unique')
        if not isinstance(sw,dict) or not isinstance(se,dict) or set(se)!=stages:raise DataError('Stage exposure must define A1/A2/A3/B1/B2/B3/UNKNOWN')
        clean={}
        for stage,w in sw.items():
            if stage not in stages:raise DataError('Unknown stage in weights')
            if not isinstance(w,dict) or not 1<=len(w)<=4 or not set(w)<=set(available):raise DataError('Each active stage uses 1..4 available factors')
            q={k:float(v) for k,v in sorted(w.items())}
            if any(not np.isfinite(v) or v<=0 for v in q.values()) or abs(sum(q.values())-1)>1e-8:raise DataError('Positive stage weights must sum to 1')
            clean[stage]=q
        exp={k:float(v) for k,v in se.items()}
        if any(not np.isfinite(v) or not 0<=v<=1 for v in exp.values()):raise DataError('Stage exposures must be within [0,1]')
        for s,v in exp.items():
            if v>0 and s!='UNKNOWN' and s not in clean:raise DataError('Positive exposure requires weights for stage '+s)
        key=digest({'stage_weights':clean,'stage_exposure':exp,'discipline':disc})[:20]
        if key in seen:raise DataError('Duplicate V22 strategy hypothesis')
        seen.add(key);names.add(name);out.append({'id':key,'name':name,'stage_weights':clean,'stage_exposure':exp,'discipline':disc,
            'complexity':sum(len(x) for x in clean.values())})
    return out


def make_cases(snap,labels,pools,context,pool,cfg,factors):
    lookup={d:x.set_index('ticker') for d,x in snap.groupby('date')}
    outcomes={(d,int(h)):x.set_index('ticker') for (d,h),x in labels.groupby(['date','horizon'])}
    contexts={(x['pool'],pd.Timestamp(x['date'])):x for x in context.to_dict('records')}
    rows=pools[pools.pool==pool]
    if rows.empty:raise DataError('V22 pool absent: '+pool)
    cases={}
    for h in cfg.horizons:
        cc=[]
        for date,m in rows.groupby('date',sort=True):
            m=m.sort_values('ticker');t=m.ticker.to_numpy();s=lookup[date].reindex(t)
            if (date,h) not in outcomes:raise DataError('Matured label case missing')
            l=outcomes[(date,h)];ll=l.reindex(t);b=l.loc['0050.TW'];net=ll.net.to_numpy(float)
            ranks=s[['rank_'+k for k in factors]].to_numpy(float)
            base=Case(pd.Timestamp(date),t,ranks,m.amount20.to_numpy(float),net,ll.filled.fillna(False).to_numpy(bool),
                ll.path_mdd.to_numpy(float),float(b.net),float(net.mean()) if np.isfinite(net).all() else np.nan,
                float((~np.isfinite(net)).mean()),pd.Timestamp(b.entry),pd.Timestamp(b.exit),tuple(factors))
            cx=contexts[(pool,pd.Timestamp(date))]
            def col(name,default=np.nan):return s[name].to_numpy(float) if name in s else np.full(len(t),default)
            cc.append(V22Case(base,s.sector.fillna('UNKNOWN').astype(str).to_numpy() if 'sector' in s else np.repeat('UNKNOWN',len(t)),
                col('rsi14'),col('bias20'),col('rank_ks_double_top_risk'),col('rank_ks_capitulation_reversal'),col('rank_ks_double_bottom'),
                str(cx['stage']),float(cx['stage_confidence']),str(cx['stage_action'])))
        cases[h]=cc
    return cases


def evaluate(sc,candidate,cfg,spec):
    c=sc.base;stage=sc.stage;exposure=float(candidate['stage_exposure'].get(stage,0.0))
    base={'date':c.date,'entry':c.entry,'exit':c.exit,'benchmark':c.benchmark,'pool_net':c.pool_net,'pool_missing_weight':c.pool_missing_weight,
          'stage':stage,'stage_confidence':sc.confidence,'stage_action':sc.action,'candidate_id':candidate['id'],'candidate_name':candidate['name'],
          'stage_exposure':exposure,'market_stage_dominant':True,'decision_uses_future_outcomes':False,'is_reference':False}
    if stage=='UNKNOWN' or sc.confidence<spec['stage_confidence_floor']:
        return dict(base,net=0.,loss_floor=0.,filled_weight=0.,stock_filled_weight=0.,feature_coverage=1.,selected=[],core_selected=[],
            missing_slots=cfg.top_k,cash_slots=cfg.top_k,position_mdd=0.,state='STAGE_UNCERTAIN_CASH',excess_0050=-c.benchmark if np.isfinite(c.benchmark) else np.nan,
            excess_pool=-c.pool_net if np.isfinite(c.pool_net) else np.nan,exposure_matched_excess=0.)
    weights=candidate['stage_weights'].get(stage)
    if exposure<=0 or not weights:
        return dict(base,net=0.,loss_floor=0.,filled_weight=0.,stock_filled_weight=0.,feature_coverage=1.,selected=[],core_selected=[],
            missing_slots=cfg.top_k,cash_slots=cfg.top_k,position_mdd=0.,state='BOOK_STAGE_CASH_OR_SELL',excess_0050=-c.benchmark if np.isfinite(c.benchmark) else np.nan,
            excess_pool=-c.pool_net if np.isfinite(c.pool_net) else np.nan,exposure_matched_excess=0.)
    w=np.array([weights.get(k,0.) for k in c.features]);known=np.isfinite(c.ranks);full=(known*w).sum(1)>=1-1e-9
    coverage=float(full.mean())
    if coverage<cfg.min_feature_coverage:
        return dict(base,net=np.nan,loss_floor=-exposure,filled_weight=0.,stock_filled_weight=0.,feature_coverage=coverage,selected=[],core_selected=[],
            missing_slots=cfg.top_k,cash_slots=0,position_mdd=np.nan,state='INPUT_COVERAGE_BLOCKED',excess_0050=np.nan,excess_pool=np.nan,exposure_matched_excess=np.nan)
    capacity=np.isfinite(c.amount20)&(c.amount20*cfg.max_amount_participation>=cfg.capital/cfg.top_k)
    score=np.nan_to_num(c.ranks,nan=0.)@w;score=np.where(full&capacity,score,-np.inf)
    order=np.lexsort((c.tickers,-score));core=order[np.isfinite(score[order])][:cfg.top_k]
    chosen=[];counts={};cap=max(1,int(np.floor(cfg.top_k*spec['sector_cap'])))
    for i in core:
        sec=sc.sectors[i]
        if candidate['discipline']:
            if not np.isfinite(sc.rsi[i]) or not np.isfinite(sc.bias[i]) or sc.rsi[i]>spec['max_rsi'] or sc.bias[i]>spec['max_bias20']:continue
            if np.isfinite(sc.double_top_rank[i]) and sc.double_top_rank[i]>.90:continue
            if counts.get(sec,0)>=cap:continue
            if stage=='B3':
                confirm=max(sc.capitulation_rank[i] if np.isfinite(sc.capitulation_rank[i]) else 0,
                            sc.double_bottom_rank[i] if np.isfinite(sc.double_bottom_rank[i]) else 0)
                if confirm<spec['b3_confirmation_rank']:continue
        chosen.append(i);counts[sec]=counts.get(sec,0)+1
    vals=c.net[chosen];missing=int((~np.isfinite(vals)).sum());net=float(vals.sum()/cfg.top_k*exposure) if missing==0 else np.nan
    filled=float(c.filled[chosen].sum()/cfg.top_k*exposure);mdd=c.path_mdd[chosen]
    loss_floor=float(np.nansum(vals)/cfg.top_k*exposure-missing/cfg.top_k*exposure)
    pmdd=float(np.nanmean(mdd)) if len(mdd) and np.isfinite(mdd).any() else (0. if not chosen else np.nan)
    state='SELECTED_EXIT_UNKNOWN' if missing else ('SELECTED' if chosen else 'NO_CONFIRMED_SETUP')
    return dict(base,net=net,loss_floor=loss_floor,filled_weight=filled,stock_filled_weight=filled,feature_coverage=coverage,
        selected=c.tickers[chosen].tolist(),core_selected=c.tickers[core].tolist(),selected_scores=[float(score[i]) for i in chosen],
        missing_slots=cfg.top_k-len(chosen),cash_slots=cfg.top_k-len(chosen),position_mdd=pmdd,state=state,
        excess_0050=net-c.benchmark if np.isfinite(net) and np.isfinite(c.benchmark) else np.nan,
        excess_pool=net-c.pool_net if np.isfinite(net) and np.isfinite(c.pool_net) else np.nan,
        exposure_matched_excess=net-c.benchmark*exposure if np.isfinite(net) and np.isfinite(c.benchmark) else np.nan)


def _restore(rows):
    for r in rows:
        for k in ('date','entry','exit'):
            if k in r:r[k]=pd.Timestamp(r[k])
    return rows


def cached_rows(root,cases,candidate,cfg,spec,identity,indices,check=lambda:None):
    key=digest({'data':identity,'candidate':candidate['id'],'pool':spec['pool_rule'],
        'settings':{k:spec[k] for k in ('top_k','capital','stage_confidence_floor','sector_cap','max_rsi','max_bias20','b3_confirmation_rank')},
        'dates':[str(cases[int(i)].base.date) for i in indices],'code':algorithm_hash(('v22_engine.py',))})[:32]
    folder=Path(root)/'v22'/'strategy_cache';folder.mkdir(parents=True,exist_ok=True);path=folder/(key+'.json');meta=folder/(key+'.sha.json')
    if path.is_file() and meta.is_file():
        if sha(path)!=read_json(meta).get('sha256'):raise DataError('V22 strategy cache checksum mismatch')
        return _restore(read_json(path,[])),True
    rows=[]
    for i in indices:check();rows.append(evaluate(cases[int(i)],candidate,cfg,spec))
    atomic_json(path,rows);atomic_json(meta,{'sha256':sha(path)})
    return rows,False
