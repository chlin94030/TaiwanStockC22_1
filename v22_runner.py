"""V22 fast stage-first research loop and one-time pool calibration.
Fast mode never inspects the final tail. Repeated tuning is explicitly exposed development.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import fields, replace
import json,time,zipfile
import numpy as np
import pandas as pd
from lab_config import Config
from lab_io import DataError,PauseRequested,JobLock,atomic_json,read_json,frame_write,sha
from lab_search import matured_training_indices,summarize_rows,verdict
from lab_stats import finite_mean
from studio_pool import load_dataset,digest,timestamp
from studio_cache import load_labels,algorithm_hash
from v22_cache import prepare_v22
from v22_engine import make_cases,validate_profiles,cached_rows,evaluate
from studio_runner import assert_confirmation_unused

V22_VERSION='22.1-kostolany-fast-bootstrap'
DEFAULT_SPEC={'schema':1,'pool_rule':'KOSTO300','top_k':10,'capital':1000000.0,'fast_folds':3,'bootstrap_samples':300,
    'stage_confidence_floor':.03,'sector_cap':.30,'max_rsi':84.,'max_bias20':.14,'b3_confirmation_rank':.45,
    'cost_overrides':{}}
ALLOWED=set(DEFAULT_SPEC)


def configure(meta,spec):
    if set(spec)-ALLOWED:raise DataError('Unknown V22 setting: '+str(sorted(set(spec)-ALLOWED)))
    s={**DEFAULT_SPEC,**spec}
    if s['schema']!=1 or not isinstance(s['top_k'],int) or not 5<=s['top_k']<=30:raise DataError('V22 uses 5..30 funded slots')
    if not 1<=s['fast_folds']<=8 or not 100<=s['bootstrap_samples']<=3000:raise DataError('Invalid fast validation budget')
    if not 0<=s['stage_confidence_floor']<=.5 or not .1<=s['sector_cap']<=1:raise DataError('Invalid stage/sector guard')
    if not 50<=s['max_rsi']<=100 or not 0<=s['max_bias20']<=1 or not 0<=s['b3_confirmation_rank']<=1:raise DataError('Invalid discipline threshold')
    allowed={f.name for f in fields(Config)};base={k:v for k,v in meta['source_config'].items() if k in allowed}
    for k,v in s['cost_overrides'].items():
        if k not in {'fee_buy','fee_sell','stock_tax','benchmark_tax','slippage','max_gap','extra_roundtrip_stress'}:raise DataError('Unsupported cost override')
        if not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1:raise DataError('Invalid cost override')
    cfg=replace(Config(**base),top_k=s['top_k'],capital=float(s['capital']),**s['cost_overrides']).resolved()
    return cfg,s


def _score_summary(rows,cfg):
    sm=summarize_rows(rows,replace(cfg,block_bootstrap_samples=max(100,min(500,cfg.block_bootstrap_samples))))
    a=sm.get('versus_0050',{}).get('mean');b=sm.get('versus_same_pool',{}).get('mean');stress=sm.get('extra_cost_stress_excess')
    vals=[x for x in (a,b,stress) if x is not None and np.isfinite(x)]
    score=min(vals) if len(vals)==3 else -999.
    return sm,float(score)


def _candidate_table(candidates,records,test_ids,cfg,h,pool):
    rows=[]
    for c in candidates:
        rr=[records[c['id']][i] for i in test_ids]
        sm,score=_score_summary(rr,cfg)
        rows.append({'horizon':h,'pool':pool,'name':c['name'],'candidate_id':c['id'],'score':score,
            'excess_0050':sm.get('versus_0050',{}).get('mean'),'excess_same_pool':sm.get('versus_same_pool',{}).get('mean'),
            'extra_cost_stress':sm.get('extra_cost_stress_excess'),'stock_capital_fraction':sm.get('stock_capital_fraction'),
            'known_0050':sm.get('versus_0050',{}).get('complete'),'planned':sm.get('versus_0050',{}).get('planned'),
            'scope':'EXPOSED_DEVELOPMENT_DIAGNOSTIC'})
    return sorted(rows,key=lambda x:(-x['score'],x['name']))


def _stage_table(rows):
    out=[]
    for stage in ('A1','A2','A3','B1','B2','B3','UNKNOWN'):
        q=[r for r in rows if r['stage']==stage]
        if not q:continue
        out.append({'stage':stage,'planned':len(q),'known':sum(np.isfinite(r['excess_0050']) for r in q),
            'excess_0050':finite_mean([r['excess_0050'] for r in q]),'excess_same_pool':finite_mean([r['excess_pool'] for r in q]),
            'mean_stock_exposure':finite_mean([r['stock_filled_weight'] for r in q]),'book_action':q[0]['stage_action']})
    return out


def _factor_sensitivity(cases,candidate,indices,cfg,spec):
    rows=[]
    base=[evaluate(cases[int(i)],candidate,cfg,spec) for i in indices]
    base_ex=finite_mean([r['excess_0050'] for r in base])
    for stage,w in candidate['stage_weights'].items():
        if len(w)<=1:continue
        for factor in w:
            nw={k:v for k,v in w.items() if k!=factor};total=sum(nw.values());nw={k:v/total for k,v in nw.items()}
            alt={**candidate,'stage_weights':{**candidate['stage_weights'],stage:nw},'id':candidate['id']+'_abl_'+stage+'_'+factor}
            rr=[evaluate(cases[int(i)],alt,cfg,spec) for i in indices]
            alt_ex=finite_mean([r['excess_0050'] for r in rr])
            rows.append({'stage':stage,'removed_factor':factor,'base_excess_0050':base_ex,'without_factor_excess_0050':alt_ex,
                'delta_full_minus_removed':None if base_ex is None or alt_ex is None else base_ex-alt_ex,
                'scope':'EXPOSED_DEVELOPMENT_SENSITIVITY_NOT_CAUSAL'})
    return rows


def _transition_table(context,pool):
    x=context[context.pool==pool].sort_values('date');pairs={}
    prev=None
    for s in x.stage:
        if prev is not None:pairs[(prev,s)]=pairs.get((prev,s),0)+1
        prev=s
    return [{'from_stage':a,'to_stage':b,'count':n} for (a,b),n in sorted(pairs.items())]


def _export(run,result,files):
    atomic_json(run/'results.json',result)
    data_scope=result.get('data_scope',{})
    proxy_note=('TURNOVER SOURCE: DEVELOPMENT PROXY ((High+Low+Close)/3)*Volume; NOT exchange-reported Trading_money.'
                if data_scope.get('development_turnover_proxy') else 'TURNOVER SOURCE: exchange-reported / licensed actual traded amount.')
    lines=['Alpha Radar V22.1 | Kostolany Stage-First Research','',
        'Core architecture: MARKET STAGE decides whether/how much to invest; STOCK FACTORS decide what to buy.',
        proxy_note,
        'A1/B3 are accumulation zones, A2 is hold/wait proxy, A3/B1 are sell/avoid-new-buy, B2 is cash/wait.',
        'Holder counts and news reactions are NOT observed; breadth/turnover are disclosed proxies.','']
    for h,m in result['models'].items():
        lines += [f'=== {h} sessions ===',f"Pool: {m['pool']}",f"Winner: {m['choice']['name']}",f"Development score: {m['choice_score']:+.4%}",
            f"Verdict: {m['verdict']}",f"Final tail inspected: {m['final_tail_inspected']}",
            'Stage performance: '+json.dumps(m['stage_performance'],ensure_ascii=False,default=lambda o:o.item() if hasattr(o,'item') else str(o)),'']
    lines += ['LIMITATIONS','- Book concepts are translated into testable proxies; they are not original formulas or guaranteed laws.',
        '- Turnover is not money supply; breadth is not literal holder-count evidence; news reaction is not tested without timestamped news.',
        '- Current-snapshot membership bias can remain unless a historical-membership dataset is supplied.',
        '- If YAHOO_FAST is used, turnover is a disclosed OHLCV proxy for development; exact pool confirmation requires an actual-amount source.',
        '- Fast mode is repeated development, not a fresh independent holdout.','- No automatic orders.']
    (run/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    md=['# Alpha Radar V22.1 — 完整快速測試報告','',*['- '+x for x in lines[2:6]],'']
    for h,m in result['models'].items():
        md += [f"## {h}日",f"- 母池：**{m['pool']}**",f"- 開發期排名第一：**{m['choice']['name']}**",f"- 快速開發分數：**{m['choice_score']:+.3%}**",f"- 判定：**{m['verdict']}**",'']
    (run/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    names=['REPORT.txt','REPORT.md','results.json']+[p.name for p in files if p.is_file()]
    atomic_json(run/'EXPORT_SHA256.json',{n:sha(run/n) for n in names})
    names.append('EXPORT_SHA256.json');tmp=run/'V22_TEST_REPORT.tmp.zip';out=run/'V22_TEST_REPORT.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(run/n,n)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip():raise DataError('V22 report ZIP CRC failed')
    tmp.replace(out)


def run_experiment(root,profiles,spec=None,mode='fast',progress=None,check=None):
    root=Path(root);spec=spec or DEFAULT_SPEC;start=time.perf_counter()
    if mode not in ('fast','confirm'):raise DataError('V22 mode must be fast or confirm')
    _,meta=load_dataset(root);cfg,spec=configure(meta,spec);fdir,info,feature_hit=prepare_v22(root,progress or (lambda **kw:None),check or (lambda:None))
    if spec['pool_rule'] not in info['pools']:raise DataError('Unknown V22 pool rule '+spec['pool_rule'])
    snap=pd.read_csv(fdir/'features.csv.gz',parse_dates=['date']);pools=pd.read_csv(fdir/'pools.csv.gz',parse_dates=['date']);context=pd.read_csv(fdir/'market_stage.csv.gz',parse_dates=['date'])
    candidates=validate_profiles(profiles,info['factors']);labels,label_info=load_labels(root,fdir,cfg,progress or (lambda **kw:None),check or (lambda:None))
    cases=make_cases(snap,labels,pools,context,spec['pool_rule'],cfg,info['factors'])
    identity={'version':V22_VERSION,'dataset':meta['dataset_id'],'features':info['feature_id'],'profiles':profiles,'settings':spec,'mode':mode,
              'code':algorithm_hash(('v22_runner.py','v22_engine.py','v22_regime.py','v22_pool.py','v22_cache.py','studio_behavior.py'))}
    eid=digest(identity)[:24];run=root/'v22'/'experiments'/eid;run.mkdir(parents=True,exist_ok=True)
    with JobLock(root/'worker.lock'):
        atomic_json(root/'v22'/'latest.json',{'experiment_id':eid,'run':str(run)})
        if read_json(run/'state.json').get('status')=='COMPLETE':return run
        confirm_meta=None
        if mode=='confirm':
            era=digest({str(h):[str(x.base.date) for x in cc[-cfg.holdout_months:]] for h,cc in cases.items()})[:24]
            windows=sorted(set((str(x.base.date.date()),str(x.base.exit.date())) for cc in cases.values() for x in cc[-cfg.holdout_months:] if pd.notna(x.base.exit)))
            ledger=read_json(root/'v22'/'confirmation_ledger.json')
            assert_confirmation_unused(ledger,eid,era,windows)
            confirm_meta=(era,windows,ledger)
        atomic_json(run/'state.json',{'status':'RUNNING','phase':'FAST','started_at':timestamp()});atomic_json(run/'profiles.json',profiles);atomic_json(run/'settings.json',spec)
        results={};rank_export=[];stage_export=[];decision_export=[];sensitivity=[];failure_export=[];tail_export=[];hits=miss=0
        calendar=pd.DatetimeIndex(read_json(Path(root)/'studio'/'datasets'/meta['dataset_id']/'calendar.json'))
        for h,cc in cases.items():
            dates=pd.DatetimeIndex([x.base.date for x in cc]);exits=pd.DatetimeIndex([x.base.exit for x in cc])
            if len(cc)<cfg.holdout_months+cfg.min_train_months+6:raise DataError('Not enough V22 monthly history')
            boundary=dates[len(cc)-cfg.holdout_months];dev=matured_training_indices(dates,exits,boundary,calendar,cfg.gap_sessions)
            allfolds=[]
            for t in range(cfg.min_train_months,len(dev),cfg.outer_test_months):
                test=dev[t:t+cfg.outer_test_months]
                if len(test)>=3:allfolds.append(test)
            folds=allfolds[-spec['fast_folds']:] if mode=='fast' else allfolds;test_ids=sorted(set(int(i) for f in folds for i in f))
            records={}
            for j,c in enumerate(candidates):
                if check:check()
                if progress:progress(phase='V22_FAST',done=j+1,total=len(candidates),message=f'{h}d {c["name"]}')
                rr,hit=cached_rows(root,cc,c,cfg,spec,{'feature':info['feature_id'],'labels':label_info['label_id'],'h':h},dev,check or (lambda:None));hits+=hit;miss+=not hit
                records[c['id']]={int(i):r for i,r in zip(dev,rr)}
            table=_candidate_table(candidates,records,test_ids,cfg,h,spec['pool_rule']);rank_export+=table
            for row in table:
                reasons=[]
                if row.get('excess_0050') is None or not np.isfinite(row.get('excess_0050')):reasons.append('NO_0050_EVIDENCE')
                elif row['excess_0050']<=0:reasons.append('NONPOSITIVE_VS_0050')
                if row.get('excess_same_pool') is None or not np.isfinite(row.get('excess_same_pool')):reasons.append('NO_SAME_POOL_EVIDENCE')
                elif row['excess_same_pool']<=0:reasons.append('NONPOSITIVE_VS_SAME_POOL')
                if row.get('extra_cost_stress') is None or not np.isfinite(row.get('extra_cost_stress')):reasons.append('NO_COST_STRESS_EVIDENCE')
                elif row['extra_cost_stress']<=0:reasons.append('NONPOSITIVE_AFTER_EXTRA_COST')
                if row.get('planned') and (row.get('known_0050') or 0)/row['planned']<cfg.min_fold_coverage:reasons.append('INSUFFICIENT_COVERAGE')
                failure_export.append({'horizon':h,'pool':spec['pool_rule'],'candidate':row['name'],'score':row['score'],'reasons':'|'.join(reasons) if reasons else 'NO_BASIC_BLOCKER'})
            choice_row=table[0];choice=next(c for c in candidates if c['id']==choice_row['candidate_id'])
            chosen_rows=[records[choice['id']][i] for i in test_ids];sm=summarize_rows(chosen_rows,replace(cfg,block_bootstrap_samples=spec['bootstrap_samples']))
            st=_stage_table(chosen_rows);stage_export += [dict(x,horizon=h,candidate=choice['name'],pool=spec['pool_rule']) for x in st]
            for r in chosen_rows:
                decision_export.append({'horizon':h,'date':r['date'],'pool':spec['pool_rule'],'stage':r['stage'],'stage_confidence':r['stage_confidence'],
                    'action':r['stage_action'],'candidate':choice['name'],'selected':'|'.join(r['selected']),'state':r['state'],'net':r['net'],
                    'benchmark':r['benchmark'],'excess_0050':r['excess_0050'],'stock_exposure':r['stock_filled_weight']})
            sensitivity += [dict(x,horizon=h,pool=spec['pool_rule'],candidate=choice['name']) for x in _factor_sensitivity(cc,choice,test_ids,cfg,spec)]
            result={'pool':spec['pool_rule'],'choice':choice,'choice_score':choice_row['score'],'development_validation':sm,'stage_performance':st,
                'verdict':'DEVELOPMENT_ONLY_NOT_ACCEPTANCE','final_tail_inspected':False,'book_stage_gate_dominant':True}
            if mode=='confirm':
                tail_idx=np.arange(len(cc)-cfg.holdout_months,len(cc));tail=[evaluate(cc[int(i)],choice,cfg,spec) for i in tail_idx]
                for r in tail:
                    tail_export.append({'horizon':h,'date':r['date'],'pool':spec['pool_rule'],'stage':r['stage'],'stage_confidence':r['stage_confidence'],
                        'action':r['stage_action'],'candidate':choice['name'],'selected':'|'.join(r['selected']),'state':r['state'],'net':r['net'],
                        'benchmark':r['benchmark'],'excess_0050':r['excess_0050'],'stock_exposure':r['stock_filled_weight']})
                result['final_confirmation']=summarize_rows(tail,replace(cfg,block_bootstrap_samples=spec['bootstrap_samples']));result['final_tail_inspected']=True
                result['verdict']=('PROXY_DATA_CONFIRMATION_NOT_FINAL' if meta.get('scope',{}).get('development_turnover_proxy') else verdict(result['final_confirmation'],cfg))
            results[str(h)]=result
        frame_write(run/'candidate_ranking.csv',pd.DataFrame(rank_export));frame_write(run/'strategy_by_stage.csv',pd.DataFrame(stage_export));frame_write(run/'monthly_decisions.csv',pd.DataFrame(decision_export));frame_write(run/'factor_sensitivity.csv',pd.DataFrame(sensitivity));frame_write(run/'failure_analysis.csv',pd.DataFrame(failure_export))
        if tail_export:frame_write(run/'final_confirmation_records.csv',pd.DataFrame(tail_export))
        stages=context[context.pool==spec['pool_rule']].copy();frame_write(run/'market_stage.csv',stages);frame_write(run/'regime_transitions.csv',pd.DataFrame(_transition_table(context,spec['pool_rule'])))
        dq=pd.DataFrame([{'dataset_id':meta['dataset_id'],'feature_id':info['feature_id'],'pool':spec['pool_rule'],'feature_cache_hit':feature_hit,
            'strategy_cache_hits':hits,'strategy_cache_misses':miss,'daily_symbols_read':info['daily_symbols_read'],'http_requests':0,
            'holder_counts_observed':False,'news_reaction_observed':False,'macro_money_supply_observed':False}]);frame_write(run/'data_quality.csv',dq)
        result={'version':V22_VERSION,'mode':mode,'models':results,'pool_rule':spec['pool_rule'],'data_scope':meta.get('scope',{}),'cache':{'feature_hit':feature_hit,'strategy_hits':hits,'strategy_misses':miss},
            'book_translation':{'market_stage_dominates_exposure':True,'A1_B3_buy':True,'A2_hold_wait_proxy':True,'A3_B1_sell':True,'B2_cash_wait':True,
                'holder_counts_are_breadth_turnover_proxies':True,'news_reaction_not_tested':True,'money_supply_not_inferred_from_turnover':True},
            'elapsed_seconds':time.perf_counter()-start,'automatic_orders_authorized':False,'global_optimum_proven':False}
        files=[run/n for n in ('profiles.json','settings.json','candidate_ranking.csv','strategy_by_stage.csv','monthly_decisions.csv','factor_sensitivity.csv','failure_analysis.csv','market_stage.csv','regime_transitions.csv','data_quality.csv')]
        if (run/'final_confirmation_records.csv').is_file():files.append(run/'final_confirmation_records.csv')
        _export(run,result,files)
        if confirm_meta is not None:
            era,windows,ledger=confirm_meta
            ledger[era]={'experiment_id':eid,'released_at':timestamp(),'holding_windows':windows,'pool_rule':spec['pool_rule'],'version':V22_VERSION}
            atomic_json(root/'v22'/'confirmation_ledger.json',ledger)
        atomic_json(run/'state.json',{'status':'COMPLETE','phase':'DONE','updated_at':timestamp()})
        return run


def run_pool_study(root,profiles,spec=None):
    """One-time training/development comparison. Never reads the final tail."""
    root=Path(root);_,meta=load_dataset(root);cfg,spec=configure(meta,spec or DEFAULT_SPEC);fdir,info,_=prepare_v22(root)
    snap=pd.read_csv(fdir/'features.csv.gz',parse_dates=['date']);pools=pd.read_csv(fdir/'pools.csv.gz',parse_dates=['date']);context=pd.read_csv(fdir/'market_stage.csv.gz',parse_dates=['date']);labels,_=load_labels(root,fdir,cfg)
    candidates=validate_profiles(profiles,info['factors']);base=candidates[0];calendar=pd.DatetimeIndex(read_json(root/'studio'/'datasets'/meta['dataset_id']/'calendar.json'))
    rows=[]
    for pool in info['pools']:
        cases=make_cases(snap,labels,pools,context,pool,cfg,info['factors'])
        for h,cc in cases.items():
            dates=pd.DatetimeIndex([x.base.date for x in cc]);exits=pd.DatetimeIndex([x.base.exit for x in cc]);boundary=dates[len(cc)-cfg.holdout_months]
            dev=matured_training_indices(dates,exits,boundary,calendar,cfg.gap_sessions);rr=[evaluate(cc[int(i)],base,cfg,spec or DEFAULT_SPEC) for i in dev]
            sm,score=_score_summary(rr,cfg);rows.append({'pool':pool,'horizon':h,'candidate':base['name'],'score':score,
                'excess_0050':sm.get('versus_0050',{}).get('mean'),'excess_same_pool':sm.get('versus_same_pool',{}).get('mean'),
                'extra_cost_stress':sm.get('extra_cost_stress_excess'),'dates':len(dev),'tail_used':False})
    frame=pd.DataFrame(rows);summary=frame.groupby('pool').score.agg(['mean','min']).reset_index().sort_values(['min','mean','pool'],ascending=[False,False,True])
    folder=root/'v22'/'pool_study'/digest({'dataset':meta['dataset_id'],'candidate':base['id'],'code':algorithm_hash(('v22_pool.py','v22_regime.py','v22_engine.py'))})[:24];folder.mkdir(parents=True,exist_ok=True)
    frame_write(folder/'POOL_STUDY.csv',frame);frame_write(folder/'POOL_SUMMARY.csv',summary)
    report=['V22 one-time pool calibration','Final tail used: False','Baseline profile: '+base['name'],'',summary.to_string(index=False),'',
        'Recommendation is training/development-only, not proof of an optimal pool. Lock one rule and stop rerunning this step during strategy tuning.']
    (folder/'POOL_REPORT.txt').write_text('\n'.join(report)+'\n',encoding='utf-8-sig')
    with zipfile.ZipFile(folder/'V22_POOL_REPORT.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in ('POOL_STUDY.csv','POOL_SUMMARY.csv','POOL_REPORT.txt'):z.write(folder/n,n)
    atomic_json(root/'v22'/'latest_pool_study.json',{'run':str(folder),'suggested_pool':summary.iloc[0].pool if len(summary) else None})
    return folder

# Streamlit helper: computation runs independently of the browser rerun loop on
# the current host. Persistent cross-host/background execution still belongs to
# GitHub Actions; this helper is only for responsive UI.
import threading
_THREADS={};_THREAD_LOCK=threading.Lock()

def start_v22(root,profiles,spec,mode='fast',pool_study=False):
    root=Path(root);key=str(root.resolve())
    with _THREAD_LOCK:
        if _THREADS.get(key) and _THREADS[key].is_alive():raise DataError('A V22 task is already running')
        def target():
            try:
                if pool_study:run_pool_study(root,profiles,spec)
                else:run_experiment(root,profiles,spec,mode)
            except Exception as exc:
                atomic_json(root/'v22'/'launch_error.json',{'type':type(exc).__name__,'message':str(exc)[:500],'at':timestamp()})
        (root/'v22').mkdir(parents=True,exist_ok=True);atomic_json(root/'v22'/'editor.json',{'profiles':profiles,'settings':spec,'mode':mode})
        (root/'v22'/'launch_error.json').unlink(missing_ok=True)
        t=threading.Thread(target=target,daemon=True,name='v22-stage-studio');_THREADS[key]=t;t.start()

def v22_alive(root):
    t=_THREADS.get(str(Path(root).resolve()));return bool(t and t.is_alive())
