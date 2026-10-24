"""V22 mobile-friendly stage-first research studio."""
from pathlib import Path
import json
from lab_io import read_json,DataError
from studio_pool import discover_prepared,adopt_dataset,load_dataset
from v22_runner import start_v22,v22_alive,V22_VERSION

HERE=Path(__file__).resolve().parent

def render(st,root):
    root=Path(root);st.title('Alpha Radar V22｜科斯托蘭尼階段先決模型')
    st.caption(V22_VERSION+'｜市場階段先決，個股因子其次；快速測試不重新下載行情、不重做全市場母池。')
    st.info('A1／B3：買進區；A2：等待／續抱代理；A3／B1：賣出且不開新倉；B2：現金等待。持有人數以市場廣度／成交熱度代理，並非直接觀察。')
    from lab_jobs import workspace_busy
    busy=v22_alive(root) or workspace_busy(root)
    prepared=discover_prepared(root);dataset=None
    try:
        if (root/'studio'/'active_dataset.json').exists():_,dataset=load_dataset(root)
    except Exception as exc:st.warning(str(exc))
    with st.expander('① 沿用既有資料庫',expanded=dataset is None):
        st.write('V22直接沿用V21.6 bulk/cache資料層；不需要再下載約2,000檔多年歷史。')
        if prepared:
            selected=st.selectbox('已準備資料',prepared,format_func=lambda p:p['name']+' | '+str(p['start'])+' ~ '+str(p['end']))
            if st.button('採用這份資料（不下載）',disabled=busy):
                try:adopt_dataset(root,selected['run']);st.rerun()
                except Exception as exc:st.error(str(exc))
        if dataset:st.caption('Dataset ID: '+dataset['dataset_id']+'｜'+dataset['source_config']['start']+' ~ '+dataset['source_config']['end'])
    if not dataset:return
    editor=read_json(root/'v22'/'editor.json');profiles=editor.get('profiles',read_json(HERE/'v22_profiles.json'));settings=editor.get('settings',read_json(HERE/'v22_settings.json'))
    with st.expander('② 一次性母池校準（策略調整時不要重跑）',expanded=False):
        st.write('比較ADV200／ADV300／KOSTO250／KOSTO300。KOSTO母池先保留長期流動性，再降低對短期過熱成交量的追逐。只使用開發期，不碰最後保留期。')
        latest=read_json(root/'v22'/'latest_pool_study.json')
        if latest.get('suggested_pool'):st.success('上次開發期建議母池：'+str(latest['suggested_pool'])+'（不是全局最適證明）')
        if st.button('執行一次母池校準',disabled=busy):
            try:start_v22(root,profiles,settings,pool_study=True);st.success('已開始母池校準；完成後即可固定，不需每次重跑。')
            except Exception as exc:st.error(str(exc))
        if latest.get('run'):
            p=Path(latest['run'])
            if (p/'V22_POOL_REPORT.zip').is_file():st.download_button('下載母池校準完整報告', (p/'V22_POOL_REPORT.zip').read_bytes(),'V22_POOL_REPORT.zip','application/zip',on_click='ignore')
    st.subheader('③ 快速調整V22策略')
    st.write('預設只測4個明確假說；修改權重後直接使用因子／報酬快取。市場階段不是小權重因子，而是先決曝險閘門。')
    pool_options=['ADV200','ADV300','KOSTO250','KOSTO300']
    current=settings.get('pool_rule','KOSTO300')
    suggested=read_json(root/'v22'/'latest_pool_study.json').get('suggested_pool')
    if suggested in pool_options and current not in pool_options:current=suggested
    pool_choice=st.selectbox('固定母池規則（母池校準後固定，平常不要重選）',pool_options,index=pool_options.index(current) if current in pool_options else 3)
    settings=dict(settings);settings['pool_rule']=pool_choice
    with st.form('v22_editor'):
        profile_text=st.text_area('V22策略JSON',json.dumps(profiles,ensure_ascii=False,indent=2),height=430)
        setting_text=st.text_area('V22測試設定',json.dumps(settings,ensure_ascii=False,indent=2),height=260)
        fast=st.form_submit_button('▶ 快速驗證（不看最後保留期）',disabled=busy)
    if fast:
        try:start_v22(root,json.loads(profile_text),json.loads(setting_text),'fast');st.success('V22快速測試已啟動。')
        except Exception as exc:st.error(str(exc))
    with st.expander('④ 凍結後才做一次最終確認'):
        st.warning('同一保留期反覆重開會失去盲測意義。平常只用快速驗證；確定假說後才執行一次確認。')
        ok=st.checkbox('我已凍結本次母池、階段規則、因子與權重')
        if st.button('執行凍結確認',disabled=busy or not ok):
            try:start_v22(root,json.loads(profile_text),json.loads(setting_text),'confirm');st.success('V22確認測試已啟動。')
            except Exception as exc:st.error(str(exc))
    @st.fragment(run_every='3s')
    def monitor():
        err=read_json(root/'v22'/'launch_error.json')
        if err:st.error(err.get('type','')+': '+err.get('message',''))
        latest=read_json(root/'v22'/'latest.json')
        if not latest.get('run'):return
        run=Path(latest['run']);state=read_json(run/'state.json')
        if state:st.caption('狀態：'+str(state.get('status'))+'｜'+str(state.get('phase')))
        result=read_json(run/'results.json')
        if result:
            for h,m in result['models'].items():
                st.write(f"**{h}日｜{m['pool']}｜{m['choice']['name']}｜{m['verdict']}**")
                if m.get('stage_performance'):st.dataframe(m['stage_performance'],use_container_width=True)
            if (run/'REPORT.txt').is_file():
                with st.expander('直接閱讀測試報告'):st.text((run/'REPORT.txt').read_text(encoding='utf-8-sig'))
            if (run/'V22_TEST_REPORT.zip').is_file():st.download_button('下載V22完整測試報告ZIP',(run/'V22_TEST_REPORT.zip').read_bytes(),'V22_TEST_REPORT.zip','application/zip',on_click='ignore')
            for fn,mime in [('candidate_ranking.csv','text/csv'),('strategy_by_stage.csv','text/csv'),('monthly_decisions.csv','text/csv'),('factor_sensitivity.csv','text/csv'),('market_stage.csv','text/csv')]:
                if (run/fn).is_file():st.download_button('下載 '+fn,(run/fn).read_bytes(),fn,mime,on_click='ignore',key='v22'+fn)
    monitor()
