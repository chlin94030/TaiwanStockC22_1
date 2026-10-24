import numpy as np
import pandas as pd
from pathlib import Path
from lab_config import Config
from lab_yahoo_bulk import _ticker_frame,_raw_adj,YahooFastBulk


def _frame(tickers, periods=320):
    idx=pd.bdate_range('2024-01-01',periods=periods)
    cols=[];data={}
    for j,t in enumerate(tickers):
        base=np.linspace(50+j,70+j,periods)
        vals={'Open':base-.2,'High':base+.8,'Low':base-.8,'Close':base,
              'Adj Close':base*.98,'Volume':np.full(periods,1_000_000+j*1000.)}
        for k,v in vals.items():data[(t,k)]=v;cols.append((t,k))
    return pd.DataFrame(data,index=idx)


def test_config_accepts_fast_provider():
    c=Config(end='2026-10-20',history_provider='AUTO_FAST')
    c.validate()
    Config(end='2026-10-20',history_provider='YAHOO_FAST').validate()


def test_ticker_frame_both_multiindex_orientations():
    d=_frame(['1101.TW','1102.TW'],10)
    a=_ticker_frame(d,'1101.TW')
    assert 'Close' in a and len(a)==10
    e=d.copy();e.columns=pd.MultiIndex.from_tuples([(b,a) for a,b in e.columns])
    b=_ticker_frame(e,'1102.TW')
    assert 'Close' in b and len(b)==10


def test_proxy_is_explicit_typical_price_times_volume():
    f=_ticker_frame(_frame(['1101.TW'],5),'1101.TW')
    raw,adj=_raw_adj(f,'1101.TW')
    expect=((f['High']+f['Low']+f['Close'])/3*f['Volume']).iloc[0]
    assert abs(raw.amount.iloc[0]-expect)<1e-6
    assert abs(adj.close.iloc[0]-f['Adj Close'].iloc[0])<1e-8
    assert raw.attrs.get('quality_issues')==[]


def test_turnover_uses_batch_cache_without_finmind(tmp_path):
    cfg=Config(start='2024-01-01',end='2025-03-31',history_provider='YAHOO_FAST',
               min_train_months=12,holdout_months=6,min_outer_folds=1).resolved()
    tickers=['1101.TW','1102.TW','1103.TW']
    cat=[{'ticker':t} for t in tickers]
    calls=[]
    b=YahooFastBulk(cfg,lambda **kw:None,lambda:None,chunk_size=2)
    b._download=lambda ts:(calls.append(tuple(ts)) or _frame(list(ts),320))
    signals=pd.DatetimeIndex(pd.bdate_range('2025-02-03',periods=2))
    out,meta=b.turnover(signals,cat,tmp_path)
    assert not out.empty
    assert meta['estimated_amount'] is True and meta['actual_amount'] is False
    first=len(calls)
    calls.clear()
    out2,meta2=b.turnover(signals,cat,tmp_path)
    assert len(calls)==0
    assert len(out2)==len(out)
    assert first==2


class DummyStore:
    def __init__(self):self.rows=[]
    def put(self,table,ticker,frame,start,end,source):self.rows.append((table,ticker,source,len(frame)))


def test_selected_prices_saved_as_proxy_and_adjusted():
    cfg=Config(start='2024-01-01',end='2025-03-31',history_provider='YAHOO_FAST',
               min_train_months=12,holdout_months=6,min_outer_folds=1).resolved()
    b=YahooFastBulk(cfg,lambda **kw:None,lambda:None,chunk_size=10)
    b._download=lambda ts:_frame(list(ts),320)
    s=DummyStore();info=b.populate_selected_prices(s,['1101.TW','0050.TW'])
    assert info['estimated_amount'] is True
    assert {x[0] for x in s.rows}=={'raw','adj'}
    assert any('PROXY_AMOUNT' in x[2] for x in s.rows if x[0]=='raw')
from lab_official_bulk import parse_twse,parse_tpex,OfficialDailyBulk


def test_parse_twse_official_whole_market():
    payload={'stat':'OK','tables':[{'title':'每日收盤行情','fields':['證券代號','證券名稱','成交股數','成交筆數','成交金額','開盤價','最高價','最低價','收盤價'],
              'data':[['1101','台泥','1,000','10','50,000','49','51','48','50'],['0050','元大台灣50','2,000','20','400,000','199','202','198','200']]}]}
    f=parse_twse(payload,'2026-10-21')
    assert set(f.ticker)=={'1101.TW','0050.TW'}
    assert float(f.loc[f.ticker=='1101.TW','amount'].iloc[0])==50000


def test_parse_tpex_official_legacy():
    payload={'aaData':[['6488','環球晶','500','+5','495','510','490','1,000','500,000','100']]}
    f=parse_tpex(payload,'2026-10-21')
    assert f.ticker.iloc[0]=='6488.TWO'
    assert f.amount.iloc[0]==500000


def test_official_cache_resume_without_network(tmp_path):
    cfg=Config(start='2025-01-02',end='2025-01-03',history_provider='OFFICIAL_BULK',
               min_train_months=12,holdout_months=6,min_outer_folds=1).resolved()
    calls=[]
    b=OfficialDailyBulk(cfg,lambda **kw:None,lambda:None,interval=0).bind_cache(tmp_path)
    def tw(d):
        calls.append(('tw',str(pd.Timestamp(d).date())))
        return pd.DataFrame([{'date':pd.Timestamp(d),'ticker':'1101.TW','board':'TWSE','open':50,'high':51,'low':49,'close':50,'volume':1000,'amount':50000}])
    def otc(d):
        calls.append(('otc',str(pd.Timestamp(d).date())))
        return pd.DataFrame([{'date':pd.Timestamp(d),'ticker':'6488.TWO','board':'TPEX','open':500,'high':510,'low':490,'close':500,'volume':1000,'amount':500000}])
    b._fetch_twse=tw;b._fetch_tpex=otc
    cal=b.calendar();assert len(cal)==2 and len(calls)==4
    calls.clear();cal2=b.calendar();assert cal2==cal and len(calls)==0
