"""Fast no-key historical bootstrap using Yahoo Finance multi-ticker downloads.

This is a DEVELOPMENT transport.  It deliberately labels turnover as a proxy:
    typical_price * volume = ((high + low + close) / 3) * volume
It is never presented as exchange-reported Trading_money.  The purpose is to
avoid ~2,000 serial FinMind requests while iterating on V22 strategies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import time
import numpy as np
import pandas as pd

from lab_io import DataError, frame_write, read_json, atomic_json, sha
from lab_bulk import vectorized_turnover

BENCHMARK = '0050.TW'


def _clean_index(index):
    idx=pd.DatetimeIndex(index)
    if idx.tz is not None:
        try: idx=idx.tz_convert('Asia/Taipei').tz_localize(None)
        except Exception: idx=idx.tz_localize(None)
    return idx.normalize()


def _ticker_frame(downloaded,ticker):
    """Extract one ticker from either yfinance MultiIndex orientation."""
    if downloaded is None or getattr(downloaded,'empty',True):
        return pd.DataFrame()
    d=pd.DataFrame(downloaded)
    if isinstance(d.columns,pd.MultiIndex):
        l0=[str(x) for x in d.columns.get_level_values(0)]
        l1=[str(x) for x in d.columns.get_level_values(1)]
        if ticker in l0:
            try: out=d.xs(ticker,axis=1,level=0,drop_level=True)
            except Exception:return pd.DataFrame()
        elif ticker in l1:
            try: out=d.xs(ticker,axis=1,level=1,drop_level=True)
            except Exception:return pd.DataFrame()
        else:return pd.DataFrame()
    else:
        out=d.copy()
    out.index=_clean_index(out.index)
    return out[~out.index.duplicated(keep='last')].sort_index()


def _raw_adj(frame,ticker):
    if frame is None or frame.empty:return None,None
    aliases={str(c).lower().replace('_',' ').strip():c for c in frame.columns}
    def col(*names):
        for n in names:
            key=n.lower().replace('_',' ').strip()
            if key in aliases:return pd.to_numeric(frame[aliases[key]],errors='coerce')
        return pd.Series(np.nan,index=frame.index,dtype=float)
    op,hi,lo,cl,vol=col('Open'),col('High'),col('Low'),col('Close'),col('Volume')
    ac=col('Adj Close','AdjClose')
    typical=(hi+lo+cl)/3.0
    amount=(typical*vol).where((typical>0)&(vol>=0))
    raw=pd.DataFrame({'date':frame.index,'open':op.to_numpy(),'high':hi.to_numpy(),'low':lo.to_numpy(),'close':cl.to_numpy(),'volume':vol.to_numpy(),'amount':amount.to_numpy()}).reset_index(drop=True)
    positive=raw[['open','high','low','close']].gt(0).all(axis=1)
    high_bad=positive & (raw.high < raw[['open','close','low']].max(axis=1)-1e-6)
    low_bad=positive & (raw.low > raw[['open','close','high']].min(axis=1)+1e-6)
    nonfinite=~np.isfinite(raw[['open','high','low','close']]).all(axis=1)
    bad=high_bad|low_bad|nonfinite|(~positive)
    issues=[]
    for ix,row in raw.loc[bad].iterrows():
        issues.append({'ticker':ticker,'date':pd.Timestamp(row.date).strftime('%Y-%m-%d'),
            'kind':'YAHOO_OHLC_RANGE_MISMATCH' if bool(high_bad.loc[ix] or low_bad.loc[ix]) else 'YAHOO_NO_VALID_OHLC',
            'observed':{k:(float(row[k]) if pd.notna(row[k]) and np.isfinite(row[k]) else None)
                        for k in ('open','high','low','close','volume','amount')}})
    if bad.any():raw.loc[bad,['open','high','low','close']]=np.nan
    raw.attrs['quality_issues']=issues
    ratio=(ac/cl.where(cl>0)).where(np.isfinite(ac)&np.isfinite(cl)&(ac>0)&(cl>0))
    adj=pd.DataFrame({'date':frame.index})
    for name,series in [('open',op),('high',hi),('low',lo),('close',cl)]:adj[name]=(series*ratio).to_numpy()
    adj=adj.reset_index(drop=True)
    return raw,adj


def _chunk_key(tickers,start,end):
    payload=('|'.join(tickers)+'|'+str(start)+'|'+str(end)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class YahooFastBulk:
    cfg: object
    progress: object
    check_pause: object
    chunk_size: int = 80

    provider_name='YAHOO_FAST'
    actual_amount=False

    def _yf(self):
        try:import yfinance as yf
        except Exception as exc:raise DataError('Yahoo fast mode requires yfinance') from exc
        return yf

    def _download(self,tickers):
        yf=self._yf();tickers=list(tickers)
        end=(pd.Timestamp(self.cfg.end)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        last=None
        for attempt in range(3):
            self.check_pause()
            try:
                d=yf.download(tickers=tickers,start=self.cfg.start,end=end,auto_adjust=False,
                    actions=False,progress=False,threads=True,group_by='ticker',timeout=self.cfg.request_timeout)
                if d is not None and not d.empty:return d
                last=DataError('Yahoo returned empty batch')
            except Exception as exc:last=exc
            time.sleep(2*(attempt+1))
        raise DataError('Yahoo batch download failed: '+type(last).__name__)

    def calendar(self):
        d=self._download([BENCHMARK]);f=_ticker_frame(d,BENCHMARK)
        if f.empty:raise DataError('Yahoo benchmark calendar is empty')
        idx=_clean_index(f.index)
        idx=idx[(idx>=pd.Timestamp(self.cfg.start))&(idx<=pd.Timestamp(self.cfg.end))]
        if len(idx)<300:raise DataError('Yahoo benchmark calendar incomplete')
        return [x.strftime('%Y-%m-%d') for x in idx]

    def turnover(self,signals,cat,cache_dir):
        root=Path(cache_dir)/'yahoo_fast';root.mkdir(parents=True,exist_ok=True)
        tickers=sorted({r['ticker'] for r in cat})
        manifest={'provider':'YAHOO_FAST','start':self.cfg.start,'end':self.cfg.end,'roster_count':len(tickers),
                  'actual_amount':False,'estimated_amount':True,
                  'proxy_formula':'((High+Low+Close)/3)*Volume','development_only':True}
        wide_parts=[];available=set();failed=[]
        chunks=[tickers[i:i+self.chunk_size] for i in range(0,len(tickers),self.chunk_size)]
        for ci,chunk in enumerate(chunks,1):
            self.check_pause();key=_chunk_key(chunk,self.cfg.start,self.cfg.end);path=root/f'turnover_{ci:03d}_{key}.csv.gz'
            if path.exists():
                w=pd.read_csv(path,index_col=0,parse_dates=True);w.index=pd.to_datetime(w.index)
                self.progress(phase='YAHOO_FAST',done=ci,total=len(chunks),message=f'Yahoo批次快取 {ci}/{len(chunks)}')
            else:
                self.progress(phase='YAHOO_FAST',done=ci-1,total=len(chunks),message=f'Yahoo批次下載 {ci}/{len(chunks)}（每批最多{self.chunk_size}檔）')
                d=self._download(chunk);series={}
                for t in chunk:
                    f=_ticker_frame(d,t)
                    if f.empty:failed.append(t);continue
                    raw,_=_raw_adj(f,t)
                    if raw is None or raw.empty:failed.append(t);continue
                    s=raw.set_index('date')['amount'];s.name=t;series[t]=s;available.add(t)
                if not series:
                    raise DataError('Yahoo batch returned no usable symbols')
                w=pd.concat(series.values(),axis=1).sort_index();w.to_csv(path,index_label='date',compression={'method':'gzip','compresslevel':1})
                self.progress(phase='YAHOO_FAST',done=ci,total=len(chunks),message=f'Yahoo批次已儲存 {ci}/{len(chunks)}')
            available.update(str(c) for c in w.columns);wide_parts.append(w)
        amount=pd.concat(wide_parts,axis=1).sort_index();amount=amount.loc[:,~amount.columns.duplicated()]
        # Require broad recent coverage, but do not require old history for recent IPOs.
        recent=amount.tail(min(60,len(amount))).notna().any(axis=0)
        recent_fraction=float(recent.sum()/max(1,len(tickers)))
        if recent_fraction<0.90:
            raise DataError(f'Yahoo fast recent symbol coverage only {recent_fraction:.1%}; refusing a biased universe')
        codes={str(r['ticker']).split('.')[0]:r['ticker'] for r in cat}
        amount.columns=[str(c).split('.')[0] for c in amount.columns]
        out=vectorized_turnover(amount,signals,self.cfg,codes)
        manifest.update({'rows':len(out),'recent_symbol_coverage':recent_fraction,'failed_symbols':sorted(set(failed))[:200]})
        meta=root/f'turnover_{self.cfg.start}_{self.cfg.end}.json';atomic_json(meta,manifest);manifest['checkpoint_sha']=sha(meta)
        return out,manifest

    def populate_adjusted_only(self,store,tickers):
        tickers=sorted(set(tickers));chunks=[tickers[i:i+self.chunk_size] for i in range(0,len(tickers),self.chunk_size)]
        saved=0;missing=[]
        for ci,chunk in enumerate(chunks,1):
            self.check_pause();self.progress(phase='YAHOO_ADJUSTED',done=ci-1,total=len(chunks),message=f'Yahoo還原價批次 {ci}/{len(chunks)}')
            d=self._download(chunk)
            for t in chunk:
                f=_ticker_frame(d,t);_,adj=_raw_adj(f,t)
                if adj is None or adj.empty:missing.append(t);continue
                store.put('adj',t,adj,self.cfg.start,self.cfg.end,'YAHOO_ADJ_CLOSE_RATIO_OHLC');saved+=1
            self.progress(phase='YAHOO_ADJUSTED',done=ci,total=len(chunks),message=f'Yahoo還原價已儲存 {saved}/{len(tickers)}')
        if missing:raise DataError('Yahoo adjusted missing symbols: '+','.join(missing[:30]))
        return {'provider':'YAHOO_ADJUSTED_BATCH','symbols':saved}

    def populate_selected_prices(self,store,tickers):
        tickers=sorted(set(tickers));chunks=[tickers[i:i+self.chunk_size] for i in range(0,len(tickers),self.chunk_size)]
        saved=0;missing=[]
        for ci,chunk in enumerate(chunks,1):
            self.check_pause();self.progress(phase='YAHOO_SELECTED',done=ci-1,total=len(chunks),message=f'Yahoo固定母池批次 {ci}/{len(chunks)}')
            d=self._download(chunk)
            for t in chunk:
                f=_ticker_frame(d,t)
                raw,adj=_raw_adj(f,t)
                if raw is None or raw.empty or adj is None or adj.empty:
                    missing.append(t);continue
                store.put('raw',t,raw,self.cfg.start,self.cfg.end,'YAHOO_FAST_PROXY_AMOUNT_RAW_OHLC')
                store.put('adj',t,adj,self.cfg.start,self.cfg.end,'YAHOO_ADJ_CLOSE_RATIO_OHLC')
                saved+=1
            self.progress(phase='YAHOO_SELECTED',done=ci,total=len(chunks),message=f'Yahoo固定母池已儲存 {saved}/{len(tickers)}')
        if missing:
            raise DataError('Yahoo selected-union missing symbols: '+','.join(missing[:30]))
        return {'provider':'YAHOO_FAST','symbols':saved,'actual_amount':False,'estimated_amount':True,
                'proxy_formula':'((High+Low+Close)/3)*Volume','development_only':True}
