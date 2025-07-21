#!/usr/bin/env python3
"""
Korean Stock Monitor – Golden / Dead Cross on Composite Lines + 전분기 EPS•PER
────────────────────────────────────────────────────────────────────
📌 **매수·매도 규칙**
- **Composite K** = MACD(12,26) + Slow %K(14,3)
- **Composite D** = MACD(12,26) + Slow %D(14,3)
- **Golden Cross** (Composite K ↑ Composite D) → **BUY**
- **Dead Cross**   (Composite K ↓ Composite D) → **SELL**

매 실행 시 마지막 두 일자의 교차여부를 판정하고, 전분기 EPS 및 현재 PER을 계산하여 Telegram으로 **텍스트 + 차트 이미지** 전송.

환경 변수
-----------
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (필수)
- `DART_API_KEY` (OpenDartReader API Key)
- `STOCK_LIST="005930.KS,000660.KS"` 모니터링 대상
- `SCALE_MACD=true` → MACD 0‑100 정규화
- `SAVE_CSV=true`   → CSV 저장
- `FONT_PATH` (한글 폰트 TTF 경로)

requirements.txt
-----------------------------------
```
pandas>=1.5.3
requests>=2.28.2
finance-datareader>=0.9.59
ta>=0.10.2
matplotlib>=3.8.4
opendartreader>=0.2.6
```"""

import os
import sys
import logging
import datetime as dt
from typing import List, Optional, Tuple

import pandas as pd
import requests
import FinanceDataReader as fdr
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from opendartreader import OpenDartReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as fm

# ───── 한글 폰트 설정 ───── #
FONT_PATH = os.getenv("FONT_PATH", "")
if FONT_PATH and os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None  # 한글 깨짐시 FONT_PATH 설정 필요

# ───── 환경 변수 ───── #
TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID")
DART_KEY   = os.getenv("DART_API_KEY")
STOCK_LIST = os.getenv("STOCK_LIST", "").split(",")
STOCKS     = [s.strip().upper() for s in STOCK_LIST if s.strip()]
SCALE_MACD = os.getenv("SCALE_MACD", "false").lower() == "true"
SAVE_CSV   = os.getenv("SAVE_CSV",   "false").lower() == "true"

if not (TOKEN and CHAT_ID and STOCKS and DART_KEY):
    sys.exit("필수 환경변수 누락: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, STOCK_LIST, DART_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───── OpenDartReader 초기화 ───── #
dart = OpenDartReader(DART_KEY)

# ───── 이름 매핑 (KRX, KOSDAQ) ───── #
krx = fdr.StockListing('KRX')[['Code','Name']]
kosdaq = fdr.StockListing('KOSDAQ')[['Code','Name']]
name_map = {f"{r.Code}.KS": r.Name for _,r in krx.iterrows()}
name_map.update({f"{r.Code}.KQ": r.Name for _,r in kosdaq.iterrows()})
def get_name(code: str) -> str:
    return name_map.get(code, code)

# ───── 전분기 EPS 조회 ───── #
def get_last_quarter_eps(corp_code: str) -> Optional[float]:
    now = dt.datetime.now()
    q = (now.month - 1) // 3
    if q == 0:
        year = now.year - 1; rep = '11014'
    else:
        year = now.year; rep = f'1101{q}'
    df_fs = dart.fnltt_singl_acnt_all(corp_code, year, rep)
    row = df_fs[df_fs['account_nm'].str.contains('주당순이익')]
    if row.empty: return None
    return float(row.iloc[0]['thstrm_amount'])

# ───── 지표 계산 헬퍼 ───── #

def latest(s: pd.Series, n: int = 1) -> Optional[float]:
    if len(s) < n or s.isna().all(): return None
    return float(s.iloc[-n])


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    st   = StochasticOscillator(df['Close'], df['High'], df['Low'], window=14, smooth_window=3)
    df['MACD'] = macd.macd(); df['MACD_SIG'] = macd.macd_signal()
    df['SlowK'] = st.stoch(); df['SlowD'] = st.stoch_signal()
    mv = df['MACD']
    if SCALE_MACD:
        mv = (mv - mv.min())/(mv.max()-mv.min())*100
    df['CompK'] = mv + df['SlowK']; df['CompD'] = mv + df['SlowD']
    df['Diff']  = df['CompK'] - df['CompD']
    return df

# ───── 교차 판정 ───── #

def detect_cross(df: pd.DataFrame) -> Optional[str]:
    p,c = df['Diff'].iloc[-2], df['Diff'].iloc[-1]
    if p <= 0 < c: return 'BUY'
    if p >= 0 > c: return 'SELL'
    return None

# ───── 데이터 조회 ───── #

def fetch_daily(code: str, days: int = 120) -> Optional[pd.DataFrame]:
    end,start = dt.datetime.now(), dt.datetime.now()-dt.timedelta(days=days)
    # 원본 코드(시장 접미사 포함)로 조회
    try:
        df = fdr.DataReader(code, start, end)
        df = df.reset_index(); df.rename(columns={'Date':'Date','Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'}, inplace=True)
        return df[['Date','Open','High','Low','Close','Volume']]
    except:
        return None

# ───── 차트 생성 ───── #

def make_chart(df: pd.DataFrame, code: str) -> str:
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,6),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    name = get_name(code)
    ax1.plot(df['Date'],df['Close'],label='종가'); ax1.plot(df['Date'],df['Close'].rolling(20).mean(),linestyle='--',label='MA20')
    ax1.set_title(f"{code} ({name})",fontproperties=font_prop); ax1.legend(prop=font_prop)
    ax2.plot(df['Date'],df['CompK'],label='CompK'); ax2.plot(df['Date'],df['CompD'],label='CompD')
    ax2.axhline(0,linewidth=0.5,color='gray'); ax2.set_title('Composite Cross',fontproperties=font_prop); ax2.legend(prop=font_prop)
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d')); fig.autofmt_xdate(); fig.tight_layout()
    path=f"{code}_chart.png"; fig.savefig(path,dpi=100); plt.close(fig)
    return path

# ───── Telegram 전송 ───── #

def tg_text(msg:str):
    u=f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for c in [msg[i:i+3000]for i in range(0,len(msg),3000)]: requests.post(u,json={'chat_id':CHAT_ID,'text':c})

def tg_photo(path:str,cap:str=''):
    u=f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(path,'rb')as f: requests.post(u,data={'chat_id':CHAT_ID,'caption':cap},files={'photo':f})

# ───── 메인 ───── #

def main():
    alerts=[]
    for code in STOCKS:
        df=fetch_daily(code)
        if df is None or len(df)<40: continue
        df=add_composites(df)
        sig=detect_cross(df)
        sym=code.split('.')[0]
        eps=get_last_quarter_eps(sym) or 0.0
        price=latest(df['Close']) or 0.0
        per=price/eps if eps else None
        name=get_name(code)
        cap=f"{code} ({name}) | EPS:{eps:.2f} | PER:{per:.2f}" if per else f"{code} ({name}) | EPS:NA"
        if sig:
            cap=f"{sig} Signal - {cap}"; alerts.append(cap)
        img=make_chart(df.tail(120),code)
        tg_photo(img,caption=cap)
        if SAVE_CSV: df.to_csv(f"{code}_hist.csv",index=False)
    tg_text("\n".join(alerts) if alerts else '신호 없음')

if __name__=='__main__': main()
