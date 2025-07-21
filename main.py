#!/usr/bin/env python3
"""
Korean Stock Monitor – Golden / Dead Cross on Composite Lines
────────────────────────────────────────────────────────────────────
📌 **매수·매도 규칙**
- **Composite K** = MACD(12,26) + Slow %K(14,3)
- **Composite D** = MACD(12,26) + Slow %D(14,3)
- **Golden Cross** (Composite K ↑ Composite D) → **BUY**
- **Dead Cross**   (Composite K ↓ Composite D) → **SELL**

매 실행 시 마지막 두 일자의 교차여부를 판정해 신호가 발생하면 Telegram으로 **텍스트 + 차트 이미지**를 전송합니다.

환경 변수
-----------
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (필수)
- `STOCK_LIST="005930.KS,000660.KS"` 모니터링 대상 (콤마 구분)
- `SCALE_MACD=true` → MACD 0‑100 정규화 후 합산
- `SAVE_CSV=true`   → CSV 저장
- `FONT_PATH` (한글 폰트 TTF 경로)

requirements.txt (추가 패키지 포함)
-----------------------------------
```
pandas>=1.5.3
requests>=2.28.2
finance-datareader>=0.9.59
ta>=0.10.2
matplotlib>=3.8.4
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
    font_prop = None

# ───── 환경 변수 ───── #
TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID")
STOCK_LIST = os.getenv("STOCK_LIST", "").split(",")
STOCKS     = [s.strip().upper() for s in STOCK_LIST if s.strip()]
SCALE_MACD = os.getenv("SCALE_MACD", "false").lower() == "true"
SAVE_CSV   = os.getenv("SAVE_CSV",   "false").lower() == "true"

if not (TOKEN and CHAT_ID and STOCKS):
    sys.exit("필수 환경변수 누락: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, STOCK_LIST")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───── 이름 매핑 (KRX, KOSDAQ) ───── #
krx = fdr.StockListing('KRX')[['Code','Name']]
kosdaq = fdr.StockListing('KOSDAQ')[['Code','Name']]
name_map = {f"{row.Code}.KS": row.Name for _, row in krx.iterrows()}
name_map.update({f"{row.Code}.KQ": row.Name for _, row in kosdaq.iterrows()})
def get_name(code: str) -> str:
    return name_map.get(code, code)

# ───── 지표 계산 헬퍼 ───── #

def latest(s: pd.Series, n: int = 1) -> Optional[float]:
    if len(s) < n or s.isna().all(): return None
    return float(s.iloc[-n])


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    macd  = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    stoch = StochasticOscillator(df['Close'], df['High'], df['Low'], window=14, smooth_window=3)

    df['MACD']     = macd.macd()
    df['MACD_SIG'] = macd.macd_signal()
    df['SlowK']    = stoch.stoch()
    df['SlowD']    = stoch.stoch_signal()

    macd_vals = df['MACD']
    if SCALE_MACD:
        min_m, max_m = macd_vals.min(), macd_vals.max()
        macd_vals = (macd_vals - min_m) / (max_m - min_m) * 100

    df['CompK'] = macd_vals + df['SlowK']
    df['CompD'] = macd_vals + df['SlowD']
    df['Diff']  = df['CompK'] - df['CompD']
    return df

# ───── 시그널 판정 ───── #

def detect_cross(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 2: return None
    prev, curr = df['Diff'].iloc[-2], df['Diff'].iloc[-1]
    if prev <= 0 < curr: return 'BUY'
    if prev >= 0 > curr: return 'SELL'
    return None

# ───── 데이터 조회 ───── #

def fetch_daily(code: str, days: int = 120) -> Optional[pd.DataFrame]:
    """FinanceDataReader로 한국 주식 과거 데이터 조회, 'Date' 포함 반환"""
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    # 종목코드에서 시장 접미사 제거 (e.g., '005930.KS' → '005930')
    symbol = code.split('.')[0]
    try:
        df = fdr.DataReader(symbol, start, end)
        if df.empty:
            logging.warning(f"{code}: 데이터 없음")
            return None
        df = df.reset_index()
        # DataFrame 컬럼 이름 통일
        df.rename(columns={
            'Date':'Date', 'Open':'Open', 'High':'High',
            'Low':'Low', 'Close':'Close', 'Volume':'Volume'
        }, inplace=True)
        return df[['Date','Open','High','Low','Close','Volume']]
    except Exception as e:
        logging.warning(f"{code}: 데이터 조회 실패 - {e}")
        return None

# ───── 차트 생성 ───── # ───── #

def make_chart(df: pd.DataFrame, code: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6), sharex=True, gridspec_kw={'height_ratios':[3,1]})

    name = get_name(code)
    ax1.plot(df['Date'], df['Close'], label='종가')
    ax1.plot(df['Date'], df['Close'].rolling(20).mean(), linestyle='--', label='MA20')
    ax1.set_title(f"{code} ({name})", fontproperties=font_prop)
    ax1.legend(prop=font_prop)

    ax2.plot(df['Date'], df['CompK'], label='CompK')
    ax2.plot(df['Date'], df['CompD'], label='CompD')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_title('Composite Cross', fontproperties=font_prop)
    ax2.legend(prop=font_prop)

    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(); fig.tight_layout()

    path = f"{code}_chart.png"
    fig.savefig(path, dpi=100); plt.close(fig)
    return path

# ───── Telegram 전송 ───── #

def tg_text(msg: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for chunk in [msg[i:i+3000] for i in range(0, len(msg), 3000)]:
        requests.post(url, json={'chat_id':CHAT_ID,'text':chunk})


def tg_photo(path: str, caption: str=''):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(path,'rb') as f:
        requests.post(url, data={'chat_id':CHAT_ID,'caption':caption}, files={'photo':f})

# ───── 메인 ───── #

def main():
    alerts=[]
    for code in STOCKS:
        df = fetch_daily(code)
        if df is None or len(df)<40:
            continue
        df = add_composites(df)
        sig = detect_cross(df)
        cap = f"{code} {get_name(code)}"
        if sig:
            cap = f"{sig} Signal - {cap}"
            alerts.append(cap)
        img = make_chart(df.tail(120), code)
        tg_photo(img, caption=cap)
        if SAVE_CSV:
            df.to_csv(f"{code}_hist.csv", index=False)
    if alerts:
        tg_text("\n".join(alerts))
    else:
        tg_text('신호 없음')

if __name__=='__main__':
    main()
