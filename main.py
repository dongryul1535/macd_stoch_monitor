#!/usr/bin/env python3
"""
Stock Monitor – Golden / Dead Cross with NH MTS‑style MACD+Stochastic
────────────────────────────────────────────────────────────────────
* Composite K  = ( StochNormalize(MACD_raw) + Slow %K ) / 2
* Composite D  = SMA(Composite K, 3)
* Golden Cross = CompK ↑ CompD  → BUY
* Dead   Cross = CompK ↓ CompD  → SELL

각 종목에 대한 차트 이미지를 생성해 텔레그램으로 알림을 보냅니다.
"""

import os
import logging
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib import font_manager
import FinanceDataReader as fdr

# ──────────────────────────── 환경 설정 ────────────────────────────
TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STOCKS  = [s.strip() for s in os.getenv("STOCK_LIST", "005930.KS").split(",") if s.strip()]
SAVE_CSV = os.getenv("SAVE_CSV", "false").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 한글 폰트 (서버 환경에 맞게 조정)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_prop = font_manager.FontProperties(fname=font_path) if os.path.exists(font_path) else None

# ────────────────────────────── 지표 계산 ───────────────────────────

def add_composites(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    k_window: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
    use_ema: bool = True,
    clip: bool = True,
) -> pd.DataFrame:
    """NH 나무 MTS ‘MACD+Stochastic’ 유사 Composite K/D 추가"""

    close, high, low = df['Close'], df['High'], df['Low']

    # 1) MACD 원시값
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_raw = ema_fast - ema_slow

    # 2) MACD → 0~100 스토캐스틱화
    macd_min = macd_raw.rolling(k_window, min_periods=1).min()
    macd_max = macd_raw.rolling(k_window, min_periods=1).max()
    macd_norm = (macd_raw - macd_min) / (macd_max - macd_min).replace(0, np.nan) * 100
    macd_norm = macd_norm.fillna(50)

    if k_smooth > 1:
        macd_norm = (
            macd_norm.ewm(span=k_smooth, adjust=False).mean() if use_ema else
            macd_norm.rolling(k_smooth, min_periods=1).mean()
        )

    # 3) 가격 기반 Slow %K
    ll = low.rolling(k_window, min_periods=1).min()
    hh = high.rolling(k_window, min_periods=1).max()
    k_raw = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    k_raw = k_raw.fillna(50)

    slow_k = (
        k_raw.ewm(span=k_smooth, adjust=False).mean() if (k_smooth > 1 and use_ema) else
        k_raw.rolling(k_smooth, min_periods=1).mean() if k_smooth > 1 else k_raw
    )

    # 4) Composite K / D
    comp_k = (macd_norm + slow_k) / 2.0
    comp_d = comp_k.rolling(d_smooth, min_periods=1).mean() if d_smooth > 1 else comp_k

    if clip:
        comp_k = comp_k.clip(0, 100)
        comp_d = comp_d.clip(0, 100)

    df['CompK'] = comp_k
    df['CompD'] = comp_d
    df['Diff']  = comp_k - comp_d
    return df


# ──────────────────────────── 신호 판정 ────────────────────────────

def detect_cross(df: pd.DataFrame, ob: int = 80, os: int = 20) -> Optional[str]:
    if len(df) < 2:
        return None
    prev_diff, curr_diff = df['Diff'].iloc[-2], df['Diff'].iloc[-1]
    prev_k = df['CompK'].iloc[-2]

    # 골든 / 데드 크로스 + 과매수·과매도 필터
    if prev_diff <= 0 < curr_diff:
        return 'BUY' if prev_k < os else 'BUY_W'
    if prev_diff >= 0 > curr_diff:
        return 'SELL' if prev_k > ob else 'SELL_W'
    return None


# ──────────────────────────── 차트 그리기 ───────────────────────────

def make_chart(df: pd.DataFrame, code: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # 가격
    ax1.plot(df['Date'], df['Close'], label='Close')
    ax1.plot(df['Date'], df['Close'].rolling(20).mean(), '--', label='MA20')
    ax1.set_title(code, fontproperties=font_prop)
    ax1.legend(prop=font_prop)

    # Composite K/D
    ax2.plot(df['Date'], df['CompK'], color='red', label='MACD+Slow%K')
    ax2.plot(df['Date'], df['CompD'], color='purple', label='MACD+Slow%D')
    ax2.axhline(20, color='gray', linestyle='--', linewidth=0.5)
    ax2.axhline(80, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_title('MACD+Stochastic (NH Style)', fontproperties=font_prop)
    ax2.legend(prop=font_prop, loc='upper left')
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate()
    fig.tight_layout()

    path = f"{code}_chart.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ───────────────────────────── 텔레그램 ────────────────────────────

def send_telegram(message: str, photo_path: Optional[str] = None) -> None:
    """문자·이미지 전송 (사진 있으면 sendPhoto, 없으면 sendMessage)"""
    if not TOKEN or not CHAT_ID:
        logging.error("Telegram TOKEN / CHAT_ID 환경변수가 설정되지 않았습니다.")
        return
    try:
        if photo_path and os.path.exists(photo_path):
            with open(photo_path, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                    data={"chat_id": CHAT_ID, "caption": message},
                    files={"photo": f},
                    timeout=10,
                )
        else:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": CHAT_ID, "text": message},
                timeout=10,
            )
        logging.info("Telegram 전송 완료: %s", message)
    except Exception as e:
        logging.exception("Telegram 전송 실패: %s", e)


# ──────────────────────────── 데이터 수집 ───────────────────────────

def fetch_price_data(code: str, start: str) -> pd.DataFrame:
    """FinanceDataReader 사용 – 일봉 데이터 수집"""
    df = fdr.DataReader(code, start)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df


# ───────────────────────────────── Main ────────────────────────────

def main() -> None:
    start_date = (dt.date.today() - dt.timedelta(days=365)).isoformat()

    for code in STOCKS:
        logging.info("%s: 데이터 수집", code)
        df = fetch_price_data(code, start_date)
        if df.empty:
