#!/usr/bin/env python3
"""
Korean Stock Monitor – Golden / Dead Cross on Composite Lines + 전분기 EPS•PER (DART API)
────────────────────────────────────────────────────────────────────
📌 **매수·매도 규칙 (NH MTS MACD+Stochastic 스타일)**
- **Composite K** = ( StochNormalize(MACD_raw) + Slow %K ) / 2
- **Composite D** = SMA(Composite K, 3)
- **Golden Cross** (Composite K ↑ Composite D) → **BUY**
- **Dead Cross**   (Composite K ↓ Composite D) → **SELL**

매 실행 시 마지막 두 일자의 교차 여부를 판정하고, 전분기 EPS 및 현재 PER을 계산하여 Telegram으로 **텍스트 + 차트 이미지** 전송.

환경 변수
-----------
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (필수)
- `DART_API_KEY` (전자공시 OpenDart 인증키)
- `STOCK_LIST="005930.KS, ..."`  모니터링 대상 (없으면 기본 리스트)
- `SAVE_CSV=true` 저장 여부
"""

import os
import sys
import logging
import datetime as dt
from typing import List, Optional
import io
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib import font_manager
from ta import add_all_ta_features  # 여전히 사용 가능 (필요 시)

# 한글 폰트 지정 (환경에 맞춰 수정)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
else:
    font_prop = None

# ───── 설정값 ───── #
TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DART_KEY = os.getenv("DART_API_KEY")
STOCKS  = os.getenv("STOCK_LIST", "005930.KS").split(",")
SAVE_CSV = os.getenv("SAVE_CSV", "false").lower() == "true"

# ───── 유틸 ───── #

def latest(s: pd.Series, n: int = 1) -> Optional[float]:
    val = s.iloc[-n] if len(s) >= n else None
    return None if pd.isna(val) else float(val)

# ───── NH 스타일 Composite 지표 ───── #

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
    """NH 나무 MTS 'MACD+Stochastic' 유사 복제 지표를 df에 추가"""

    close, high, low = df['Close'], df['High'], df['Low']

    # 1) MACD (fast/slow EMA)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_raw = ema_fast - ema_slow

    # 2) MACD를 Stochastic 방식으로 0~100 정규화
    macd_min = macd_raw.rolling(k_window, min_periods=1).min()
    macd_max = macd_raw.rolling(k_window, min_periods=1).max()
    macd_norm = (macd_raw - macd_min) / (macd_max - macd_min).replace(0, np.nan) * 100
    macd_norm = macd_norm.fillna(50)

    # 3) smoothing (Slow%K 기간2)
    if k_smooth > 1:
        if use_ema:
            macd_norm = macd_norm.ewm(span=k_smooth, adjust=False).mean()
        else:
            macd_norm = macd_norm.rolling(k_smooth, min_periods=1).mean()

    # 4) 가격 기반 Slow%K
    ll = low.rolling(k_window, min_periods=1).min()
    hh = high.rolling(k_window, min_periods=1).max()
    k_raw = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    k_raw = k_raw.fillna(50)

    if k_smooth > 1:
        if use_ema:
            slow_k = k_raw.ewm(span=k_smooth, adjust=False).mean()
        else:
            slow_k = k_raw.rolling(k_smooth, min_periods=1).mean()
    else:
        slow_k = k_raw

    # 5) Composite K : 두 오실레이터 평균
    comp_k = (macd_norm + slow_k) / 2.0

    # 6) Composite D : d_smooth 단순이동평균
    if d_smooth > 1:
        comp_d = comp_k.rolling(d_smooth, min_periods=1).mean()
    else:
        comp_d = comp_k

    if clip:
        comp_k = comp_k.clip(0, 100)
        comp_d = comp_d.clip(0, 100)

    df['CompK'] = comp_k
    df['CompD'] = comp_d
    df['Diff']  = df['CompK'] - df['CompD']
    return df

# ───── 교차 판정 ───── #

def detect_cross(df: pd.DataFrame, ob: int = 80, os: int = 20) -> Optional[str]:
    """CompK / CompD 골든·데드 크로스 + 과매수·과매도 필터"""
    if len(df) < 2:
        return None
    prev_diff, curr_diff = df['Diff'].iloc[-2], df['Diff'].iloc[-1]
    prev_k = df['CompK'].iloc[-2]

    if prev_diff <= 0 < curr_diff:  # 골든 크로스
        return 'BUY' if prev_k < os else 'BUY_W'  # 과매도 영역에서만 강 신호
    if prev_diff >= 0 > curr_diff:  # 데드 크로스
        return 'SELL' if prev_k > ob else 'SELL_W'  # 과매수 영역에서만 강 신호
    return None

# ───── 차트 생성 ───── #

def make_chart(df: pd.DataFrame, code: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    name = code  # get_name 함수가 있다면 교체

    # 가격 패널
    ax1.plot(df['Date'], df['Close'], label='종가')
    ax1.plot(df['Date'], df['Close'].rolling(20).mean(), linestyle='--', label='MA20')
    ax1.set_title(f"{code} ({name})", fontproperties=font_prop)
    ax1.legend(prop=font_prop)

    # Composite 패널
    ax2.plot(df['Date'], df['CompK'], color='red', label='MACD+Slow%K')
    ax2.plot(df['Date'], df['CompD'], color='purple', label='MACD+Slow%D')
    ax2.axhline(20, color='gray', linestyle='--', linewidth=0.5)
    ax2.axhline(80, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_title('MACD+Stochastic (NH Style)', fontproperties=font_prop)
    ax2.legend(prop=font_prop, loc='upper left')

    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()

    path = f"{code}_chart.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path

# ───── 이하: 데이터 수집, EPS/PER 계산, Telegram 전송 등 기존 로직 그대로 (생략) ───── #

def main():
    """원본 main() 함수에서 add_composites / detect_cross / make_chart 만 교체"""
    pass  # 원본 main 구현을 여기에 복사하세요.

if __name__ == '__main__':
    main()
