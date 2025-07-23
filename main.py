#!/usr/bin/env python3
"""
Stock Monitor – NH MTS‑style MACD+Stochastic + Hankyung Consensus
────────────────────────────────────────────────────────────────────
- 모든 종목 차트 전송 (개별 메시지)
- 마지막에 신호 종목 요약 전송
- (옵션) 한경 컨센서스 상향 종목 자동 수집
- 메시지/요약에 [HK] 또는 [ENV] 태그로 소스 구분
- (옵션) 현재가·목표가·리포트 제목 포함
"""

import os
import logging
import datetime as dt
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib import font_manager
import FinanceDataReader as fdr

# ──────────────────────────── ENV ────────────────────────────
TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")
SAVE_CSV      = os.getenv("SAVE_CSV", "false").lower() == "true"
FONT_PATH     = os.getenv("FONT_PATH", "fonts/NanumGothic.ttf")

USE_CONS      = os.getenv("USE_CONSENSUS", "false").lower() == "true"
CONS_DAYS     = int(os.getenv("CONS_DAYS", "14"))
CONS_PAGES    = int(os.getenv("CONS_PAGES", "50"))

STOCKS_ENV    = [s.strip() for s in os.getenv("STOCK_LIST", "005930").split(",") if s.strip()]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ──────────────────────────── Font ────────────────────────────
FONT_PATH = os.getenv("FONT_PATH", "fonts/NanumGothic.ttf")

def setup_korean_font(path: str):
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        fp = font_manager.FontProperties(fname=path)
        plt.rcParams['font.family'] = fp.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        return fp
    logging.warning("FONT_PATH not found: %s", path)
    return None

font_prop = setup_korean_font(FONT_PATH)

# ──────────────────────────── Consensus ────────────────────────────
def load_consensus_df() -> pd.DataFrame:
    """hk_consensus_up.get_upgraded() 결과 DF. 실패 시 빈 DF"""
    if not USE_CONS:
        return pd.DataFrame()
    try:
        from hk_consensus_up import get_upgraded
    except Exception as e:
        logging.error("hk_consensus_up import 실패: %s", e)
        return pd.DataFrame()
    try:
        df = get_upgraded(days=CONS_DAYS, max_pages=CONS_PAGES)
        if df is None or df.empty or '종목코드' not in df.columns:
            return pd.DataFrame()
        df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
        return df
    except Exception as e:
        logging.error("get_upgraded 실행 실패: %s", e)
        return pd.DataFrame()

# ──────────────────────────── Indicator ────────────────────────────
def add_composites(df: pd.DataFrame,
                   fast=12, slow=26,
                   k_window=14, k_smooth=3,
                   d_smooth=3, use_ema=True, clip=True) -> pd.DataFrame:
    close, high, low = df['Close'], df['High'], df['Low']

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_raw = ema_fast - ema_slow

    macd_min = macd_raw.rolling(k_window, min_periods=1).min()
    macd_max = macd_raw.rolling(k_window, min_periods=1).max()
    macd_norm = (macd_raw - macd_min) / (macd_max - macd_min).replace(0, np.nan) * 100
    macd_norm = macd_norm.fillna(50)
    if k_smooth > 1:
        macd_norm = macd_norm.ewm(span=k_smooth, adjust=False).mean() if use_ema \
            else macd_norm.rolling(k_smooth, min_periods=1).mean()

    ll = low.rolling(k_window, min_periods=1).min()
    hh = high.rolling(k_window, min_periods=1).max()
    k_raw = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    k_raw = k_raw.fillna(50)
    slow_k = (k_raw.ewm(span=k_smooth, adjust=False).mean() if (k_smooth > 1 and use_ema)
              else k_raw.rolling(k_smooth, min_periods=1).mean() if k_smooth > 1 else k_raw)

    comp_k = (macd_norm + slow_k) / 2.0
    comp_d = comp_k.rolling(d_smooth, min_periods=1).mean() if d_smooth > 1 else comp_k

    if clip:
        comp_k = comp_k.clip(0, 100)
        comp_d = comp_d.clip(0, 100)

    df['CompK'] = comp_k
    df['CompD'] = comp_d
    df['Diff']  = comp_k - comp_d
    return df


def detect_cross(df: pd.DataFrame, ob=80, os=20) -> Optional[str]:
    if len(df) < 2:
        return None
    prev_diff, curr_diff = df['Diff'].iloc[-2], df['Diff'].iloc[-1]
    prev_k = df['CompK'].iloc[-2]
    if prev_diff <= 0 < curr_diff:
        return 'BUY' if prev_k < os else 'BUY_W'
    if prev_diff >= 0 > curr_diff:
        return 'SELL' if prev_k > ob else 'SELL_W'
    return None

# ──────────────────────────── Data utils ────────────────────────────
_name_map: Dict[str, str] = {}

def normalize_code(code: str) -> str:
    return code.split('.')[0]

def get_korean_name(code: str) -> str:
    global _name_map
    if not _name_map:
        try:
            lst = fdr.StockListing('KRX')
            _name_map = lst.set_index('Code')['Name'].to_dict()
        except Exception:
            _name_map = {}
    return _name_map.get(normalize_code(code), code)

# fallback: yfinance
try:
    import yfinance as yf
except Exception:
    yf = None

def fetch_price_data(code: str, start: str) -> pd.DataFrame:
    norm = normalize_code(code)
    try:
        df = fdr.DataReader(norm, start)
        if df is not None and not df.empty:
            return df.reset_index()[['Date','Open','High','Low','Close','Volume']]
    except Exception:
        pass
    if yf is not None:
        try:
            ydf = yf.download(code if '.' in code else f"{code}.KS", start=start, progress=False)
            if not ydf.empty:
                ydf = ydf.rename(columns=str.title).reset_index()
                return ydf[['Date','Open','High','Low','Close','Volume']]
        except Exception:
            pass
    return pd.DataFrame()

# ──────────────────────────── Chart ────────────────────────────
def make_chart(df: pd.DataFrame, code: str) -> str:
    name = get_korean_name(code)
    title = f"{normalize_code(code)} ({name})"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={'height_ratios':[3,1]})

    ax1.plot(df['Date'], df['Close'], label='Close')
    ax1.plot(df['Date'], df['Close'].rolling(20).mean(), '--', label='MA20')
    ax1.set_title(title, fontproperties=font_prop)
    ax1.legend(prop=font_prop)

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

    path = f"{normalize_code(code)}_chart.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path

# ──────────────────────────── Telegram ────────────────────────────
def send_telegram(message: str, photo_path: Optional[str] = None) -> None:
    if not TOKEN or not CHAT_ID:
        logging.error("Telegram TOKEN / CHAT_ID 미설정")
        return
    try:
        if photo_path and os.path.exists(photo_path):
            with open(photo_path, 'rb') as f:
                requests.post(
                    f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                    data={'chat_id': CHAT_ID, 'caption': message},
                    files={'photo': f}, timeout=10)
        else:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={'chat_id': CHAT_ID, 'text': message}, timeout=10)
        logging.info("Telegram 전송 완료: %s", message)
    except Exception as e:
        logging.exception("Telegram 전송 실패: %s", e)

# ──────────────────────────── Main ────────────────────────────
def main() -> None:
    cons_df = load_consensus_df()
    if not cons_df.empty:
        stocks = cons_df['종목코드'].tolist()
        logging.info("콘센서스 상향 종목 %d개 사용", len(stocks))
    else:
        stocks = STOCKS_ENV

    cons_set = set(cons_df['종목코드']) if not cons_df.empty else set()
    cons_meta: Dict[str, Dict[str, Any]] = {}
    if not cons_df.empty:
        for _, r in cons_df.iterrows():
            cons_meta[r['종목코드']] = {
                '현재가': r.get('현재가'),
                '목표가': r.get('목표가'),
                '제목'  : r.get('리포트제목') or r.get('제목')
            }

    start_date = (dt.date.today() - dt.timedelta(days=365)).isoformat()
    alerts: List[Tuple[str, str, str]] = []

    for code in stocks:
        logging.info("%s: 데이터 수집", code)
        df = fetch_price_data(code, start_date)
        if df.empty:
            logging.warning("%s: 가격 데이터를 가져오지 못했습니다.", code)
            continue

        df = add_composites(df)
        signal = detect_cross(df)
        name = get_korean_name(code)
        chart_path = make_chart(df, code)

        norm = normalize_code(code)
        meta = cons_meta.get(norm, {})
        src_tag = "[HK]" if norm in cons_set else "[ENV]"

        parts = [src_tag, f"{norm} ({name})", f"신호: {signal if signal else '없음'}"]
        if meta.get('현재가') is not None:
            parts.append(f"현재가 {meta['현재가']}")
        if meta.get('목표가') is not None:
            parts.append(f"목표가 {meta['목표가']}")
        if meta.get('제목'):
            parts.append(f"리포트: {meta['제목']}")
        msg = " | ".join(parts)

        send_telegram(msg, chart_path)

        if signal:
            alerts.append((norm, name, signal))

        if SAVE_CSV:
            df.to_csv(f"{norm}_data.csv", index=False)

    # 요약 전송
    if alerts:
        summary_lines = [f"📈 오늘 신호 종목 ({len(alerts)}개)\n"]
        for c, n, s in alerts:
            tag = "[HK]" if c in cons_set else "[ENV]"
            summary_lines.append(f"- {tag} {c} ({n}): {s}")
        send_telegram("\n".join(summary_lines))
    else:
        send_telegram("오늘 신호 없음")

if __name__ == "__main__":
    main()
