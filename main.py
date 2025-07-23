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
STOCKS  = [s.strip() for s in os.getenv("STOCK_LIST", "005930").split(",") if s.strip()]
SAVE_CSV = os.getenv("SAVE_CSV", "false").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ──────────────────────────── 폰트 설정 ────────────────────────────
FONT_PATH = os.getenv("FONT_PATH", "fonts/NanumGothic.ttf")

def setup_korean_font(path: str):
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        fp = font_manager.FontProperties(fname=path)
        plt.rcParams['font.family'] = fp.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        return fp
    logging.warning("FONT_PATH not found: %s", path)
    return None

font_prop = setup_korean_font(FONT_PATH)

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
    name = get_korean_name(code)
    title = f"{normalize_code(code)} ({name})"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

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

_name_map = None  # 캐시용

def normalize_code(code: str) -> str:
    """'000660.KS' -> '000660' 처럼 접미사를 제거"""
    return code.split('.')[0]

def get_korean_name(code: str) -> str:
    """FinanceDataReader 상장목록에서 한글 종목명 조회 (실패 시 코드 반환)"""
    global _name_map
    if _name_map is None:
        try:
            lst = fdr.StockListing('KRX')  # Code, Name 등
            _name_map = lst.set_index('Code')['Name'].to_dict()
        except Exception:
            _name_map = {}
    return _name_map.get(normalize_code(code), code)

# yfinance는 선택적 사용
try:
    import yfinance as yf
except Exception:  # 설치 안 됐으면 무시
    yf = None

def fetch_price_data(code: str, start: str) -> pd.DataFrame:
    """우선 FDR, 실패 시 yfinance로 백업 조회"""
    norm = normalize_code(code)
    # 1) FDR
    try:
        df = fdr.DataReader(norm, start)
        if df is not None and not df.empty:
            return df.reset_index()[['Date','Open','High','Low','Close','Volume']]
    except Exception:
        pass
    # 2) yfinance fallback
    if yf is not None:
        try:
            ydf = yf.download(code if '.' in code else f"{code}.KS", start=start, progress=False)
            if not ydf.empty:
                ydf = ydf.rename(columns=str.title).reset_index()
                return ydf[['Date','Open','High','Low','Close','Volume']]
        except Exception:
            pass
    return pd.DataFrame()


# ───────────────────────────────── Main ────────────────────────────

def main() -> None:
    start_date = (dt.date.today() - dt.timedelta(days=365)).isoformat()
    alerts = []  # [(code,name,signal)]

    for code in STOCKS:
        logging.info("%s: 데이터 수집", code)
        df = fetch_price_data(code, start_date)
        if df.empty:
            logging.warning("%s: 가격 데이터를 가져오지 못했습니다.", code)
            continue

        df = add_composites(df)
        signal = detect_cross(df)
        name = get_korean_name(code)
        chart_path = make_chart(df, code)

        # 차트는 항상 전송
        sig_txt = signal if signal else '신호 없음'
        msg = f"{normalize_code(code)} ({name}) ➜ {sig_txt}"
        send_telegram(msg, chart_path)

        if signal:
            alerts.append((normalize_code(code), name, signal))

        if SAVE_CSV:
            df.to_csv(f"{normalize_code(code)}_data.csv", index=False)

    # 전체 요약 전송
    if alerts:
        summary_lines = [f"📈 오늘 신호 종목 ({len(alerts)}개)\n"]
        summary_lines += [f"- {c} ({n}): {s}" for c, n, s in alerts]
        send_telegram("\n".join(summary_lines))
    else:
        send_telegram("오늘 신호 없음")

if __name__ == "__main__":
    main()
