#!/usr/bin/env python3
import requests, pandas as pd, re
from datetime import date, timedelta

URL = "https://www.hankyung.com/koreamarket/data/consensus/list"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "*/*",
    "Referer": "https://www.hankyung.com/koreamarket/consensus/stock",
}

def only_code(s): return re.sub(r"\D", "", s or "")[-6:]

def to_num(x):
    try: return float(re.sub(r"[^\d.]", "", x)) if x not in (None, "", "0") else 0.0
    except: return 0.0

def get_upgraded(days=14, max_pages=20, size=200):
    end = date.today()
    start = end - timedelta(days=days)
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    rows = []
    for p in range(1, max_pages+1):
        params = {
            "listSize": size,
            "page": p,
            "search": "",
            "type": "CO",
            "start_date": s,
            "end_date": e,
        }
        r = requests.get(URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()  # ✅ JSON
        lst = data.get("list", [])
        if not lst: break

        for it in lst:
            tp  = to_num(it.get("target_price"))
            prv = to_num(it.get("previous_price"))
            if prv > 0 and tp > prv:              # 상향 조건
                code = only_code(it.get("itemcode"))
                if not code: continue
                rows.append({
                    "작성일": it.get("date"),
                    "종목코드": code,
                    "종목명": it.get("itemname"),
                    "이전TP": prv,
                    "현재TP": tp,
                    "의견": it.get("opnion"),
                    "증권사": it.get("provider"),
                    "제목": it.get("title"),
                })
    return pd.DataFrame(rows).drop_duplicates("종목코드")

if __name__ == "__main__":
    df = get_upgraded(days=14, max_pages=50)
    if df.empty:
        print("🔍 상향 종목 없음")
    else:
        print(df)
        print("\nSTOCK_LIST =", ",".join(df["종목코드"]))
