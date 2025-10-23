# -*- coding: utf-8 -*-
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple
import requests
import pandas as pd

UTC = timezone.utc
BINANCE_ENDPOINT = "https://api.binance.com/api/v3/klines"  # 공개 REST

def _binance_klines(symbol: str, interval: str, start_ms: Optional[int], end_ms: Optional[int], limit: int = 1000):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None: params["startTime"] = start_ms
    if end_ms   is not None: params["endTime"]   = end_ms
    r = requests.get(BINANCE_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_btc_usdt_1h_last_days(days: int = 65) -> pd.DataFrame:
    """
    Binance 1시간봉(UTC)을 최근 days일 + 여유분 가져옴.
    반환: ts_utc, open, high, low, close, volume
    """
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=days+2)
    end_ms = int(end.timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)

    rows = []
    cur_start = start_ms
    while True:
        data = _binance_klines("BTCUSDT", "1h", cur_start, end_ms, limit=1000)
        if not data:
            break
        for k in data:
            ts = datetime.fromtimestamp(k[0]/1000, tz=UTC)  # open time
            rows.append((
                ts, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
            ))
        last_close_ms = data[-1][6]
        if last_close_ms >= end_ms or len(data) < 1000:
            break
        cur_start = last_close_ms + 1
        time.sleep(0.05)

    if not rows:
        raise RuntimeError("Binance에서 klines를 받지 못했습니다.")
    df = pd.DataFrame(rows, columns=["ts_utc","open","high","low","close","volume"])
    df = df.sort_values("ts_utc").reset_index(drop=True)
    return df

def resample_to_kst_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    1시간봉(UTC) → KST 00:00–24:00 일봉 종가/수익률/RSI(14)
    """
    df = df_1h.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["ts_kst"] = df["ts_utc"].dt.tz_convert("Asia/Seoul")
    g = df.set_index("ts_kst").sort_index()

    # KST 일 단위 종가
    daily_close = g["close"].resample("1D").last().to_frame(name="close_kst")
    daily = daily_close.copy()
    daily["date_kst"] = daily.index.date
    daily["ret_kst"] = daily["close_kst"].pct_change()

    # RSI(14): 단순 평균 기반
    delta = daily["close_kst"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-12)
    rs = gain / loss
    daily["rsi14"] = 100 - (100 / (1 + rs))
    daily["rsi_sig"] = 0.0
    daily.loc[daily["rsi14"] > 55, "rsi_sig"] = 1.0
    daily.loc[daily["rsi14"] < 45, "rsi_sig"] = -1.0

    return daily.reset_index(drop=True)
