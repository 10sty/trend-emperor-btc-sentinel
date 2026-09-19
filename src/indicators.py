from __future__ import annotations

import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    out["ma20"] = close.rolling(20, min_periods=20).mean()
    out["ma60"] = close.rolling(60, min_periods=60).mean()
    out["ma120"] = close.rolling(120, min_periods=120).mean()
    return out


def ma_state(row: pd.Series, compression_threshold: float = 0.006) -> str:
    ma20 = row.get("ma20")
    ma60 = row.get("ma60")
    ma120 = row.get("ma120")
    close = row.get("close")
    if pd.isna(ma20) or pd.isna(ma60) or pd.isna(ma120) or pd.isna(close) or close == 0:
        return "unknown"
    spread = (max(ma20, ma60, ma120) - min(ma20, ma60, ma120)) / close
    if spread <= compression_threshold:
        return "compressed"
    if ma20 > ma60 > ma120:
        return "bullish_alignment"
    if ma20 < ma60 < ma120:
        return "bearish_alignment"
    return "mixed"


def price_vs_ma120(row: pd.Series) -> str:
    ma120 = row.get("ma120")
    close = row.get("close")
    if pd.isna(ma120) or pd.isna(close):
        return "unknown"
    if close > ma120:
        return "above_ma120"
    if close < ma120:
        return "below_ma120"
    return "at_ma120"


def add_ma_structure(df: pd.DataFrame) -> pd.DataFrame:
    out = add_moving_averages(df)
    out["ma_state"] = out.apply(ma_state, axis=1)
    out["price_vs_ma120"] = out.apply(price_vs_ma120, axis=1)
    return out
