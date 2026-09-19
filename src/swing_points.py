from __future__ import annotations

import pandas as pd


def add_swing_points(df: pd.DataFrame, left_bars: int = 3, right_bars: int = 3) -> pd.DataFrame:
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be positive integers.")
    out = df.copy()
    highs = pd.to_numeric(out["high"], errors="coerce")
    lows = pd.to_numeric(out["low"], errors="coerce")
    swing_high = pd.Series(True, index=out.index)
    swing_low = pd.Series(True, index=out.index)
    for offset in range(1, left_bars + 1):
        swing_high &= highs > highs.shift(offset)
        swing_low &= lows < lows.shift(offset)
    for offset in range(1, right_bars + 1):
        swing_high &= highs >= highs.shift(-offset)
        swing_low &= lows <= lows.shift(-offset)
    edge = left_bars + right_bars
    if edge:
        swing_high.iloc[:left_bars] = False
        swing_high.iloc[-right_bars:] = False
        swing_low.iloc[:left_bars] = False
        swing_low.iloc[-right_bars:] = False
    out["swing_type"] = None
    out.loc[swing_high, "swing_type"] = "swing_high"
    out.loc[swing_low, "swing_type"] = "swing_low"
    return out


def add_structure_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["structure_label"] = None
    last_high = None
    last_low = None
    swing_idx = out.index[out["swing_type"].notna()]
    highs = out["high"].to_numpy()
    lows = out["low"].to_numpy()
    swing_types = out["swing_type"].to_numpy()
    labels = [None] * len(out)
    for idx in swing_idx:
        pos = out.index.get_loc(idx)
        swing_type = swing_types[pos]
        if swing_type == "swing_high":
            label = "HH" if last_high is not None and highs[pos] > last_high else "LH"
            labels[pos] = label
            last_high = highs[pos]
        elif swing_type == "swing_low":
            label = "HL" if last_low is not None and lows[pos] > last_low else "LL"
            labels[pos] = label
            last_low = lows[pos]
    out["structure_label"] = labels
    return out
