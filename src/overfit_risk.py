from __future__ import annotations

from typing import Any


def stability_score(in_sample: dict[str, Any], out_of_sample: dict[str, Any]) -> float:
    if not in_sample.get("sample_size") or not out_of_sample.get("sample_size"):
        return 0.0
    win_gap = abs(float(in_sample.get("win_rate_20") or 0) - float(out_of_sample.get("win_rate_20") or 0))
    avg_gap = abs(float(in_sample.get("avg_return_20") or 0) - float(out_of_sample.get("avg_return_20") or 0))
    median_gap = abs(float(in_sample.get("median_return_20") or 0) - float(out_of_sample.get("median_return_20") or 0))
    sample_penalty = 0.25 if min(in_sample["sample_size"], out_of_sample["sample_size"]) < 30 else 0.0
    score = 100.0 - min(100.0, win_gap * 180 + avg_gap * 900 + median_gap * 700 + sample_penalty * 100)
    return round(max(0.0, score), 2)


def classify_overfit(
    in_sample: dict[str, Any],
    out_of_sample: dict[str, Any],
    score: float,
) -> str:
    min_sample = min(int(in_sample.get("sample_size") or 0), int(out_of_sample.get("sample_size") or 0))
    is_avg = float(in_sample.get("avg_return_20") or 0)
    oos_avg = float(out_of_sample.get("avg_return_20") or 0)
    if min_sample < 20 or score < 35 or (is_avg > 0 and oos_avg < 0 and abs(is_avg - oos_avg) > 0.01):
        return "HIGH"
    if min_sample < 75 or score < 65:
        return "MEDIUM"
    return "LOW"
