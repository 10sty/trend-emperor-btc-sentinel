from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from config import DB_PATH, REPORTS_DIR


CANONICAL_TABLES = {
    "15m": "btc_15m_canonical",
    "1h": "btc_1h_canonical",
    "4h": "btc_4h_canonical",
    "1d": "btc_1d_canonical",
}

HORIZON_BARS = {
    "15m": 16,
    "1h": 24,
    "4h": 12,
    "1d": 7,
}

COMPRESSION_MIN_BARS = {
    "15m": 96,
    "1h": 72,
    "4h": 42,
    "1d": 90,
}

BOX_LOOKBACK = {
    "15m": 96,
    "1h": 72,
    "4h": 42,
    "1d": 90,
}

PULLBACK_TOLERANCE = 0.018
COMPRESSION_THRESHOLD = 0.02
ACCELERATION_SLOPE = 0.10


@dataclass(frozen=True)
class TrendClockBacktestRequest:
    timeframes: tuple[str, ...] = ("1h", "4h", "1d")
    start: str | None = None
    end: str | None = None
    sample: int | None = None
    mode: str = "direct"
    db_path: Path = DB_PATH
    output_dir: Path = REPORTS_DIR / "youtube_strategy_import"


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _month_key(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _can_connect_readonly(db_path: Path) -> bool:
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        con.close()
        return True
    except duckdb.IOException:
        return False


def _snapshot_db_if_locked(db_path: Path, output_dir: Path) -> tuple[Path, Path | None]:
    """Wait for the writer instead of duplicating the entire database on disk."""
    del output_dir  # Kept in the signature for existing callers.
    for attempt in range(180):
        if _can_connect_readonly(db_path):
            return db_path, None
        if attempt < 179:
            time.sleep(1)
    raise RuntimeError(f"Database remained locked for read-only access: {db_path}")


def load_ohlc(
    *,
    timeframe: str,
    start: datetime | None,
    end: datetime | None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    table = CANONICAL_TABLES[timeframe]
    where = ["is_complete = true"]
    params: list[Any] = []
    if start:
        where.append("timestamp_utc >= ?")
        params.append(start)
    if end:
        where.append("timestamp_utc <= ?")
        params.append(end)
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(
            f"""
            SELECT timestamp_utc, open, high, low, close, volume
            FROM {table}
            WHERE {' AND '.join(where)}
            ORDER BY timestamp_utc
            """,
            params,
        ).fetchdf()
    if df.empty:
        return df
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def add_trend_clock_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["ma120"] = close.rolling(120).mean()
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema60"] = close.ewm(span=60, adjust=False).mean()
    out["ema120"] = close.ewm(span=120, adjust=False).mean()

    ma_max = out[["ma20", "ma60", "ma120"]].max(axis=1)
    ma_min = out[["ma20", "ma60", "ma120"]].min(axis=1)
    out["ma_spread_pct"] = (ma_max - ma_min) / close
    out["compressed"] = out["ma_spread_pct"] <= COMPRESSION_THRESHOLD
    out["compression_run"] = out["compressed"].groupby((out["compressed"] != out["compressed"].shift()).cumsum()).cumcount() + 1
    out.loc[~out["compressed"], "compression_run"] = 0

    out["bullish_stack"] = (out["ma20"] > out["ma60"]) & (out["ma60"] > out["ma120"])
    out["bearish_stack"] = (out["ma20"] < out["ma60"]) & (out["ma60"] < out["ma120"])
    out["ma20_slope"] = out["ma20"].pct_change(5, fill_method=None)
    out["ma60_slope"] = out["ma60"].pct_change(10, fill_method=None)
    out["ma120_slope"] = out["ma120"].pct_change(20, fill_method=None)
    out["log_slope_120"] = (close / close.shift(120)).apply(lambda x: pd.NA if x <= 0 else x).astype("float64")
    out["log_slope_120"] = out["log_slope_120"].map(lambda x: None if pd.isna(x) else math.log(x))

    lookback = BOX_LOOKBACK[timeframe]
    out["box_high"] = high.shift(1).rolling(lookback).max()
    out["box_low"] = low.shift(1).rolling(lookback).min()
    out["prior_high_20"] = high.shift(1).rolling(20).max()
    out["prior_low_20"] = low.shift(1).rolling(20).min()

    out["above_ma20_group"] = (close > out["ma20"]) & (close > out["ema20"])
    out["below_ma20_group"] = (close < out["ma20"]) & (close < out["ema20"])
    out["near_ma60"] = ((low - out["ma60"]).abs() / close <= PULLBACK_TOLERANCE) | ((high - out["ma60"]).abs() / close <= PULLBACK_TOLERANCE)
    out["near_ma120"] = ((low - out["ma120"]).abs() / close <= PULLBACK_TOLERANCE) | ((high - out["ma120"]).abs() / close <= PULLBACK_TOLERANCE)
    out["volume_ratio"] = out["volume"] / out["volume"].rolling(60).mean()
    return out


def classify_clock_slope(row: pd.Series) -> str:
    slope = row.get("log_slope_120")
    if pd.isna(slope):
        return "unknown"
    if slope >= ACCELERATION_SLOPE:
        return "accelerating_up"
    if slope >= 0.025:
        return "stable_up"
    if slope > -0.025:
        return "sideways"
    if slope > -ACCELERATION_SLOPE:
        return "stable_down"
    return "accelerating_down"


def detect_opportunity(row: pd.Series, previous: pd.Series | None, timeframe: str) -> dict[str, Any] | None:
    if pd.isna(row.get("ma120")):
        return None
    clock = classify_clock_slope(row)
    min_run = COMPRESSION_MIN_BARS[timeframe]
    compressed_ready = int(row.get("compression_run") or 0) >= min_run
    prior_compressed = bool(previous is not None and int(previous.get("compression_run") or 0) >= min_run)

    common = {
        "timestamp_utc": row["timestamp_utc"],
        "timeframe": timeframe,
        "close": float(row["close"]),
        "clock_slope_type": clock,
        "ma_spread_pct": round(float(row.get("ma_spread_pct") or 0.0), 6),
        "compression_run": int(row.get("compression_run") or 0),
    }

    if (
        prior_compressed
        and row["bullish_stack"]
        and row["close"] > row["box_high"]
        and previous is not None
        and previous["close"] <= previous["box_high"]
        and row["ma20_slope"] > 0
    ):
        return {
            **common,
            "setup_type": "compression_breakout",
            "direction": "long",
            "reason": "密集成交区后均线张嘴多头排列，并向上突破区间。",
            "is_trade_setup": True,
            "entry_reference": float(row["box_high"]),
            "invalidation": float(row["ma60"]),
        }
    if (
        prior_compressed
        and row["bearish_stack"]
        and row["close"] < row["box_low"]
        and previous is not None
        and previous["close"] >= previous["box_low"]
        and row["ma20_slope"] < 0
    ):
        return {
            **common,
            "setup_type": "compression_breakout",
            "direction": "short",
            "reason": "密集成交区后均线张嘴空头排列，并向下突破区间。",
            "is_trade_setup": True,
            "entry_reference": float(row["box_low"]),
            "invalidation": float(row["ma60"]),
        }

    if (
        previous is not None
        and previous["high"] > previous["prior_high_20"]
        and row["close"] < previous["prior_high_20"]
        and row["below_ma20_group"]
        and row["ma20_slope"] < 0
    ):
        return {
            **common,
            "setup_type": "false_break_reversal",
            "direction": "short",
            "reason": "突破最近波峰后快速跌回，且跌破20均线组并拐头。",
            "is_trade_setup": True,
            "entry_reference": float(previous["prior_high_20"]),
            "invalidation": float(previous["high"]),
        }
    if (
        previous is not None
        and previous["low"] < previous["prior_low_20"]
        and row["close"] > previous["prior_low_20"]
        and row["above_ma20_group"]
        and row["ma20_slope"] > 0
    ):
        return {
            **common,
            "setup_type": "false_break_reversal",
            "direction": "long",
            "reason": "跌破最近波谷后快速收回，且站上20均线组并拐头。",
            "is_trade_setup": True,
            "entry_reference": float(previous["prior_low_20"]),
            "invalidation": float(previous["low"]),
        }

    if (
        clock == "stable_up"
        and row["bullish_stack"]
        and row["ma60_slope"] > 0
        and row["ma120_slope"] > 0
        and (row["near_ma60"] or row["near_ma120"])
        and row["above_ma20_group"]
        and previous is not None
        and not bool(previous["above_ma20_group"])
    ):
        return {
            **common,
            "setup_type": "trend_pullback",
            "direction": "long",
            "pullback_zone": "ma60" if bool(row["near_ma60"]) else "ma120",
            "reason": "稳定多头趋势中回撤到中长期均线附近后重新站回20均线组。",
            "is_trade_setup": True,
            "entry_reference": float(row["close"]),
            "invalidation": float(min(row["ma60"], row["ma120"])),
        }
    if (
        clock == "stable_down"
        and row["bearish_stack"]
        and row["ma60_slope"] < 0
        and row["ma120_slope"] < 0
        and (row["near_ma60"] or row["near_ma120"])
        and row["below_ma20_group"]
        and previous is not None
        and not bool(previous["below_ma20_group"])
    ):
        return {
            **common,
            "setup_type": "trend_pullback",
            "direction": "short",
            "pullback_zone": "ma60" if bool(row["near_ma60"]) else "ma120",
            "reason": "稳定空头趋势中反弹到中长期均线附近后重新跌回20均线组。",
            "is_trade_setup": True,
            "entry_reference": float(row["close"]),
            "invalidation": float(max(row["ma60"], row["ma120"])),
        }

    if clock in {"accelerating_up", "accelerating_down"} and float(row.get("volume_ratio") or 0) >= 1.8:
        return {
            **common,
            "setup_type": "acceleration_warning",
            "direction": "long" if clock == "accelerating_up" else "short",
            "reason": "斜率加速且放量；按视频原则，有持仓才跟随，无持仓不追。",
            "is_trade_setup": False,
            "entry_reference": float(row["close"]),
            "invalidation": float(row["ma20"]),
        }
    return None


def evaluate_outcome(df: pd.DataFrame, index: int, opportunity: dict[str, Any], timeframe: str) -> dict[str, Any]:
    horizon = HORIZON_BARS[timeframe]
    future = df.iloc[index + 1 : index + 1 + horizon]
    if future.empty:
        return {
            "horizon_bars": horizon,
            "return_pct": 0.0,
            "mfe": 0.0,
            "mae": 0.0,
            "outcome_label": "uncertain",
        }
    entry = float(opportunity["close"])
    direction = opportunity["direction"]
    if direction == "long":
        return_pct = (float(future.iloc[-1]["close"]) - entry) / entry
        mfe = (float(future["high"].max()) - entry) / entry
        mae = (float(future["low"].min()) - entry) / entry
    else:
        return_pct = (entry - float(future.iloc[-1]["close"])) / entry
        mfe = (entry - float(future["low"].min())) / entry
        mae = (entry - float(future["high"].max())) / entry

    if opportunity["setup_type"] == "acceleration_warning":
        label = "warning_followed" if return_pct > 0 else "warning_reversed"
    elif return_pct > 0 and mfe > abs(mae) * 1.05:
        label = "correct"
    elif return_pct < 0 and abs(mae) > mfe * 1.05:
        label = "incorrect"
    else:
        label = "uncertain"
    return {
        "horizon_bars": horizon,
        "return_pct": round(return_pct, 6),
        "mfe": round(mfe, 6),
        "mae": round(mae, 6),
        "outcome_label": label,
    }


def scan_timeframe(df: pd.DataFrame, timeframe: str, sample: int | None = None) -> list[dict[str, Any]]:
    featured = add_trend_clock_features(df, timeframe)
    events: list[dict[str, Any]] = []
    cooldown_until: dict[tuple[str, str], int] = {}
    pullback_zones_seen: set[tuple[str, str]] = set()
    start_index = max(140, BOX_LOOKBACK[timeframe] + 2)
    for index in range(start_index, len(featured)):
        row = featured.iloc[index]
        if not bool(row.get("bullish_stack")):
            pullback_zones_seen = {item for item in pullback_zones_seen if item[0] != "long"}
        if not bool(row.get("bearish_stack")):
            pullback_zones_seen = {item for item in pullback_zones_seen if item[0] != "short"}
        previous = featured.iloc[index - 1] if index > 0 else None
        opportunity = detect_opportunity(row, previous, timeframe)
        if not opportunity:
            continue
        if opportunity["setup_type"] == "trend_pullback":
            zone_key = (str(opportunity["direction"]), str(opportunity.get("pullback_zone") or "unknown"))
            if zone_key in pullback_zones_seen:
                continue
            pullback_zones_seen.add(zone_key)
        key = (str(opportunity["setup_type"]), str(opportunity["direction"]))
        if index < cooldown_until.get(key, -1):
            continue
        cooldown_until[key] = index + HORIZON_BARS[timeframe]
        opportunity["future_outcome"] = evaluate_outcome(featured, index, opportunity, timeframe)
        events.append(opportunity)
        if sample and len(events) >= sample:
            break
    return events


def _latest_before(featured: pd.DataFrame, timestamp: Any) -> pd.Series | None:
    if featured.empty:
        return None
    times = featured["timestamp_utc"]
    index = int(times.searchsorted(pd.Timestamp(timestamp), side="right")) - 1
    if index < 0:
        return None
    return featured.iloc[index]


def anchor_direction(row: pd.Series | None) -> str:
    if row is None or pd.isna(row.get("ma120")):
        return "none"
    clock = classify_clock_slope(row)
    if (
        bool(row.get("bullish_stack"))
        and float(row.get("ma60_slope") or 0.0) >= 0
        and float(row.get("ma120_slope") or 0.0) >= 0
        and clock in {"stable_up", "accelerating_up"}
    ):
        return "long"
    if (
        bool(row.get("bearish_stack"))
        and float(row.get("ma60_slope") or 0.0) <= 0
        and float(row.get("ma120_slope") or 0.0) <= 0
        and clock in {"stable_down", "accelerating_down"}
    ):
        return "short"
    return "none"


def scan_aligned_anchor_trigger(
    frames: dict[str, pd.DataFrame],
    *,
    sample: int | None = None,
) -> list[dict[str, Any]]:
    featured = {timeframe: add_trend_clock_features(df, timeframe) for timeframe, df in frames.items()}
    trigger_events = scan_timeframe(frames.get("1h", pd.DataFrame()), "1h", sample=None)
    aligned: list[dict[str, Any]] = []
    for event in trigger_events:
        if event.get("is_trade_setup") is False:
            continue
        timestamp = event["timestamp_utc"]
        anchors: dict[str, dict[str, Any]] = {}
        support = 0
        oppose = 0
        for timeframe in ("4h", "1d"):
            row = _latest_before(featured.get(timeframe, pd.DataFrame()), timestamp)
            direction = anchor_direction(row)
            anchors[timeframe] = {
                "direction": direction,
                "clock_slope_type": classify_clock_slope(row) if row is not None else "none",
                "timestamp_utc": row["timestamp_utc"] if row is not None else None,
                "ma_spread_pct": round(float(row.get("ma_spread_pct") or 0.0), 6) if row is not None else None,
            }
            if direction == event["direction"]:
                support += 1
            elif direction in {"long", "short"}:
                oppose += 1
        if support < 1 or oppose > 0:
            continue
        aligned_event = dict(event)
        aligned_event["setup_type"] = f"aligned_{event['setup_type']}"
        aligned_event["anchor_timeframes"] = anchors
        aligned_event["anchor_support_count"] = support
        aligned_event["anchor_opposition_count"] = oppose
        aligned_event["reason"] = f"4h/1d 高周期至少一个同向锚定，1h 触发：{event['reason']}"
        aligned.append(aligned_event)
        if sample and len(aligned) >= sample:
            break
    return aligned


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    trade_setups = [event for event in events if event.get("is_trade_setup") is not False]
    warnings = [event for event in events if event.get("is_trade_setup") is False]
    by_month: Counter[str] = Counter(_month_key(event["timestamp_utc"]) for event in trade_setups)
    by_type: Counter[str] = Counter(event["setup_type"] for event in events)
    by_trade_type: Counter[str] = Counter(event["setup_type"] for event in trade_setups)
    by_timeframe: Counter[str] = Counter(event["timeframe"] for event in trade_setups)
    by_direction: Counter[str] = Counter(event["direction"] for event in trade_setups)
    by_outcome: Counter[str] = Counter((event.get("future_outcome") or {}).get("outcome_label", "unknown") for event in trade_setups)

    type_quality: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in trade_setups:
        grouped[event["setup_type"]].append(event)
    for setup_type, items in grouped.items():
        correct = sum(1 for item in items if (item.get("future_outcome") or {}).get("outcome_label") == "correct")
        incorrect = sum(1 for item in items if (item.get("future_outcome") or {}).get("outcome_label") == "incorrect")
        returns = [float((item.get("future_outcome") or {}).get("return_pct") or 0.0) for item in items]
        type_quality[setup_type] = {
            "events": len(items),
            "correct": correct,
            "incorrect": incorrect,
            "accuracy_ex_uncertain": _safe_div(correct, correct + incorrect),
            "avg_return_pct": round(sum(returns) / len(returns), 6) if returns else 0.0,
        }

    months = sorted(by_month)
    target_months = sum(1 for count in by_month.values() if 2 <= count <= 3)
    high_noise_months = sum(1 for count in by_month.values() if count > 8)
    return {
        "total_events": len(events),
        "trade_setup_events": len(trade_setups),
        "warning_events": len(warnings),
        "month_count": len(months),
        "avg_events_per_month": round(sum(by_month.values()) / len(months), 4) if months else 0.0,
        "months_with_2_to_3_events": target_months,
        "months_with_2_to_3_rate": _safe_div(target_months, len(months)),
        "months_above_8_events": high_noise_months,
        "by_type": dict(sorted(by_type.items())),
        "by_trade_type": dict(sorted(by_trade_type.items())),
        "by_timeframe": dict(sorted(by_timeframe.items())),
        "by_direction": dict(sorted(by_direction.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "type_quality": type_quality,
        "monthly_distribution_sample": {month: by_month[month] for month in months[-24:]},
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Trend Clock Strategy Backtest",
        "",
        "只读历史审计：根据 YouTube 趋势时钟方法，扫描 BTC spot canonical 历史机会；不下单、不接 broker、不修改 DBE。",
        "",
        "## 视频方法映射",
        "- 趋势斜率：用固定窗口 log slope 分为加速上涨、稳定上涨、横盘、稳定下跌、加速下跌。",
        "- 密集成交区：MA20/60/120 之间距离 <= 2%，并持续足够长时间。",
        "- 突破埋伏：密集区后均线张嘴排列，并突破区间。",
        "- 假突破反转：刺破最近高/低后快速收回，并重新跌破/站上 20 均线组。",
        "- 稳定趋势回撤：中长期均线平行同向，价格回撤到 MA60/MA120 附近后恢复 20 均线组方向。",
        "- 加速行情：作为持仓/不追价警示，不直接作为新开机会。",
        "",
        "## 总体结果",
        f"- mode：{report.get('request', {}).get('mode', 'direct')}",
        f"- total_events：{summary['total_events']}",
        f"- trade_setup_events：{summary['trade_setup_events']}",
        f"- warning_events：{summary['warning_events']}",
        f"- avg_events_per_month：{summary['avg_events_per_month']}",
        f"- months_with_2_to_3_events：{summary['months_with_2_to_3_events']} / {summary['month_count']}",
        f"- months_with_2_to_3_rate：{summary['months_with_2_to_3_rate']}",
        f"- months_above_8_events：{summary['months_above_8_events']}",
        f"- by_type：{summary['by_type']}",
        f"- by_trade_type：{summary['by_trade_type']}",
        f"- by_timeframe：{summary['by_timeframe']}",
        f"- by_direction：{summary['by_direction']}",
        f"- by_outcome：{summary['by_outcome']}",
        "",
        "## Setup Quality",
    ]
    for setup_type, item in sorted(summary["type_quality"].items()):
        lines.append(
            f"- {setup_type}: events={item['events']} correct={item['correct']} "
            f"incorrect={item['incorrect']} accuracy_ex_uncertain={item['accuracy_ex_uncertain']} "
            f"avg_return_pct={item['avg_return_pct']}"
        )
    lines.extend(["", "## 最近 24 个月机会分布"])
    for month, count in summary["monthly_distribution_sample"].items():
        lines.append(f"- {month}: {count}")
    lines.extend(
        [
            "",
            "## 结论",
            report["interpretation"],
        ]
    )
    return "\n".join(lines) + "\n"


def interpret_summary(summary: dict[str, Any]) -> str:
    avg = float(summary.get("avg_events_per_month") or 0.0)
    target_rate = float(summary.get("months_with_2_to_3_rate") or 0.0)
    noise_months = int(summary.get("months_above_8_events") or 0)
    if 1.5 <= avg <= 4.0 and target_rate >= 0.25 and noise_months <= max(3, int(summary.get("month_count") or 0) * 0.15):
        return "视频趋势时钟规则能产生接近目标频率的机会，可进入更严格的 outcome replay 与系统主链路映射。"
    if avg < 1.5:
        return "机会仍偏少，说明仅靠严格趋势时钟条件仍过于保守；需要从密集区持续时间、回撤容忍度和低周期扩散条件继续审计。"
    return "机会偏多或噪音偏高；需要先强化趋势质量和假突破过滤，再考虑接入系统主链路。"


def run_trend_clock_backtest(request: TrendClockBacktestRequest) -> dict[str, Any]:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    start = _parse_dt(request.start)
    end = _parse_dt(request.end)
    events: list[dict[str, Any]] = []
    source_db_path, snapshot_path = _snapshot_db_if_locked(request.db_path, request.output_dir)
    try:
        if request.mode == "aligned_anchor_trigger":
            frames = {
                timeframe: load_ohlc(timeframe=timeframe, start=start, end=end, db_path=source_db_path)
                for timeframe in ("1h", "4h", "1d")
            }
            events = scan_aligned_anchor_trigger(frames, sample=request.sample)
        else:
            for timeframe in request.timeframes:
                if timeframe not in CANONICAL_TABLES:
                    raise ValueError(f"Unsupported timeframe: {timeframe}")
                df = load_ohlc(timeframe=timeframe, start=start, end=end, db_path=source_db_path)
                events.extend(scan_timeframe(df, timeframe, sample=request.sample))
    finally:
        if snapshot_path and snapshot_path.exists():
            snapshot_path.unlink()

    events.sort(key=lambda item: (item["timestamp_utc"], item["timeframe"], item["setup_type"]))
    summary = summarize_events(events)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            "timeframes": request.timeframes,
            "start": request.start,
            "end": request.end,
            "sample": request.sample,
            "mode": request.mode,
        },
        "source_video": "https://www.youtube.com/watch?v=MTmz6OLCykc",
        "used_temporary_db_snapshot": snapshot_path is not None,
        "summary": summary,
        "interpretation": interpret_summary(summary),
        "safety": {
            "read_only_backtest": True,
            "does_not_generate_orders": True,
            "does_not_modify_dbe": True,
            "spot_canonical_only": True,
        },
    }
    tag = _utc_tag()
    json_path = request.output_dir / f"trend_clock_strategy_backtest_{tag}.json"
    md_path = request.output_dir / f"trend_clock_strategy_backtest_{tag}.md"
    events_path = request.output_dir / f"trend_clock_strategy_events_{tag}.jsonl"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
    report["summary_json_path"] = str(json_path)
    report["summary_md_path"] = str(md_path)
    report["events_path"] = str(events_path)
    return report
