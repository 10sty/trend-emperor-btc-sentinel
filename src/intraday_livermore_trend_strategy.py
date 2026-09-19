from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import DB_PATH, REPORTS_DIR
from trend_clock_strategy_engine import (
    CANONICAL_TABLES,
    add_trend_clock_features,
    anchor_direction,
    classify_clock_slope,
    load_ohlc,
    _snapshot_db_if_locked,
)


INTRADAY_TIMEFRAMES = ("1h",)
ANCHOR_TIMEFRAMES = ("4h", "1d")
OUTCOME_HORIZON_BARS = {
    "15m": 64,
    "1h": 24,
}
COOLDOWN_BARS = {
    "15m": 96,
    "1h": 24,
}
MIN_SETUP_SCORE = 0.82
MAX_INTRADAY_RISK_PCT = 0.035
MIN_INTRADAY_RISK_PCT = 0.0012
REQUIRED_ANCHOR_SUPPORT = 2


@dataclass(frozen=True)
class IntradayLivermoreBacktestRequest:
    timeframes: tuple[str, ...] = INTRADAY_TIMEFRAMES
    start: str | None = None
    end: str | None = None
    sample: int | None = None
    full_history: bool = False
    rr_multiple: float = 3.0
    min_setup_score: float = MIN_SETUP_SCORE
    required_anchor_support: int = REQUIRED_ANCHOR_SUPPORT
    db_path: Path = DB_PATH
    data_source_label: str = "binance_spot_canonical"
    output_dir: Path = REPORTS_DIR / "intraday_livermore_trend"


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


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _month_key(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")


def _round(value: Any, digits: int = 6) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def add_intraday_livermore_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    featured = add_trend_clock_features(df, timeframe)
    if featured.empty:
        return featured
    prev_close = featured["close"].shift(1)
    true_range = pd.concat(
        [
            featured["high"] - featured["low"],
            (featured["high"] - prev_close).abs(),
            (featured["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    featured["atr14"] = true_range.rolling(14).mean()
    featured["range_pct"] = (featured["high"] - featured["low"]) / featured["close"]
    return featured


def _latest_before(featured: pd.DataFrame, timestamp: Any) -> pd.Series | None:
    if featured.empty:
        return None
    times = featured["timestamp_utc"]
    index = int(times.searchsorted(pd.Timestamp(timestamp), side="right")) - 1
    if index < 0:
        return None
    return featured.iloc[index]


def _anchor_snapshot(anchor_rows: dict[str, pd.Series | None], direction: str) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    support = 0
    opposition = 0
    for timeframe, row in anchor_rows.items():
        anchor = anchor_direction(row)
        if anchor == direction:
            support += 1
        elif anchor in {"long", "short"}:
            opposition += 1
        snapshots[timeframe] = {
            "direction": anchor,
            "clock_slope_type": classify_clock_slope(row) if row is not None else "none",
            "timestamp_utc": row["timestamp_utc"] if row is not None else None,
            "ma_spread_pct": _round(row.get("ma_spread_pct"), 6) if row is not None else None,
        }
    return {
        "anchors": snapshots,
        "support_count": support,
        "opposition_count": opposition,
    }


def anchor_timeframes_for_signal(timeframe: str) -> tuple[str, ...]:
    if timeframe == "15m":
        return ("1h", "4h", "1d")
    return ANCHOR_TIMEFRAMES


def required_anchor_support_for_signal(timeframe: str, configured: int) -> int:
    if timeframe == "15m":
        return max(3, configured)
    return configured


def _risk_is_usable(entry: float, invalidation: float) -> bool:
    risk_pct = abs(entry - invalidation) / entry if entry else 0.0
    return MIN_INTRADAY_RISK_PCT <= risk_pct <= MAX_INTRADAY_RISK_PCT


def _setup_score(row: pd.Series, direction: str, anchor: dict[str, Any], setup_type: str) -> float:
    score = 0.30
    if direction == "long":
        if bool(row.get("bullish_stack")):
            score += 0.12
        if float(row.get("ma20_slope") or 0.0) > 0 and float(row.get("ma60_slope") or 0.0) >= 0:
            score += 0.10
        if bool(row.get("above_ma20_group")):
            score += 0.08
    else:
        if bool(row.get("bearish_stack")):
            score += 0.12
        if float(row.get("ma20_slope") or 0.0) < 0 and float(row.get("ma60_slope") or 0.0) <= 0:
            score += 0.10
        if bool(row.get("below_ma20_group")):
            score += 0.08
    score += min(int(anchor["support_count"]), 2) * 0.14
    if int(anchor["opposition_count"]) == 0:
        score += 0.08
    if setup_type == "pullback_restart":
        score += 0.08
    if float(row.get("volume_ratio") or 0.0) >= 1.15:
        score += 0.04
    return round(min(score, 1.0), 4)


def _proof_strength_score(
    row: pd.Series,
    direction: str,
    anchor: dict[str, Any],
    setup_type: str,
    *,
    risk_pct: float,
) -> float:
    score = 0.20
    if setup_type == "pullback_restart":
        score += 0.24
    else:
        score += 0.10
    score += min(int(anchor["support_count"]), 2) * 0.16
    if int(anchor["opposition_count"]) == 0:
        score += 0.12
    if 0.012 <= risk_pct <= 0.035:
        score += 0.08
    if float(row.get("volume_ratio") or 0.0) >= 0.5:
        score += 0.04
    if direction == "long":
        if bool(row.get("bullish_stack")):
            score += 0.08
        if float(row.get("ma20_slope") or 0.0) > 0 and float(row.get("ma60_slope") or 0.0) >= 0:
            score += 0.08
    else:
        if bool(row.get("bearish_stack")):
            score += 0.08
        if float(row.get("ma20_slope") or 0.0) < 0 and float(row.get("ma60_slope") or 0.0) <= 0:
            score += 0.08
    return round(min(score, 1.0), 4)


def _market_proof_stage(setup_type: str) -> str:
    if setup_type == "pullback_restart":
        return "reaction_revalidated"
    return "initial_breakout_probe"


def _livermore_sequence(direction: str, setup_type: str) -> list[str]:
    side = "多头" if direction == "long" else "空头"
    if setup_type == "pullback_restart":
        return [
            f"{side}最小阻力线已由高周期确认",
            "市场先给出趋势方向，再出现回撤反应",
            "回撤后重新启动，说明趋势仍被市场照顾",
            "只允许把它视为趋势证明，不允许把它当作自动交易命令",
        ]
    return [
        f"{side}最小阻力线开始定义",
        "突破后只属于初始试探",
        "还需要后续回撤不破并再启动，才算完整趋势证明",
        "只允许观察，不允许因突破本身自动追单",
    ]


def detect_intraday_livermore_event(
    row: pd.Series,
    previous: pd.Series | None,
    timeframe: str,
    anchor_rows: dict[str, pd.Series | None],
    *,
    min_setup_score: float = MIN_SETUP_SCORE,
    required_anchor_support: int = REQUIRED_ANCHOR_SUPPORT,
) -> dict[str, Any] | None:
    if previous is None or pd.isna(row.get("ma120")):
        return None

    candidates: list[dict[str, Any]] = []
    close = float(row["close"])
    timestamp = row["timestamp_utc"]

    long_breakout = (
        close > float(row.get("prior_high_20") or math.inf)
        and float(previous["close"]) <= float(previous.get("prior_high_20") or math.inf)
        and bool(row.get("above_ma20_group"))
        and float(row.get("ma20_slope") or 0.0) > 0
        and float(row.get("volume_ratio") or 0.0) >= 1.05
    )
    short_breakout = (
        close < float(row.get("prior_low_20") or -math.inf)
        and float(previous["close"]) >= float(previous.get("prior_low_20") or -math.inf)
        and bool(row.get("below_ma20_group"))
        and float(row.get("ma20_slope") or 0.0) < 0
        and float(row.get("volume_ratio") or 0.0) >= 1.05
    )
    long_restart = (
        bool(row.get("above_ma20_group"))
        and not bool(previous.get("above_ma20_group"))
        and float(row.get("ma20_slope") or 0.0) > 0
        and close > float(previous.get("high") or close)
        and (bool(row.get("near_ma60")) or bool(row.get("near_ma120")))
    )
    short_restart = (
        bool(row.get("below_ma20_group"))
        and not bool(previous.get("below_ma20_group"))
        and float(row.get("ma20_slope") or 0.0) < 0
        and close < float(previous.get("low") or close)
        and (bool(row.get("near_ma60")) or bool(row.get("near_ma120")))
    )

    if long_breakout:
        candidates.append(
            {
                "setup_type": "breakout_followthrough",
                "direction": "long",
                "entry_reference": _round(row.get("prior_high_20"), 2),
                "invalidation": float(min(row.get("ma20"), row.get("prior_low_20"))),
                "proof_chain": ["突破最近 20 根高点", "收盘站上 20 均线组", "短均线斜率向上"],
            }
        )
    if short_breakout:
        candidates.append(
            {
                "setup_type": "breakout_followthrough",
                "direction": "short",
                "entry_reference": _round(row.get("prior_low_20"), 2),
                "invalidation": float(max(row.get("ma20"), row.get("prior_high_20"))),
                "proof_chain": ["跌破最近 20 根低点", "收盘跌回 20 均线组下方", "短均线斜率向下"],
            }
        )
    if long_restart:
        candidates.append(
            {
                "setup_type": "pullback_restart",
                "direction": "long",
                "entry_reference": close,
                "invalidation": float(min(row.get("low"), row.get("prior_low_20"), row.get("ma60"))),
                "proof_chain": ["回撤后重新站上 20 均线组", "收盘突破上一根高点", "市场重新证明多头有效"],
            }
        )
    if short_restart:
        candidates.append(
            {
                "setup_type": "pullback_restart",
                "direction": "short",
                "entry_reference": close,
                "invalidation": float(max(row.get("high"), row.get("prior_high_20"), row.get("ma60"))),
                "proof_chain": ["反弹后重新跌回 20 均线组下方", "收盘跌破上一根低点", "市场重新证明空头有效"],
            }
        )

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        if not _risk_is_usable(close, float(candidate["invalidation"])):
            continue
        anchor = _anchor_snapshot(anchor_rows, candidate["direction"])
        if anchor["support_count"] < required_anchor_support or anchor["opposition_count"] > 0:
            continue
        score = _setup_score(row, candidate["direction"], anchor, candidate["setup_type"])
        if score < min_setup_score:
            continue
        risk_pct = abs(close - float(candidate["invalidation"])) / close
        proof_stage = _market_proof_stage(candidate["setup_type"])
        proof_strength = _proof_strength_score(
            row,
            candidate["direction"],
            anchor,
            candidate["setup_type"],
            risk_pct=risk_pct,
        )
        scored.append(
            {
                "timestamp_utc": timestamp,
                "timeframe": timeframe,
                "setup_type": f"intraday_{candidate['setup_type']}",
                "direction": candidate["direction"],
                "close": close,
                "entry": close,
                "entry_reference": candidate["entry_reference"],
                "invalidation": round(float(candidate["invalidation"]), 2),
                "risk_pct": round(risk_pct, 6),
                "setup_score": score,
                "proof_strength_score": proof_strength,
                "market_proof_stage": proof_stage,
                "reaction_test_present": candidate["setup_type"] == "pullback_restart",
                "least_resistance_direction": candidate["direction"],
                "pyramid_eligible_shadow": (
                    proof_stage == "reaction_revalidated"
                    and proof_strength >= 0.80
                    and anchor["support_count"] >= required_anchor_support
                    and anchor["opposition_count"] == 0
                ),
                "clock_slope_type": classify_clock_slope(row),
                "ma_spread_pct": _round(row.get("ma_spread_pct"), 6),
                "volume_ratio": _round(row.get("volume_ratio"), 4),
                "anchor_timeframes": anchor["anchors"],
                "anchor_support_count": anchor["support_count"],
                "anchor_opposition_count": anchor["opposition_count"],
                "livermore_principle": "市场证明方向后才跟随；用失效位保护，不预测。",
                "proof_chain": candidate["proof_chain"],
                "livermore_sequence": _livermore_sequence(candidate["direction"], candidate["setup_type"]),
                "reason": _human_reason(candidate["direction"], candidate["setup_type"], timeframe, score),
            }
        )
    if not scored:
        return None
    return max(scored, key=lambda item: item["setup_score"])


def _human_reason(direction: str, setup_type: str, timeframe: str, score: float) -> str:
    side = "多头" if direction == "long" else "空头"
    if setup_type == "pullback_restart":
        return f"{timeframe} {side}趋势回撤后再启动，高周期至少一个同向锚定；利弗莫尔式等待市场证明后跟随，score={score}。"
    return f"{timeframe} {side}突破后有收盘跟随，高周期至少一个同向锚定；不追未证明波动，score={score}。"


def evaluate_rr3_outcome(
    df: pd.DataFrame,
    index: int,
    event: dict[str, Any],
    *,
    rr_multiple: float = 3.0,
) -> dict[str, Any]:
    horizon = OUTCOME_HORIZON_BARS[event["timeframe"]]
    future = df.iloc[index + 1 : index + 1 + horizon]
    entry = float(event["entry"])
    invalidation = float(event["invalidation"])
    direction = event["direction"]
    risk = abs(entry - invalidation)
    if future.empty or risk <= 0:
        return {
            "horizon_bars": horizon,
            "rr_multiple": rr_multiple,
            "target": None,
            "outcome_label": "uncertain",
            "bars_to_outcome": None,
            "return_pct": 0.0,
            "mfe": 0.0,
            "mae": 0.0,
        }
    target = entry + risk * rr_multiple if direction == "long" else entry - risk * rr_multiple
    outcome = "timeout"
    bars_to_outcome: int | None = None
    for offset, (_, bar) in enumerate(future.iterrows(), start=1):
        if direction == "long":
            hit_stop = float(bar["low"]) <= invalidation
            hit_target = float(bar["high"]) >= target
        else:
            hit_stop = float(bar["high"]) >= invalidation
            hit_target = float(bar["low"]) <= target
        if hit_stop and hit_target:
            outcome = "loss_stop_first"
            bars_to_outcome = offset
            break
        if hit_stop:
            outcome = "loss"
            bars_to_outcome = offset
            break
        if hit_target:
            outcome = "win_rr3"
            bars_to_outcome = offset
            break

    last_close = float(future.iloc[-1]["close"])
    if direction == "long":
        return_pct = (last_close - entry) / entry
        mfe = (float(future["high"].max()) - entry) / entry
        mae = (float(future["low"].min()) - entry) / entry
    else:
        return_pct = (entry - last_close) / entry
        mfe = (entry - float(future["low"].min())) / entry
        mae = (entry - float(future["high"].max())) / entry
    return {
        "horizon_bars": horizon,
        "rr_multiple": rr_multiple,
        "target": round(target, 2),
        "outcome_label": outcome,
        "bars_to_outcome": bars_to_outcome,
        "return_pct": round(return_pct, 6),
        "mfe": round(mfe, 6),
        "mae": round(mae, 6),
    }


def scan_intraday_livermore_events(
    frames: dict[str, pd.DataFrame],
    *,
    timeframes: tuple[str, ...] = INTRADAY_TIMEFRAMES,
    sample: int | None = None,
    rr_multiple: float = 3.0,
    min_setup_score: float = MIN_SETUP_SCORE,
    required_anchor_support: int = REQUIRED_ANCHOR_SUPPORT,
) -> list[dict[str, Any]]:
    featured = {timeframe: add_intraday_livermore_features(df, timeframe) for timeframe, df in frames.items()}
    events: list[dict[str, Any]] = []
    cooldown_until: dict[tuple[str, str, str], int] = {}

    for timeframe in timeframes:
        df = featured.get(timeframe, pd.DataFrame())
        if df.empty:
            continue
        start_index = 140
        for index in range(start_index, len(df)):
            row = df.iloc[index]
            previous = df.iloc[index - 1] if index > 0 else None
            anchor_rows = {
                anchor_timeframe: _latest_before(featured.get(anchor_timeframe, pd.DataFrame()), row["timestamp_utc"])
                for anchor_timeframe in anchor_timeframes_for_signal(timeframe)
            }
            event = detect_intraday_livermore_event(
                row,
                previous,
                timeframe,
                anchor_rows,
                min_setup_score=min_setup_score,
                required_anchor_support=required_anchor_support_for_signal(timeframe, required_anchor_support),
            )
            if not event:
                continue
            key = (timeframe, event["direction"], event["setup_type"])
            if index < cooldown_until.get(key, -1):
                continue
            cooldown_until[key] = index + COOLDOWN_BARS[timeframe]
            event["future_outcome"] = evaluate_rr3_outcome(df, index, event, rr_multiple=rr_multiple)
            events.append(event)
            if sample and len(events) >= sample:
                return sorted(events, key=lambda item: (item["timestamp_utc"], item["timeframe"]))
    return sorted(events, key=lambda item: (item["timestamp_utc"], item["timeframe"]))


def summarize_intraday_livermore_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_month: Counter[str] = Counter(_month_key(event["timestamp_utc"]) for event in events)
    by_timeframe: Counter[str] = Counter(event["timeframe"] for event in events)
    by_direction: Counter[str] = Counter(event["direction"] for event in events)
    by_setup: Counter[str] = Counter(event["setup_type"] for event in events)
    by_outcome: Counter[str] = Counter((event.get("future_outcome") or {}).get("outcome_label", "unknown") for event in events)
    wins = by_outcome.get("win_rr3", 0)
    losses = by_outcome.get("loss", 0) + by_outcome.get("loss_stop_first", 0)
    evaluated = wins + losses

    monthly_returns: dict[str, list[float]] = defaultdict(list)
    for event in events:
        monthly_returns[_month_key(event["timestamp_utc"])].append(
            float((event.get("future_outcome") or {}).get("return_pct") or 0.0)
        )
    month_count = len(by_month)
    months_with_2_to_3 = sum(1 for count in by_month.values() if 2 <= count <= 3)
    months_with_at_least_2 = sum(1 for count in by_month.values() if count >= 2)
    quiet_months = sum(1 for count in by_month.values() if count == 0)
    months = sorted(by_month)

    setup_quality: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["setup_type"]].append(event)
    for setup_type, items in sorted(grouped.items()):
        setup_wins = sum(1 for item in items if (item.get("future_outcome") or {}).get("outcome_label") == "win_rr3")
        setup_losses = sum(
            1
            for item in items
            if (item.get("future_outcome") or {}).get("outcome_label") in {"loss", "loss_stop_first"}
        )
        returns = [float((item.get("future_outcome") or {}).get("return_pct") or 0.0) for item in items]
        setup_quality[setup_type] = {
            "events": len(items),
            "rr3_wins": setup_wins,
            "losses": setup_losses,
            "rr3_win_rate_ex_timeout": _safe_div(setup_wins, setup_wins + setup_losses),
            "avg_return_pct": round(sum(returns) / len(returns), 6) if returns else 0.0,
        }

    returns = [float((event.get("future_outcome") or {}).get("return_pct") or 0.0) for event in events]
    return {
        "total_events": len(events),
        "month_count": month_count,
        "avg_events_per_month": round(sum(by_month.values()) / month_count, 4) if month_count else 0.0,
        "months_with_2_to_3_events": months_with_2_to_3,
        "months_with_2_to_3_rate": _safe_div(months_with_2_to_3, month_count),
        "months_with_at_least_2_events": months_with_at_least_2,
        "months_with_at_least_2_rate": _safe_div(months_with_at_least_2, month_count),
        "quiet_months": quiet_months,
        "rr3_wins": wins,
        "losses": losses,
        "rr3_win_rate_ex_timeout": _safe_div(wins, evaluated),
        "avg_return_pct": round(sum(returns) / len(returns), 6) if returns else 0.0,
        "by_timeframe": dict(sorted(by_timeframe.items())),
        "by_direction": dict(sorted(by_direction.items())),
        "by_setup": dict(sorted(by_setup.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "setup_quality": setup_quality,
        "monthly_distribution_sample": {month: by_month[month] for month in months[-36:]},
    }


def interpret_intraday_summary(summary: dict[str, Any]) -> str:
    avg = float(summary.get("avg_events_per_month") or 0.0)
    target_rate = float(summary.get("months_with_2_to_3_rate") or 0.0)
    win_rate = float(summary.get("rr3_win_rate_ex_timeout") or 0.0)
    if 2.0 <= avg <= 5.0 and target_rate >= 0.20 and win_rate >= 0.30:
        return "日内利弗莫尔规则具备可观察稳定性：机会频率接近每月 2-3 次目标，且 3:1 结果不依赖高胜率也能成立。"
    if avg < 2.0:
        return "规则仍偏保守：机会频率低于每月 2 次，需要后续审计是高周期锚定过硬，还是回撤再启动条件太窄。"
    if win_rate < 0.30:
        return "机会数量够，但 3:1 命中偏弱；需要后续检查突破跟随是否太晚，或失效位是否过窄。"
    return "机会偏多，可能引入噪音；后续应收紧趋势证明链或提高高周期锚定质量。"


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Intraday Livermore Trend Strategy Backtest",
        "",
        "只读历史回测：提炼利弗莫尔趋势派精神，扫描 BTC spot canonical 日内趋势机会；不交易、不下单、不接 broker、不修改实时 DBE/Kernel。",
        "",
        "## 核心精神",
        "- 只沿最小阻力线观察，不猜底摸顶。",
        "- 市场证明方向后才跟随：突破、回撤测试、再启动。",
        "- 每个机会必须有明确失效位；失效就承认判断错。",
        "- 日内只做 15m/1h 触发，高周期 4h/1d 做方向锚定。",
        "- 结果用 3:1 盈亏比做历史稳定性审计，但本模块不生成订单。",
        "",
        "## 总体统计",
        f"- total_events：{summary['total_events']}",
        f"- month_count：{summary['month_count']}",
        f"- avg_events_per_month：{summary['avg_events_per_month']}",
        f"- months_with_2_to_3_events：{summary['months_with_2_to_3_events']} / {summary['month_count']}",
        f"- months_with_at_least_2_events：{summary['months_with_at_least_2_events']} / {summary['month_count']}",
        f"- rr3_wins：{summary['rr3_wins']}",
        f"- losses：{summary['losses']}",
        f"- rr3_win_rate_ex_timeout：{summary['rr3_win_rate_ex_timeout']}",
        f"- avg_return_pct：{summary['avg_return_pct']}",
        f"- by_timeframe：{summary['by_timeframe']}",
        f"- by_direction：{summary['by_direction']}",
        f"- by_setup：{summary['by_setup']}",
        f"- by_outcome：{summary['by_outcome']}",
        "",
        "## Setup Quality",
    ]
    for setup_type, item in sorted(summary["setup_quality"].items()):
        lines.append(
            f"- {setup_type}: events={item['events']} rr3_wins={item['rr3_wins']} "
            f"losses={item['losses']} rr3_win_rate_ex_timeout={item['rr3_win_rate_ex_timeout']} "
            f"avg_return_pct={item['avg_return_pct']}"
        )
    lines.extend(["", "## 最近 36 个月机会分布"])
    for month, count in summary["monthly_distribution_sample"].items():
        lines.append(f"- {month}: {count}")
    lines.extend(["", "## 结论", report["interpretation"], ""])
    return "\n".join(lines)


def run_intraday_livermore_backtest(request: IntradayLivermoreBacktestRequest) -> dict[str, Any]:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    start = _parse_dt(request.start)
    end = _parse_dt(request.end)
    source_db_path, snapshot_path = _snapshot_db_if_locked(request.db_path, request.output_dir)
    try:
        frames = {
            timeframe: load_ohlc(timeframe=timeframe, start=start, end=end, db_path=source_db_path)
            for timeframe in tuple(dict.fromkeys((*request.timeframes, "1h", *ANCHOR_TIMEFRAMES)))
            if timeframe in CANONICAL_TABLES
        }
        events = scan_intraday_livermore_events(
            frames,
            timeframes=request.timeframes,
            sample=request.sample,
            rr_multiple=request.rr_multiple,
            min_setup_score=request.min_setup_score,
            required_anchor_support=request.required_anchor_support,
        )
    finally:
        if snapshot_path and snapshot_path.exists():
            snapshot_path.unlink()

    summary = summarize_intraday_livermore_events(events)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            "timeframes": request.timeframes,
            "start": request.start,
            "end": request.end,
            "sample": request.sample,
            "full_history": request.full_history,
            "rr_multiple": request.rr_multiple,
            "min_setup_score": request.min_setup_score,
            "required_anchor_support": request.required_anchor_support,
            "db_path": str(request.db_path),
            "data_source_label": request.data_source_label,
        },
        "summary": summary,
        "interpretation": interpret_intraday_summary(summary),
        "safety": {
            "read_only_backtest": True,
            "does_not_generate_orders": True,
            "does_not_modify_dbe_or_kernel": True,
            "does_not_touch_exchange_private_api": True,
            "market_data_only": True,
            "data_source_label": request.data_source_label,
            "does_not_affect_forward_shadow_72h": True,
        },
    }
    tag = _utc_tag()
    json_path = request.output_dir / f"intraday_livermore_backtest_{tag}.json"
    md_path = request.output_dir / f"intraday_livermore_backtest_{tag}.md"
    events_path = request.output_dir / f"intraday_livermore_events_{tag}.jsonl"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
    report["summary_json_path"] = str(json_path)
    report["summary_md_path"] = str(md_path)
    report["events_path"] = str(events_path)
    return report
