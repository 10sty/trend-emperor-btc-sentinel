from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import REPORTS_DIR
from intraday_livermore_profile_optimizer import (
    LivermoreProfile,
    default_profiles,
    load_expectancy_events,
    profile_matches,
)


DEFAULT_OUTPUT_DIR = REPORTS_DIR / "intraday_livermore_profile_robustness"
DEFAULT_TARGETS = (1.0, 1.5, 2.0, 3.0)
DEFAULT_COST_MULTIPLIERS = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class ProfileRobustnessRequest:
    events_path: Path
    profile_name: str
    output_dir: Path = DEFAULT_OUTPUT_DIR
    target_multiples: tuple[float, ...] = DEFAULT_TARGETS
    cost_multipliers: tuple[float, ...] = DEFAULT_COST_MULTIPLIERS


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _round(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def profile_by_name(name: str) -> LivermoreProfile:
    for profile in default_profiles():
        if profile.name == name:
            return profile
    raise ValueError(f"Unknown Livermore profile: {name}")


def _target_key(target: float) -> str:
    return f"{float(target):.1f}"


def _result_for_target(event: dict[str, Any], target: float) -> dict[str, Any]:
    results = event.get("target_results") or {}
    key = _target_key(target)
    return results.get(key) or results.get(str(int(target))) or {}


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return abs(max_drawdown)


def _max_loss_streak(values: list[float]) -> int:
    streak = 0
    maximum = 0
    for value in values:
        if value < 0:
            streak += 1
            maximum = max(maximum, streak)
        else:
            streak = 0
    return maximum


def _profit_factor(values: list[float]) -> float | None:
    gross_profit = sum(value for value in values if value > 0)
    gross_loss = abs(sum(value for value in values if value < 0))
    if gross_loss == 0:
        return None
    return _round(gross_profit / gross_loss, 4)


def _r_metrics(values: list[float]) -> dict[str, Any]:
    return {
        "event_count": len(values),
        "avg_net_r": _safe_div(sum(values), len(values)),
        "total_net_r": _round(sum(values), 4),
        "win_rate": _safe_div(sum(1 for value in values if value > 0), len(values)),
        "profit_factor": _profit_factor(values),
        "max_drawdown_r": _round(_max_drawdown(values), 4),
        "max_loss_streak": _max_loss_streak(values),
    }


def _target_metrics(events: list[dict[str, Any]], target: float, cost_multiplier: float = 1.0) -> dict[str, Any]:
    values: list[float] = []
    labels: Counter[str] = Counter()
    bars_held: list[float] = []
    for event in events:
        result = _result_for_target(event, target)
        if not result:
            continue
        gross_r = float(result.get("gross_r") or 0.0)
        cost_r = float(result.get("cost_r") or 0.0)
        values.append(gross_r - cost_r * cost_multiplier)
        labels[str(result.get("label") or "unknown")] += 1
        bars_held.append(float(result.get("bars_held") or 0.0))

    metrics = _r_metrics(values)
    metrics.update(
        {
            "target_multiple": target,
            "cost_multiplier": cost_multiplier,
            "label_distribution": dict(sorted(labels.items())),
            "avg_bars_held": _safe_div(sum(bars_held), len(bars_held)),
        }
    )
    return metrics


def _monthly_frequency(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for event in events:
        timestamp = pd.Timestamp(event["timestamp_utc"])
        counts[timestamp.strftime("%Y-%m")] += 1
    if not counts:
        return {"active_months": 0, "avg_events_per_active_month": 0.0, "months_with_2_to_3": 0, "distribution": {}}
    return {
        "active_months": len(counts),
        "avg_events_per_active_month": _safe_div(sum(counts.values()), len(counts)),
        "months_with_2_to_3": sum(1 for value in counts.values() if 2 <= value <= 3),
        "max_events_in_month": max(counts.values()),
        "distribution": dict(sorted(counts.items())),
    }


def _mfe_mae_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    mfe_values = [float(event.get("mfe_r") or 0.0) for event in events]
    mae_values = [float(event.get("mae_r") or 0.0) for event in events]
    if not events:
        return {"avg_mfe_r": 0.0, "avg_mae_r": 0.0, "median_mfe_r": 0.0, "median_mae_r": 0.0}
    return {
        "avg_mfe_r": _safe_div(sum(mfe_values), len(mfe_values)),
        "avg_mae_r": _safe_div(sum(mae_values), len(mae_values)),
        "median_mfe_r": _round(float(pd.Series(mfe_values).median())),
        "median_mae_r": _round(float(pd.Series(mae_values).median())),
    }


def _bucket_counts(events: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(event.get(key) or "unknown") for event in events).items()))


def evaluate_profile_robustness(request: ProfileRobustnessRequest) -> dict[str, Any]:
    profile = profile_by_name(request.profile_name)
    all_events = load_expectancy_events(request.events_path)
    events = [event for event in all_events if profile_matches(event, profile)]
    events.sort(key=lambda item: item["timestamp_utc"])

    target_metrics = {
        _target_key(target): _target_metrics(events, target, 1.0)
        for target in request.target_multiples
    }
    cost_stress = {
        _target_key(target): {
            f"{multiplier:g}x_cost": _target_metrics(events, target, multiplier)
            for multiplier in request.cost_multipliers
        }
        for target in request.target_multiples
    }
    best_target = max(
        target_metrics.values(),
        key=lambda item: (
            float(item.get("avg_net_r") or 0.0),
            float(item.get("profit_factor") or 0.0),
            -float(item.get("max_drawdown_r") or 0.0),
        ),
        default={},
    )
    warnings: list[str] = []
    if len(events) < 80:
        warnings.append("sample_below_80")
    if float(best_target.get("avg_net_r") or 0.0) <= 0:
        warnings.append("best_target_expectancy_not_positive")
    if float(best_target.get("max_drawdown_r") or 0.0) > 10:
        warnings.append("drawdown_above_10r")
    if int(best_target.get("max_loss_streak") or 0) >= 5:
        warnings.append("loss_streak_above_5")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_events_path": str(request.events_path),
        "profile_name": request.profile_name,
        "event_count": len(events),
        "time_range": [
            pd.Timestamp(events[0]["timestamp_utc"]).isoformat() if events else None,
            pd.Timestamp(events[-1]["timestamp_utc"]).isoformat() if events else None,
        ],
        "target_metrics": target_metrics,
        "cost_stress": cost_stress,
        "best_target_by_avg_net_r": best_target,
        "monthly_frequency": _monthly_frequency(events),
        "mfe_mae": _mfe_mae_summary(events),
        "by_timeframe": _bucket_counts(events, "timeframe"),
        "by_direction": _bucket_counts(events, "direction"),
        "by_setup": _bucket_counts(events, "setup_type"),
        "by_regime": _bucket_counts(events, "market_regime"),
        "warnings": warnings,
        "safety": {
            "read_only_profile_robustness": True,
            "does_not_generate_orders": True,
            "does_not_modify_live_strategy": True,
            "does_not_touch_exchange": True,
        },
    }


def run_profile_robustness(request: ProfileRobustnessRequest) -> dict[str, Any]:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_profile_robustness(request)
    tag = _utc_tag()
    json_path = request.output_dir / f"intraday_livermore_profile_robustness_{tag}.json"
    md_path = request.output_dir / f"intraday_livermore_profile_robustness_{tag}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    report["summary_json_path"] = str(json_path)
    report["summary_md_path"] = str(md_path)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    best = report.get("best_target_by_avg_net_r") or {}
    freq = report.get("monthly_frequency") or {}
    mfe_mae = report.get("mfe_mae") or {}
    lines = [
        "# Intraday Livermore Profile Robustness",
        "",
        "只读稳健性审计：评估固定画像在不同盈亏比、成本、月度频率与亏损序列下的承压能力；不下单、不接交易所、不修改实时策略。",
        "",
        "## Summary",
        f"- profile：{report['profile_name']}",
        f"- events：{report['event_count']}",
        f"- time_range：{report['time_range']}",
        f"- best_target：{best.get('target_multiple')}R",
        f"- best_avg_net_r：{best.get('avg_net_r')}",
        f"- best_profit_factor：{best.get('profit_factor')}",
        f"- best_max_drawdown_r：{best.get('max_drawdown_r')}",
        f"- best_max_loss_streak：{best.get('max_loss_streak')}",
        f"- avg_events_per_active_month：{freq.get('avg_events_per_active_month')}",
        f"- months_with_2_to_3：{freq.get('months_with_2_to_3')}",
        f"- avg_mfe_r：{mfe_mae.get('avg_mfe_r')}",
        f"- avg_mae_r：{mfe_mae.get('avg_mae_r')}",
        f"- warnings：{report.get('warnings')}",
        "",
        "## Target Metrics",
    ]
    for key, item in (report.get("target_metrics") or {}).items():
        lines.extend(
            [
                f"### {key}R",
                f"- avg_net_r：{item.get('avg_net_r')}",
                f"- win_rate：{item.get('win_rate')}",
                f"- profit_factor：{item.get('profit_factor')}",
                f"- max_drawdown_r：{item.get('max_drawdown_r')}",
                f"- max_loss_streak：{item.get('max_loss_streak')}",
                f"- label_distribution：{item.get('label_distribution')}",
                "",
            ]
        )
    return "\n".join(lines)
