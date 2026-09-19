from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import REPORTS_DIR


DEFAULT_INPUT_DIR = REPORTS_DIR / "intraday_livermore_expectancy"
DEFAULT_OUTPUT_DIR = REPORTS_DIR / "intraday_livermore_profile_optimizer"


@dataclass(frozen=True)
class LivermoreProfile:
    name: str
    timeframes: tuple[str, ...] = ()
    setup_types: tuple[str, ...] = ()
    directions: tuple[str, ...] = ()
    min_setup_score: float | None = None
    min_volume_ratio: float | None = None
    min_risk_pct: float | None = None
    max_risk_pct: float | None = None
    max_ma_spread_pct: float | None = None
    min_proof_strength: float | None = None
    proof_stages: tuple[str, ...] = ()
    require_reaction_test: bool = False
    require_pyramid_eligible_shadow: bool = False
    clock_slope_types: tuple[str, ...] = ()
    default_target: float = 3.0
    target_by_direction: dict[str, float] | None = None


@dataclass(frozen=True)
class ProfileOptimizerRequest:
    events_path: Path | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    recent_start_year: int = 2021


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _round(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def latest_expectancy_events_path(input_dir: Path = DEFAULT_INPUT_DIR) -> Path:
    candidates = sorted(input_dir.glob("intraday_livermore_expectancy_events_*.jsonl"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No intraday Livermore expectancy events found in {input_dir}")
    return candidates[0]


def load_expectancy_events(events_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item["timestamp_utc"] = pd.Timestamp(item["timestamp_utc"])
            events.append(item)
    return events


def default_profiles() -> list[LivermoreProfile]:
    return [
        LivermoreProfile(
            name="livermore_reaction_revalidated_1h_long_3r",
            timeframes=("1h",),
            setup_types=("intraday_pullback_restart",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.012,
            max_risk_pct=0.035,
            max_ma_spread_pct=0.025,
            min_proof_strength=1.0,
            proof_stages=("reaction_revalidated",),
            require_reaction_test=True,
            require_pyramid_eligible_shadow=True,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="livermore_market_proven_1h_long_3r",
            timeframes=("1h",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.012,
            max_risk_pct=0.035,
            max_ma_spread_pct=0.025,
            min_proof_strength=1.0,
            proof_stages=("initial_breakout_probe", "reaction_revalidated"),
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="livermore_cross_source_proof_1h_long_3r",
            timeframes=("1h",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.01,
            max_risk_pct=0.035,
            max_ma_spread_pct=0.025,
            min_proof_strength=0.84,
            proof_stages=("initial_breakout_probe", "reaction_revalidated"),
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="livermore_proof_chain_1h_revalidated_long_3r",
            timeframes=("1h",),
            setup_types=("intraday_pullback_restart",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.012,
            max_risk_pct=0.035,
            min_proof_strength=0.80,
            proof_stages=("reaction_revalidated",),
            require_reaction_test=True,
            require_pyramid_eligible_shadow=True,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="livermore_least_resistance_1h_long_3r",
            timeframes=("1h",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.012,
            max_risk_pct=0.035,
            min_proof_strength=0.72,
            proof_stages=("initial_breakout_probe", "reaction_revalidated"),
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(name="all_signals_3r"),
        LivermoreProfile(
            name="breakout_core_3r",
            setup_types=("intraday_breakout_followthrough",),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="breakout_long3_short2",
            setup_types=("intraday_breakout_followthrough",),
            default_target=3.0,
            target_by_direction={"long": 3.0, "short": 2.0},
        ),
        LivermoreProfile(
            name="breakout_long_only_3r",
            setup_types=("intraday_breakout_followthrough",),
            directions=("long",),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="breakout_long_volume_1_2_trend_clock_3r",
            setup_types=("intraday_breakout_followthrough",),
            directions=("long",),
            min_volume_ratio=1.2,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="breakout_long_volume_1_3_trend_clock_3r",
            setup_types=("intraday_breakout_followthrough",),
            directions=("long",),
            min_volume_ratio=1.3,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="breakout_long_volume_1_1_compact_clock_3r",
            setup_types=("intraday_breakout_followthrough",),
            directions=("long",),
            min_volume_ratio=1.1,
            max_risk_pct=0.035,
            max_ma_spread_pct=0.04,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="long_trend_clock_volume_1_0_3r",
            directions=("long",),
            min_volume_ratio=1.0,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="long_trend_clock_volume_0_5_3r",
            directions=("long",),
            min_volume_ratio=0.5,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="institutional_long_volume_1_3_up_clock_risk_2_5pct_score_0_90_3r",
            timeframes=("15m", "1h"),
            directions=("long",),
            min_setup_score=0.90,
            min_volume_ratio=1.3,
            max_risk_pct=0.025,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="institutional_long_volume_1_3_stable_clock_risk_2_5pct_score_0_90_3r",
            timeframes=("15m", "1h"),
            directions=("long",),
            min_setup_score=0.90,
            min_volume_ratio=1.3,
            max_risk_pct=0.025,
            clock_slope_types=("stable_up",),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="institutional_long_volume_1_3_stable_clock_ma_3_5pct_risk_2_5pct_score_0_90_3r",
            timeframes=("15m", "1h"),
            directions=("long",),
            min_setup_score=0.90,
            min_volume_ratio=1.3,
            max_risk_pct=0.025,
            max_ma_spread_pct=0.035,
            clock_slope_types=("stable_up",),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="institutional_cross_source_score_0_90_risk_1_2_3_5pct_ma_2_5pct_up_clock_3r",
            timeframes=("15m", "1h"),
            directions=("long",),
            min_setup_score=0.90,
            min_risk_pct=0.012,
            max_risk_pct=0.035,
            max_ma_spread_pct=0.025,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="institutional_cross_source_score_0_90_volume_0_5_risk_1_5_3_5pct_ma_2_5pct_3r",
            timeframes=("15m", "1h"),
            directions=("long",),
            min_setup_score=0.90,
            min_volume_ratio=0.5,
            min_risk_pct=0.015,
            max_risk_pct=0.035,
            max_ma_spread_pct=0.025,
            default_target=3.0,
        ),
        LivermoreProfile(
            name="institutional_long_volume_1_2_up_clock_risk_2_5pct_score_0_90_3r",
            timeframes=("15m", "1h"),
            directions=("long",),
            min_setup_score=0.90,
            min_volume_ratio=1.2,
            max_risk_pct=0.025,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="institutional_long_volume_0_5_up_clock_risk_1_2_2_5pct_score_0_90_3r",
            timeframes=("15m", "1h"),
            directions=("long",),
            min_setup_score=0.90,
            min_volume_ratio=0.5,
            min_risk_pct=0.012,
            max_risk_pct=0.025,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="institutional_1h_long_volume_0_5_up_clock_risk_1_0_3_5pct_3r",
            timeframes=("1h",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.01,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="institutional_1h_pullback_stable_clock_risk_1_2_2_5pct_3r",
            timeframes=("1h",),
            setup_types=("intraday_pullback_restart",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.012,
            max_risk_pct=0.025,
            clock_slope_types=("stable_up",),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="all_trend_clock_volume_1_0_3r",
            min_volume_ratio=1.0,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="breakout_short_only_2r",
            setup_types=("intraday_breakout_followthrough",),
            directions=("short",),
            default_target=2.0,
        ),
        LivermoreProfile(
            name="short_breakout_volume_1_2_2r_research",
            setup_types=("intraday_breakout_followthrough",),
            directions=("short",),
            min_volume_ratio=1.2,
            max_risk_pct=0.035,
            clock_slope_types=("stable_down", "accelerating_down", "sideways"),
            default_target=2.0,
        ),
        LivermoreProfile(
            name="short_trend_clock_volume_0_8_3r_research",
            directions=("short",),
            min_volume_ratio=0.8,
            max_risk_pct=0.035,
            clock_slope_types=("stable_down", "accelerating_down", "sideways"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="high_score_breakout_3r",
            setup_types=("intraday_breakout_followthrough",),
            min_setup_score=0.95,
            default_target=3.0,
        ),
        LivermoreProfile(
            name="pullback_restart_3r",
            setup_types=("intraday_pullback_restart",),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="pullback_restart_long_clock_risk_1pct_3r",
            setup_types=("intraday_pullback_restart",),
            directions=("long",),
            min_risk_pct=0.01,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="pullback_restart_long_volume_0_5_clock_risk_1pct_3r",
            setup_types=("intraday_pullback_restart",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.01,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="pullback_restart_long_volume_1_0_risk_1pct_3r",
            setup_types=("intraday_pullback_restart",),
            directions=("long",),
            min_volume_ratio=1.0,
            min_risk_pct=0.01,
            max_risk_pct=0.035,
            default_target=3.0,
        ),
        LivermoreProfile(
            name="pullback_restart_1h_long_volume_0_5_clock_risk_1pct_3r",
            timeframes=("1h",),
            setup_types=("intraday_pullback_restart",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.01,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="pullback_restart_15m_long_volume_0_5_clock_risk_1pct_3r",
            timeframes=("15m",),
            setup_types=("intraday_pullback_restart",),
            directions=("long",),
            min_volume_ratio=0.5,
            min_risk_pct=0.01,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="breakout_1h_long_volume_1_3_clock_3r",
            timeframes=("1h",),
            setup_types=("intraday_breakout_followthrough",),
            directions=("long",),
            min_volume_ratio=1.3,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
        LivermoreProfile(
            name="breakout_15m_long_volume_1_3_clock_3r",
            timeframes=("15m",),
            setup_types=("intraday_breakout_followthrough",),
            directions=("long",),
            min_volume_ratio=1.3,
            max_risk_pct=0.035,
            clock_slope_types=("stable_up", "accelerating_up"),
            default_target=3.0,
        ),
    ]


def profile_matches(event: dict[str, Any], profile: LivermoreProfile) -> bool:
    if profile.timeframes and str(event.get("timeframe")) not in profile.timeframes:
        return False
    if profile.setup_types and str(event.get("setup_type")) not in profile.setup_types:
        return False
    if profile.directions and str(event.get("direction")) not in profile.directions:
        return False
    if profile.min_setup_score is not None and float(event.get("setup_score") or 0.0) < profile.min_setup_score:
        return False
    if profile.min_volume_ratio is not None and float(event.get("volume_ratio") or 0.0) < profile.min_volume_ratio:
        return False
    if profile.min_risk_pct is not None and float(event.get("risk_pct") or 0.0) < profile.min_risk_pct:
        return False
    if profile.max_risk_pct is not None and float(event.get("risk_pct") or 0.0) > profile.max_risk_pct:
        return False
    if profile.max_ma_spread_pct is not None and float(event.get("ma_spread_pct") or 0.0) > profile.max_ma_spread_pct:
        return False
    if profile.min_proof_strength is not None and float(event.get("proof_strength_score") or 0.0) < profile.min_proof_strength:
        return False
    if profile.proof_stages and str(event.get("market_proof_stage")) not in profile.proof_stages:
        return False
    if profile.require_reaction_test and not bool(event.get("reaction_test_present")):
        return False
    if profile.require_pyramid_eligible_shadow and not bool(event.get("pyramid_eligible_shadow")):
        return False
    if profile.clock_slope_types and str(event.get("clock_slope_type")) not in profile.clock_slope_types:
        return False
    return True


def target_for_event(event: dict[str, Any], profile: LivermoreProfile) -> float:
    if profile.target_by_direction:
        return float(profile.target_by_direction.get(str(event.get("direction")), profile.default_target))
    return float(profile.default_target)


def _target_result_for_event(event: dict[str, Any], profile: LivermoreProfile) -> dict[str, Any]:
    target = str(target_for_event(event, profile))
    result = (event.get("target_results") or {}).get(target)
    if not result:
        # JSON keys may use "3" if an external file was generated differently.
        result = (event.get("target_results") or {}).get(str(int(float(target))))
    return result or {}


def net_r_for_event(event: dict[str, Any], profile: LivermoreProfile, cost_multiplier: float = 1.0) -> float:
    result = _target_result_for_event(event, profile)
    if cost_multiplier != 1.0 and "gross_r" in result and "cost_r" in result:
        return float(result.get("gross_r") or 0.0) - (float(result.get("cost_r") or 0.0) * cost_multiplier)
    return float(result.get("net_r") or 0.0)


def _r_metrics(values: list[float]) -> dict[str, Any]:
    total = sum(values)
    gross_profit = sum(value for value in values if value > 0)
    gross_loss = abs(sum(value for value in values if value < 0))
    return {
        "event_count": len(values),
        "expectancy_net_r": _safe_div(total, len(values)),
        "profit_factor": _round(_safe_div(gross_profit, gross_loss), 4) if gross_loss else None,
        "win_rate": _safe_div(sum(1 for value in values if value > 0), len(values)),
    }


def cost_stress_metrics(events: list[dict[str, Any]], profile: LivermoreProfile) -> dict[str, Any]:
    selected = [event for event in events if profile_matches(event, profile)]
    stress: dict[str, Any] = {}
    for multiplier in (1.0, 2.0, 3.0):
        values = [net_r_for_event(event, profile, cost_multiplier=multiplier) for event in selected]
        stress[f"{multiplier:g}x_cost"] = _r_metrics(values) if values else {
            "expectancy_net_r": 0.0,
            "profit_factor": None,
            "win_rate": 0.0,
        }
    return stress


def target_robustness_metrics(events: list[dict[str, Any]], profile: LivermoreProfile) -> dict[str, Any]:
    selected = [event for event in events if profile_matches(event, profile)]
    robustness: dict[str, Any] = {}
    for target in (1.0, 1.5, 2.0, 3.0):
        values: list[float] = []
        labels: Counter[str] = Counter()
        bars_held: list[float] = []
        for event in selected:
            result = (event.get("target_results") or {}).get(str(target)) or {}
            if not result and float(target).is_integer():
                result = (event.get("target_results") or {}).get(str(int(target))) or {}
            if not result:
                continue
            values.append(float(result.get("net_r") or 0.0))
            labels[str(result.get("label") or "unknown")] += 1
            if result.get("bars_held") is not None:
                bars_held.append(float(result.get("bars_held") or 0.0))
        metrics = _r_metrics(values) if values else {
            "event_count": 0,
            "expectancy_net_r": 0.0,
            "profit_factor": None,
            "win_rate": 0.0,
        }
        metrics["avg_bars_held"] = _safe_div(sum(bars_held), len(bars_held))
        metrics["label_distribution"] = dict(sorted(labels.items()))
        robustness[f"{target:g}r"] = metrics
    return robustness


def era_metrics(events: list[dict[str, Any]], profile: LivermoreProfile) -> dict[str, Any]:
    buckets = {
        "2012_2017": [],
        "2018_2020": [],
        "2021_present": [],
    }
    for event in events:
        year = int(pd.Timestamp(event["timestamp_utc"]).year)
        value = net_r_for_event(event, profile)
        if year <= 2017:
            buckets["2012_2017"].append(value)
        elif year <= 2020:
            buckets["2018_2020"].append(value)
        else:
            buckets["2021_present"].append(value)
    return {key: _r_metrics(values) if values else {"event_count": 0, "expectancy_net_r": 0.0, "profit_factor": None, "win_rate": 0.0} for key, values in buckets.items()}


def summarize_profile(events: list[dict[str, Any]], profile: LivermoreProfile, recent_start_year: int = 2021) -> dict[str, Any]:
    selected = [event for event in events if profile_matches(event, profile)]
    selected.sort(key=lambda item: item["timestamp_utc"])
    r_values = [net_r_for_event(event, profile) for event in selected]
    total_r = sum(r_values)
    gross_profit = sum(value for value in r_values if value > 0)
    gross_loss = abs(sum(value for value in r_values if value < 0))
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    max_loss_streak = 0
    loss_streak = 0
    yearly: dict[str, float] = defaultdict(float)
    recent_values: list[float] = []
    by_direction: Counter[str] = Counter()
    by_setup: Counter[str] = Counter()

    for event, value in zip(selected, r_values):
        equity += value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
        if value < 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0
        year = str(pd.Timestamp(event["timestamp_utc"]).year)
        yearly[year] += value
        if int(year) >= recent_start_year:
            recent_values.append(value)
        by_direction[str(event.get("direction"))] += 1
        by_setup[str(event.get("setup_type"))] += 1

    positive_years = sum(1 for value in yearly.values() if value > 0)
    profile_payload = {
        "name": profile.name,
        "timeframes": profile.timeframes,
        "setup_types": profile.setup_types,
        "directions": profile.directions,
        "min_setup_score": profile.min_setup_score,
        "min_volume_ratio": profile.min_volume_ratio,
        "min_risk_pct": profile.min_risk_pct,
        "max_risk_pct": profile.max_risk_pct,
        "max_ma_spread_pct": profile.max_ma_spread_pct,
        "min_proof_strength": profile.min_proof_strength,
        "proof_stages": profile.proof_stages,
        "require_reaction_test": profile.require_reaction_test,
        "require_pyramid_eligible_shadow": profile.require_pyramid_eligible_shadow,
        "clock_slope_types": profile.clock_slope_types,
        "default_target": profile.default_target,
        "target_by_direction": profile.target_by_direction or {},
    }
    metrics = {
        "event_count": len(selected),
        "total_net_r": _round(total_r, 4),
        "expectancy_net_r": _safe_div(total_r, len(selected)),
        "profit_factor": _round(_safe_div(gross_profit, gross_loss), 4) if gross_loss else None,
        "win_rate": _safe_div(sum(1 for value in r_values if value > 0), len(r_values)),
        "max_drawdown_r": _round(abs(max_drawdown), 4),
        "max_loss_streak": max_loss_streak,
        "positive_year_rate": _safe_div(positive_years, len(yearly)),
        "worst_year_r": _round(min(yearly.values()), 4) if yearly else 0.0,
        "recent_start_year": recent_start_year,
        "recent_event_count": len(recent_values),
        "recent_expectancy_net_r": _safe_div(sum(recent_values), len(recent_values)),
        "by_direction": dict(sorted(by_direction.items())),
        "by_setup": dict(sorted(by_setup.items())),
        "yearly_net_r": {key: round(value, 4) for key, value in sorted(yearly.items())},
        "target_robustness": target_robustness_metrics(selected, profile),
        "cost_stress": cost_stress_metrics(selected, profile),
        "era_metrics": era_metrics(selected, profile),
    }
    return {
        "profile": profile_payload,
        "metrics": metrics,
        "readiness": readiness_from_metrics(metrics),
    }


def readiness_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if int(metrics.get("event_count") or 0) < 80:
        blockers.append("sample_below_80")
    if float(metrics.get("expectancy_net_r") or 0.0) < 0.20:
        blockers.append("expectancy_below_0_20r")
    if (metrics.get("profit_factor") is None) or float(metrics.get("profit_factor") or 0.0) < 1.35:
        blockers.append("profit_factor_below_1_35")
    if float(metrics.get("positive_year_rate") or 0.0) < 0.60:
        blockers.append("positive_year_rate_below_60pct")
    if float(metrics.get("recent_expectancy_net_r") or 0.0) <= 0.0:
        blockers.append("recent_expectancy_not_positive")
    cost_stress = metrics.get("cost_stress") or {}
    if "2x_cost" in cost_stress and float((cost_stress.get("2x_cost") or {}).get("expectancy_net_r") or 0.0) <= 0.0:
        blockers.append("double_cost_expectancy_not_positive")
    if "3x_cost" in cost_stress and float((cost_stress.get("3x_cost") or {}).get("expectancy_net_r") or 0.0) <= 0.0:
        warnings.append("triple_cost_expectancy_not_positive")
    if float(metrics.get("max_drawdown_r") or 0.0) > 35.0:
        warnings.append("max_drawdown_above_35r")
    if float(metrics.get("worst_year_r") or 0.0) < -15.0:
        warnings.append("worst_year_below_minus_15r")
    if blockers:
        verdict = "not_ready"
    elif warnings:
        verdict = "shadow_only"
    else:
        verdict = "live_pilot_candidate"
    return {
        "verdict": verdict,
        "blockers": blockers,
        "warnings": warnings,
        "does_not_execute": True,
    }


def rank_profile(result: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    metrics = result["metrics"]
    event_count = int(metrics.get("event_count") or 0)
    if event_count == 0:
        return (-1.0, 0.0, 0.0, -999.0, 0.0, 0.0, -999.0)
    verdict_bonus = {
        "live_pilot_candidate": 2.0,
        "shadow_only": 1.0,
        "not_ready": 0.0,
    }.get(result["readiness"]["verdict"], 0.0)
    return (
        verdict_bonus,
        float(metrics.get("recent_expectancy_net_r") or 0.0),
        float(metrics.get("positive_year_rate") or 0.0),
        float(metrics.get("worst_year_r") or 0.0),
        float(metrics.get("profit_factor") or 0.0),
        float(metrics.get("expectancy_net_r") or 0.0),
        -float(metrics.get("max_drawdown_r") or 0.0),
    )


def select_profile(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"selected_profile": None, "verdict": "no_profiles"}
    ranked = sorted(results, key=rank_profile, reverse=True)
    selected = ranked[0]
    return {
        "selected_profile": selected["profile"]["name"],
        "verdict": selected["readiness"]["verdict"],
        "metrics": selected["metrics"],
        "blockers": selected["readiness"]["blockers"],
        "warnings": selected["readiness"]["warnings"],
    }


def run_profile_optimizer(request: ProfileOptimizerRequest) -> dict[str, Any]:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = request.events_path or latest_expectancy_events_path()
    events = load_expectancy_events(events_path)
    profile_results = [summarize_profile(events, profile, request.recent_start_year) for profile in default_profiles()]
    profile_results.sort(key=rank_profile, reverse=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_events_path": str(events_path),
        "recent_start_year": request.recent_start_year,
        "selected": select_profile(profile_results),
        "profiles": profile_results,
        "safety": {
            "read_only_optimizer": True,
            "does_not_generate_orders": True,
            "does_not_modify_live_strategy": True,
            "does_not_touch_exchange": True,
        },
    }
    tag = _utc_tag()
    json_path = request.output_dir / f"intraday_livermore_profile_optimizer_{tag}.json"
    md_path = request.output_dir / f"intraday_livermore_profile_optimizer_{tag}.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    summary["summary_json_path"] = str(json_path)
    summary["summary_md_path"] = str(md_path)
    return summary


def _render_markdown(summary: dict[str, Any]) -> str:
    selected = summary["selected"]
    lines = [
        "# Intraday Livermore Profile Optimizer",
        "",
        "只读剖面优化：从已审计事件中筛选更适合 shadow/live pilot 的结构组合；不下单、不接交易所、不修改实时策略。",
        "",
        "选择优先级：先看近期正期望、正收益年份比例和最差年份，再看平均 R 与利润因子；避免只挑历史平均值漂亮但近期走弱的组合。",
        "",
        "## Selected",
        f"- selected_profile：{selected.get('selected_profile')}",
        f"- verdict：{selected.get('verdict')}",
        f"- blockers：{selected.get('blockers')}",
        f"- warnings：{selected.get('warnings')}",
        "",
        "## Profiles",
    ]
    for result in summary["profiles"]:
        metrics = result["metrics"]
        readiness = result["readiness"]
        lines.extend(
            [
                f"### {result['profile']['name']}",
                f"- verdict：{readiness['verdict']}",
                f"- events：{metrics['event_count']}",
                f"- expectancy_net_r：{metrics['expectancy_net_r']}",
                f"- profit_factor：{metrics['profit_factor']}",
                f"- win_rate：{metrics['win_rate']}",
                f"- max_drawdown_r：{metrics['max_drawdown_r']}",
                f"- positive_year_rate：{metrics['positive_year_rate']}",
                f"- worst_year_r：{metrics['worst_year_r']}",
                f"- recent_expectancy_net_r：{metrics['recent_expectancy_net_r']}",
                f"- 2x_cost_expectancy：{metrics['cost_stress']['2x_cost']['expectancy_net_r']}",
                f"- 3x_cost_expectancy：{metrics['cost_stress']['3x_cost']['expectancy_net_r']}",
                f"- era_2012_2017_expectancy：{metrics['era_metrics']['2012_2017']['expectancy_net_r']}",
                f"- era_2018_2020_expectancy：{metrics['era_metrics']['2018_2020']['expectancy_net_r']}",
                f"- era_2021_present_expectancy：{metrics['era_metrics']['2021_present']['expectancy_net_r']}",
                f"- blockers：{readiness['blockers']}",
                "",
            ]
        )
    return "\n".join(lines)
