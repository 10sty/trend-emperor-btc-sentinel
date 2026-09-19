from __future__ import annotations

import json
from collections import Counter
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
    net_r_for_event,
    profile_matches,
    select_profile,
    summarize_profile,
)


DEFAULT_INPUT_DIR = REPORTS_DIR / "intraday_livermore_expectancy_public_spot"
DEFAULT_OUTPUT_DIR = REPORTS_DIR / "intraday_livermore_walk_forward_public_spot"


@dataclass(frozen=True)
class WalkForwardRequest:
    events_path: Path | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    fold_mode: str = "calendar"
    train_years: int = 2
    test_years: int = 1
    train_event_window: int = 60
    test_event_window: int = 20
    step_events: int = 20
    min_train_events: int = 20
    min_test_events: int = 5
    recent_start_year: int = 2021
    fixed_profile_name: str | None = None


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
        raise FileNotFoundError(f"No public-spot expectancy events found in {input_dir}")
    return candidates[0]


def _year_range(events: list[dict[str, Any]]) -> tuple[int, int]:
    if not events:
        raise ValueError("No events available for walk-forward analysis")
    years = [int(pd.Timestamp(event["timestamp_utc"]).year) for event in events]
    return min(years), max(years)


def _events_between(events: list[dict[str, Any]], start_year: int, end_year_exclusive: int) -> list[dict[str, Any]]:
    selected = [
        event
        for event in events
        if start_year <= int(pd.Timestamp(event["timestamp_utc"]).year) < end_year_exclusive
    ]
    selected.sort(key=lambda item: item["timestamp_utc"])
    return selected


def profile_by_name(name: str) -> LivermoreProfile:
    for profile in default_profiles():
        if profile.name == name:
            return profile
    raise ValueError(f"Unknown Livermore profile selected by optimizer: {name}")


def r_metrics_for_profile(events: list[dict[str, Any]], profile: LivermoreProfile) -> dict[str, Any]:
    selected = [event for event in events if profile_matches(event, profile)]
    selected.sort(key=lambda item: item["timestamp_utc"])
    values = [net_r_for_event(event, profile) for event in selected]
    gross_profit = sum(value for value in values if value > 0)
    gross_loss = abs(sum(value for value in values if value < 0))
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    by_direction: Counter[str] = Counter(str(event.get("direction")) for event in selected)
    by_setup: Counter[str] = Counter(str(event.get("setup_type")) for event in selected)
    by_regime: Counter[str] = Counter(str(event.get("market_regime")) for event in selected)

    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)

    return {
        "event_count": len(selected),
        "total_net_r": _round(sum(values), 4),
        "expectancy_net_r": _safe_div(sum(values), len(values)),
        "profit_factor": _round(_safe_div(gross_profit, gross_loss), 4) if gross_loss else None,
        "win_rate": _safe_div(sum(1 for value in values if value > 0), len(values)),
        "max_drawdown_r": _round(abs(max_drawdown), 4),
        "by_direction": dict(sorted(by_direction.items())),
        "by_setup": dict(sorted(by_setup.items())),
        "by_regime": dict(sorted(by_regime.items())),
    }


def _fold_status(test_metrics: dict[str, Any], min_test_events: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if int(test_metrics.get("event_count") or 0) < min_test_events:
        reasons.append("test_sample_below_minimum")
    if float(test_metrics.get("expectancy_net_r") or 0.0) <= 0.0:
        reasons.append("test_expectancy_not_positive")
    profit_factor = test_metrics.get("profit_factor")
    if profit_factor is not None and float(profit_factor) < 1.05:
        reasons.append("test_profit_factor_below_1_05")
    if reasons:
        return "fail", reasons
    return "pass", []


def _fold_recent_start_year(train_start: int, train_end_exclusive: int) -> int:
    return max(train_start, train_end_exclusive - 1)


def _fatal_train_blockers(selected: dict[str, Any], min_profile_events: int) -> list[str]:
    metrics = selected.get("metrics") or {}
    event_count = int(metrics.get("event_count") or 0)
    blockers = list(selected.get("blockers") or [])
    fatal: list[str] = []
    for blocker in blockers:
        if blocker == "sample_below_80" and event_count >= min_profile_events:
            continue
        fatal.append(blocker)
    return fatal


def _fixed_profile_selection(
    train_events: list[dict[str, Any]],
    profile_name: str,
    recent_start_year: int,
) -> dict[str, Any]:
    profile = profile_by_name(profile_name)
    result = summarize_profile(train_events, profile, recent_start_year)
    readiness = result.get("readiness") or {}
    return {
        "selected_profile": profile.name,
        "verdict": readiness.get("verdict"),
        "metrics": result.get("metrics") or {},
        "blockers": readiness.get("blockers") or [],
        "warnings": readiness.get("warnings") or [],
    }


def build_walk_forward_folds(
    events: list[dict[str, Any]],
    *,
    train_years: int,
    test_years: int,
    min_train_events: int,
    min_test_events: int,
    recent_start_year: int,
    fixed_profile_name: str | None = None,
) -> list[dict[str, Any]]:
    first_year, last_year = _year_range(events)
    folds: list[dict[str, Any]] = []
    final_start = last_year - train_years - test_years + 2
    for train_start in range(first_year, final_start):
        train_end = train_start + train_years
        test_end = train_end + test_years
        train_events = _events_between(events, train_start, train_end)
        test_events = _events_between(events, train_end, test_end)
        fold_recent_start_year = _fold_recent_start_year(train_start, train_end)
        fold_id = f"{train_start}_{train_end - 1}_to_{train_end}_{test_end - 1}"
        if len(train_events) < min_train_events:
            folds.append(
                {
                    "fold_id": fold_id,
                    "train_years": [train_start, train_end - 1],
                    "test_years": [train_end, test_end - 1],
                    "status": "skipped",
                    "failure_reasons": ["train_sample_below_minimum"],
                    "train_event_count": len(train_events),
                    "test_event_count": len(test_events),
                }
            )
            continue

        if fixed_profile_name:
            selection_mode = "fixed_profile"
            selected = _fixed_profile_selection(train_events, fixed_profile_name, fold_recent_start_year)
        else:
            selection_mode = "rolling_optimizer"
            train_results = [
                summarize_profile(train_events, profile, fold_recent_start_year)
                for profile in default_profiles()
            ]
            selected = select_profile(train_results)
        selected_profile_name = selected.get("selected_profile")
        if not selected_profile_name:
            folds.append(
                {
                    "fold_id": fold_id,
                    "train_years": [train_start, train_end - 1],
                    "test_years": [train_end, test_end - 1],
                    "status": "skipped",
                    "failure_reasons": ["no_profile_selected"],
                    "selection_mode": selection_mode,
                    "train_event_count": len(train_events),
                    "test_event_count": len(test_events),
                }
            )
            continue

        profile = profile_by_name(str(selected_profile_name))
        test_metrics = r_metrics_for_profile(test_events, profile)
        status, failure_reasons = _fold_status(test_metrics, min_test_events)
        fatal_train_blockers = _fatal_train_blockers(selected, min_test_events)
        if fatal_train_blockers:
            status = "fail"
            failure_reasons = [f"train_{blocker}" for blocker in fatal_train_blockers] + failure_reasons
        folds.append(
            {
                "fold_id": fold_id,
                "train_years": [train_start, train_end - 1],
                "test_years": [train_end, test_end - 1],
                "fold_recent_start_year": fold_recent_start_year,
                "status": status,
                "failure_reasons": failure_reasons,
                "selection_mode": selection_mode,
                "selected_profile": selected_profile_name,
                "train_verdict": selected.get("verdict"),
                "train_blockers": selected.get("blockers") or [],
                "train_warnings": selected.get("warnings") or [],
                "train_metrics": selected.get("metrics") or {},
                "test_metrics": test_metrics,
                "train_event_count": len(train_events),
                "test_event_count": len(test_events),
            }
        )
    return folds


def build_event_window_folds(
    events: list[dict[str, Any]],
    *,
    fixed_profile_name: str,
    train_event_window: int,
    test_event_window: int,
    step_events: int,
    min_test_events: int,
) -> list[dict[str, Any]]:
    if train_event_window <= 0 or test_event_window <= 0 or step_events <= 0:
        raise ValueError("Event-window settings must be positive")

    profile = profile_by_name(fixed_profile_name)
    matching_events = [event for event in events if profile_matches(event, profile)]
    matching_events.sort(key=lambda item: item["timestamp_utc"])

    folds: list[dict[str, Any]] = []
    final_start = len(matching_events) - train_event_window - test_event_window + 1
    if final_start <= 0:
        return folds

    for start in range(0, final_start, step_events):
        train_start = start
        train_end = start + train_event_window
        test_start = train_end
        test_end = test_start + test_event_window
        train_events = matching_events[train_start:train_end]
        test_events = matching_events[test_start:test_end]
        train_start_ts = pd.Timestamp(train_events[0]["timestamp_utc"])
        train_end_ts = pd.Timestamp(train_events[-1]["timestamp_utc"])
        test_start_ts = pd.Timestamp(test_events[0]["timestamp_utc"])
        test_end_ts = pd.Timestamp(test_events[-1]["timestamp_utc"])
        fold_recent_start_year = int(train_end_ts.year)
        fold_id = f"events_{train_start:04d}_{train_end - 1:04d}_to_{test_start:04d}_{test_end - 1:04d}"

        selected = _fixed_profile_selection(train_events, fixed_profile_name, fold_recent_start_year)
        test_metrics = r_metrics_for_profile(test_events, profile)
        status, failure_reasons = _fold_status(test_metrics, min_test_events)
        fatal_train_blockers = _fatal_train_blockers(selected, min_test_events)
        if fatal_train_blockers:
            status = "fail"
            failure_reasons = [f"train_{blocker}" for blocker in fatal_train_blockers] + failure_reasons

        folds.append(
            {
                "fold_id": fold_id,
                "fold_mode": "event_window",
                "selection_mode": "fixed_profile",
                "train_event_indexes": [train_start, train_end - 1],
                "test_event_indexes": [test_start, test_end - 1],
                "train_timestamp_range": [train_start_ts.isoformat(), train_end_ts.isoformat()],
                "test_timestamp_range": [test_start_ts.isoformat(), test_end_ts.isoformat()],
                "fold_recent_start_year": fold_recent_start_year,
                "status": status,
                "failure_reasons": failure_reasons,
                "selected_profile": fixed_profile_name,
                "train_verdict": selected.get("verdict"),
                "train_blockers": selected.get("blockers") or [],
                "train_warnings": selected.get("warnings") or [],
                "train_metrics": selected.get("metrics") or {},
                "test_metrics": test_metrics,
                "train_event_count": len(train_events),
                "test_event_count": len(test_events),
            }
        )
    return folds


def summarize_walk_forward(folds: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [fold for fold in folds if fold.get("status") in {"pass", "fail"}]
    passed = [fold for fold in evaluated if fold.get("status") == "pass"]
    test_expectancies = [float((fold.get("test_metrics") or {}).get("expectancy_net_r") or 0.0) for fold in evaluated]
    test_event_counts = [int((fold.get("test_metrics") or {}).get("event_count") or 0) for fold in evaluated]
    folds_with_test_signals = [
        fold for fold in evaluated if int((fold.get("test_metrics") or {}).get("event_count") or 0) > 0
    ]
    folds_meeting_test_sample = [
        fold for fold in evaluated if "test_sample_below_minimum" not in (fold.get("failure_reasons") or [])
    ]
    no_signal_folds = [
        fold for fold in evaluated if int((fold.get("test_metrics") or {}).get("event_count") or 0) == 0
    ]
    test_sample_gap_folds = [
        fold for fold in evaluated if "test_sample_below_minimum" in (fold.get("failure_reasons") or [])
    ]
    loss_folds = [
        fold
        for fold in folds_with_test_signals
        if float((fold.get("test_metrics") or {}).get("expectancy_net_r") or 0.0) <= 0.0
    ]
    train_blocked_folds = [
        fold
        for fold in evaluated
        if any(str(reason).startswith("train_") for reason in (fold.get("failure_reasons") or []))
    ]
    positive_signal_folds = [
        fold
        for fold in folds_with_test_signals
        if float((fold.get("test_metrics") or {}).get("expectancy_net_r") or 0.0) > 0.0
    ]
    selected_profiles = Counter(str(fold.get("selected_profile")) for fold in evaluated if fold.get("selected_profile"))
    selection_modes = Counter(str(fold.get("selection_mode")) for fold in folds if fold.get("selection_mode"))
    failure_reasons = Counter(reason for fold in folds for reason in (fold.get("failure_reasons") or []))
    avg_test_expectancy = _safe_div(sum(test_expectancies), len(test_expectancies))
    worst_test_expectancy = min(test_expectancies) if test_expectancies else 0.0
    positive_fold_rate = _safe_div(len(passed), len(evaluated))
    if len(evaluated) >= 3 and positive_fold_rate >= 0.6 and avg_test_expectancy >= 0.10 and worst_test_expectancy >= -0.75:
        verdict = "forward_shadow_candidate"
    elif len(evaluated) >= 2 and avg_test_expectancy > 0 and positive_fold_rate >= 0.5:
        verdict = "needs_more_sample"
    else:
        verdict = "not_ready"
    return {
        "fold_count": len(folds),
        "evaluated_folds": len(evaluated),
        "passed_folds": len(passed),
        "positive_fold_rate": positive_fold_rate,
        "coverage_diagnostics": {
            "folds_with_test_signals": len(folds_with_test_signals),
            "coverage_rate": _safe_div(len(folds_with_test_signals), len(evaluated)),
            "folds_meeting_test_sample": len(folds_meeting_test_sample),
            "min_sample_coverage_rate": _safe_div(len(folds_meeting_test_sample), len(evaluated)),
            "no_signal_folds": len(no_signal_folds),
            "test_sample_gap_folds": len(test_sample_gap_folds),
            "loss_folds": len(loss_folds),
            "train_blocked_folds": len(train_blocked_folds),
            "positive_signal_fold_rate": _safe_div(len(positive_signal_folds), len(folds_with_test_signals)),
            "avg_test_events_per_fold": _safe_div(sum(test_event_counts), len(test_event_counts)),
            "coverage_gap_fold_ids": [
                str(fold.get("fold_id")) for fold in evaluated if fold in no_signal_folds or fold in test_sample_gap_folds
            ],
            "loss_fold_ids": [str(fold.get("fold_id")) for fold in loss_folds],
        },
        "avg_test_expectancy_net_r": avg_test_expectancy,
        "worst_test_expectancy_net_r": _round(worst_test_expectancy),
        "selected_profile_distribution": dict(sorted(selected_profiles.items())),
        "selection_mode_distribution": dict(sorted(selection_modes.items())),
        "failure_reason_distribution": dict(sorted(failure_reasons.items())),
        "verdict": verdict,
    }


def run_walk_forward(request: WalkForwardRequest) -> dict[str, Any]:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = request.events_path or latest_expectancy_events_path()
    events = load_expectancy_events(events_path)
    if request.fold_mode == "event_window":
        if not request.fixed_profile_name:
            raise ValueError("Event-window walk-forward requires --fixed-profile")
        folds = build_event_window_folds(
            events,
            fixed_profile_name=request.fixed_profile_name,
            train_event_window=request.train_event_window,
            test_event_window=request.test_event_window,
            step_events=request.step_events,
            min_test_events=request.min_test_events,
        )
    elif request.fold_mode == "calendar":
        folds = build_walk_forward_folds(
            events,
            train_years=request.train_years,
            test_years=request.test_years,
            min_train_events=request.min_train_events,
            min_test_events=request.min_test_events,
            recent_start_year=request.recent_start_year,
            fixed_profile_name=request.fixed_profile_name,
        )
    else:
        raise ValueError(f"Unsupported walk-forward fold_mode: {request.fold_mode}")
    summary = summarize_walk_forward(folds)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_events_path": str(events_path),
        "request": {
            "fold_mode": request.fold_mode,
            "train_years": request.train_years,
            "test_years": request.test_years,
            "train_event_window": request.train_event_window,
            "test_event_window": request.test_event_window,
            "step_events": request.step_events,
            "min_train_events": request.min_train_events,
            "min_test_events": request.min_test_events,
            "recent_start_year": request.recent_start_year,
            "fixed_profile_name": request.fixed_profile_name,
        },
        "summary": summary,
        "folds": folds,
        "safety": {
            "read_only_walk_forward": True,
            "does_not_generate_orders": True,
            "does_not_modify_live_strategy": True,
            "does_not_touch_exchange": True,
        },
    }
    tag = _utc_tag()
    json_path = request.output_dir / f"intraday_livermore_walk_forward_{tag}.json"
    md_path = request.output_dir / f"intraday_livermore_walk_forward_{tag}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    report["summary_json_path"] = str(json_path)
    report["summary_md_path"] = str(md_path)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Intraday Livermore Walk-Forward Validation",
        "",
        "只读样本外验证：用前段历史选择 profile，再看后段未知样本表现；不下单、不接交易所、不修改实时策略。",
        "",
        "## Summary",
        f"- verdict：{summary['verdict']}",
        f"- folds：{summary['evaluated_folds']} / {summary['fold_count']}",
        f"- positive_fold_rate：{summary['positive_fold_rate']}",
        f"- coverage_diagnostics：{summary.get('coverage_diagnostics', {})}",
        f"- avg_test_expectancy_net_r：{summary['avg_test_expectancy_net_r']}",
        f"- worst_test_expectancy_net_r：{summary['worst_test_expectancy_net_r']}",
        f"- selected_profile_distribution：{summary['selected_profile_distribution']}",
        f"- selection_mode_distribution：{summary.get('selection_mode_distribution', {})}",
        f"- failure_reason_distribution：{summary['failure_reason_distribution']}",
        "",
        "## Fold Details",
    ]
    for fold in report["folds"]:
        test = fold.get("test_metrics") or {}
        lines.extend(
            [
                f"### {fold['fold_id']}",
                f"- status：{fold.get('status')}",
                f"- selected_profile：{fold.get('selected_profile')}",
                f"- train_verdict：{fold.get('train_verdict')}",
                f"- test_events：{test.get('event_count', fold.get('test_event_count'))}",
                f"- test_expectancy_net_r：{test.get('expectancy_net_r')}",
                f"- test_profit_factor：{test.get('profit_factor')}",
                f"- failure_reasons：{fold.get('failure_reasons')}",
                "",
            ]
        )
    return "\n".join(lines)
