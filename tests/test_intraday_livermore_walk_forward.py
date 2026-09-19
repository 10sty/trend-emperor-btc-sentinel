from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from intraday_livermore_walk_forward import (  # noqa: E402
    WalkForwardRequest,
    build_event_window_folds,
    build_walk_forward_folds,
    _fold_recent_start_year,
    _fold_status,
    r_metrics_for_profile,
    run_walk_forward,
    summarize_walk_forward,
)
from intraday_livermore_profile_optimizer import LivermoreProfile  # noqa: E402


def event(
    timestamp: str,
    *,
    direction: str = "long",
    setup_type: str = "intraday_breakout_followthrough",
    clock_slope_type: str = "stable_up",
    volume_ratio: float = 1.2,
    risk_pct: float = 0.02,
    r3: float = 1.0,
) -> dict:
    return {
        "timestamp_utc": pd.Timestamp(timestamp),
        "timeframe": "1h",
        "setup_type": setup_type,
        "direction": direction,
        "setup_score": 0.95,
        "volume_ratio": volume_ratio,
        "risk_pct": risk_pct,
        "ma_spread_pct": 0.02,
        "clock_slope_type": clock_slope_type,
        "market_regime": "bull_trend",
        "target_results": {
            "3.0": {"gross_r": r3 + 0.1, "cost_r": 0.1, "net_r": r3},
        },
    }


class IntradayLivermoreWalkForwardTest(unittest.TestCase):
    def test_fold_recent_start_year_uses_last_train_year(self) -> None:
        self.assertEqual(_fold_recent_start_year(2012, 2014), 2013)
        self.assertEqual(_fold_recent_start_year(2020, 2022), 2021)

    def test_r_metrics_for_profile_uses_matching_events_only(self) -> None:
        profile = LivermoreProfile(name="long_only", directions=("long",), default_target=3.0)
        metrics = r_metrics_for_profile(
            [
                event("2021-01-01", direction="long", r3=1.0),
                event("2021-01-02", direction="short", r3=-1.0),
                event("2021-01-03", direction="long", r3=-1.0),
            ],
            profile,
        )

        self.assertEqual(metrics["event_count"], 2)
        self.assertEqual(metrics["expectancy_net_r"], 0.0)
        self.assertEqual(metrics["by_direction"], {"long": 2})

    def test_fold_status_treats_no_loss_profit_factor_as_valid(self) -> None:
        status, reasons = _fold_status(
            {"event_count": 3, "expectancy_net_r": 0.8, "profit_factor": None, "win_rate": 1.0},
            min_test_events=3,
        )

        self.assertEqual(status, "pass")
        self.assertEqual(reasons, [])

    def test_build_walk_forward_folds_keeps_test_after_train(self) -> None:
        events = []
        for year in range(2020, 2024):
            for index in range(8):
                events.append(event(f"{year}-01-{index + 1:02d}", r3=1.0 if year < 2023 else -1.0))

        folds = build_walk_forward_folds(
            events,
            train_years=2,
            test_years=1,
            min_train_events=5,
            min_test_events=3,
            recent_start_year=2021,
        )

        self.assertEqual(folds[0]["train_years"], [2020, 2021])
        self.assertEqual(folds[0]["test_years"], [2022, 2022])
        self.assertEqual(folds[0]["fold_recent_start_year"], 2021)
        self.assertEqual(folds[1]["train_years"], [2021, 2022])
        self.assertEqual(folds[1]["test_years"], [2023, 2023])
        self.assertEqual(folds[1]["fold_recent_start_year"], 2022)
        self.assertTrue(all(fold["status"] in {"pass", "fail"} for fold in folds))

    def test_fixed_profile_mode_does_not_roll_profiles(self) -> None:
        events = []
        for year in range(2020, 2024):
            for index in range(8):
                events.append(event(f"{year}-01-{index + 1:02d}", r3=1.0))

        folds = build_walk_forward_folds(
            events,
            train_years=2,
            test_years=1,
            min_train_events=5,
            min_test_events=3,
            recent_start_year=2021,
            fixed_profile_name="breakout_1h_long_volume_1_3_clock_3r",
        )

        self.assertTrue(folds)
        self.assertTrue(all(fold["selection_mode"] == "fixed_profile" for fold in folds))
        self.assertTrue(
            all(fold["selected_profile"] == "breakout_1h_long_volume_1_3_clock_3r" for fold in folds)
        )

    def test_event_window_folds_use_contiguous_matching_events(self) -> None:
        events = []
        for index in range(30):
            events.append(event(f"2020-01-{index + 1:02d}", volume_ratio=1.3, r3=1.0))

        folds = build_event_window_folds(
            events,
            fixed_profile_name="breakout_1h_long_volume_1_3_clock_3r",
            train_event_window=10,
            test_event_window=5,
            step_events=5,
            min_test_events=5,
        )

        self.assertEqual(len(folds), 4)
        self.assertEqual(folds[0]["fold_mode"], "event_window")
        self.assertEqual(folds[0]["train_event_indexes"], [0, 9])
        self.assertEqual(folds[0]["test_event_indexes"], [10, 14])
        self.assertTrue(all(fold["selection_mode"] == "fixed_profile" for fold in folds))

    def test_summarize_walk_forward_reports_positive_fold_rate_and_failures(self) -> None:
        summary = summarize_walk_forward(
            [
                {
                    "status": "pass",
                    "selection_mode": "fixed_profile",
                    "selected_profile": "a",
                    "test_metrics": {"expectancy_net_r": 0.2},
                },
                {
                    "status": "fail",
                    "selection_mode": "fixed_profile",
                    "selected_profile": "a",
                    "test_metrics": {"expectancy_net_r": -0.5},
                    "failure_reasons": ["x"],
                },
                {"status": "skipped", "failure_reasons": ["sample"]},
            ]
        )

        self.assertEqual(summary["evaluated_folds"], 2)
        self.assertEqual(summary["positive_fold_rate"], 0.5)
        self.assertEqual(summary["selection_mode_distribution"], {"fixed_profile": 2})
        self.assertEqual(summary["failure_reason_distribution"], {"sample": 1, "x": 1})

    def test_summarize_walk_forward_reports_coverage_diagnostics(self) -> None:
        summary = summarize_walk_forward(
            [
                {
                    "fold_id": "no_signal",
                    "status": "fail",
                    "test_metrics": {"event_count": 0, "expectancy_net_r": 0.0},
                    "failure_reasons": [
                        "test_sample_below_minimum",
                        "test_expectancy_not_positive",
                    ],
                },
                {
                    "fold_id": "loss",
                    "status": "fail",
                    "test_metrics": {"event_count": 3, "expectancy_net_r": -0.2},
                    "failure_reasons": ["test_expectancy_not_positive"],
                },
                {
                    "fold_id": "win",
                    "status": "pass",
                    "test_metrics": {"event_count": 6, "expectancy_net_r": 0.4},
                    "failure_reasons": [],
                },
                {
                    "fold_id": "train_block",
                    "status": "fail",
                    "test_metrics": {"event_count": 6, "expectancy_net_r": 0.4},
                    "failure_reasons": ["train_recent_expectancy_not_positive"],
                },
            ]
        )

        coverage = summary["coverage_diagnostics"]
        self.assertEqual(coverage["folds_with_test_signals"], 3)
        self.assertEqual(coverage["folds_meeting_test_sample"], 3)
        self.assertEqual(coverage["no_signal_folds"], 1)
        self.assertEqual(coverage["test_sample_gap_folds"], 1)
        self.assertEqual(coverage["loss_folds"], 1)
        self.assertEqual(coverage["train_blocked_folds"], 1)
        self.assertEqual(coverage["coverage_gap_fold_ids"], ["no_signal"])
        self.assertEqual(coverage["loss_fold_ids"], ["loss"])

    def test_run_walk_forward_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            events_path = tmp_path / "events.jsonl"
            rows = []
            for year in range(2020, 2024):
                for index in range(8):
                    rows.append(event(f"{year}-01-{index + 1:02d}", r3=1.0))
            events_path.write_text(
                "\n".join(json.dumps(row, default=str) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = run_walk_forward(
                WalkForwardRequest(
                    events_path=events_path,
                    output_dir=tmp_path,
                    fold_mode="calendar",
                    train_years=2,
                    test_years=1,
                    min_train_events=5,
                    min_test_events=3,
                    fixed_profile_name="breakout_1h_long_volume_1_3_clock_3r",
                )
            )

            self.assertTrue(Path(report["summary_json_path"]).exists())
            self.assertTrue(Path(report["summary_md_path"]).exists())
            self.assertIn("summary", report)
            self.assertEqual(report["request"]["fixed_profile_name"], "breakout_1h_long_volume_1_3_clock_3r")

    def test_run_event_window_requires_fixed_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            events_path = tmp_path / "events.jsonl"
            rows = [event(f"2020-01-{index + 1:02d}", r3=1.0) for index in range(20)]
            events_path.write_text(
                "\n".join(json.dumps(row, default=str) for row in rows) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                run_walk_forward(
                    WalkForwardRequest(
                        events_path=events_path,
                        output_dir=tmp_path,
                        fold_mode="event_window",
                        train_event_window=10,
                        test_event_window=5,
                    )
                )


if __name__ == "__main__":
    unittest.main()
