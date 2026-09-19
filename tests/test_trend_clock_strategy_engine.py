from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from trend_clock_strategy_engine import (  # noqa: E402
    add_trend_clock_features,
    anchor_direction,
    classify_clock_slope,
    detect_opportunity,
    evaluate_outcome,
    scan_aligned_anchor_trigger,
    summarize_events,
)


def frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=len(values), freq="h"),
            "open": values,
            "high": [value * 1.01 for value in values],
            "low": [value * 0.99 for value in values],
            "close": values,
            "volume": [100.0] * len(values),
        }
    )


class TrendClockStrategyEngineTest(unittest.TestCase):
    def test_classifies_clock_slope(self) -> None:
        df = add_trend_clock_features(frame([100 + i * 0.2 for i in range(160)]), "1h")
        self.assertIn(classify_clock_slope(df.iloc[-1]), {"stable_up", "accelerating_up"})

    def test_detects_stable_trend_pullback_long(self) -> None:
        values = [100 + i * 0.4 for i in range(160)]
        values[-3] = values[-4] * 0.98
        values[-2] = values[-4] * 0.985
        values[-1] = values[-4] * 1.005
        df = add_trend_clock_features(frame(values), "1h")
        row = df.iloc[-1].copy()
        row["near_ma60"] = True
        row["above_ma20_group"] = True
        row["bullish_stack"] = True
        row["ma60_slope"] = 0.01
        row["ma120_slope"] = 0.01
        row["log_slope_120"] = 0.05
        previous = df.iloc[-2].copy()
        previous["above_ma20_group"] = False
        event = detect_opportunity(row, previous, "1h")
        self.assertIsNotNone(event)
        self.assertEqual(event["setup_type"], "trend_pullback")
        self.assertEqual(event["direction"], "long")

    def test_outcome_for_long_correct(self) -> None:
        df = frame([100.0, 101.0, 103.0, 104.0])
        event = {"close": 100.0, "direction": "long", "setup_type": "trend_pullback"}
        outcome = evaluate_outcome(df, 0, event, "1d")
        self.assertEqual(outcome["outcome_label"], "correct")
        self.assertGreater(outcome["return_pct"], 0)

    def test_summary_metrics(self) -> None:
        events = [
            {
                "timestamp_utc": pd.Timestamp("2025-01-01"),
                "timeframe": "1h",
                "setup_type": "trend_pullback",
                "direction": "long",
                "future_outcome": {"outcome_label": "correct", "return_pct": 0.02},
            },
            {
                "timestamp_utc": pd.Timestamp("2025-01-02"),
                "timeframe": "4h",
                "setup_type": "false_break_reversal",
                "direction": "short",
                "future_outcome": {"outcome_label": "incorrect", "return_pct": -0.01},
            },
        ]
        summary = summarize_events(events)
        self.assertEqual(summary["total_events"], 2)
        self.assertEqual(summary["by_direction"], {"long": 1, "short": 1})
        self.assertIn("trend_pullback", summary["type_quality"])

    def test_anchor_direction_detects_higher_timeframe_bias(self) -> None:
        df = add_trend_clock_features(frame([100 + i * 0.3 for i in range(180)]), "4h")
        row = df.iloc[-1].copy()
        row["bullish_stack"] = True
        row["ma60_slope"] = 0.01
        row["ma120_slope"] = 0.01
        row["log_slope_120"] = 0.04
        self.assertEqual(anchor_direction(row), "long")

    def test_aligned_anchor_trigger_filters_by_higher_timeframe(self) -> None:
        values = [100.0] * 160
        values += [100 + i * 0.5 for i in range(80)]
        frames = {
            "1h": frame(values),
            "4h": frame([100 + i * 0.3 for i in range(240)]),
            "1d": frame([100 + i * 0.2 for i in range(240)]),
        }
        events = scan_aligned_anchor_trigger(frames, sample=5)
        self.assertIsInstance(events, list)
        for event in events:
            self.assertGreaterEqual(event["anchor_support_count"], 1)
            self.assertEqual(event["anchor_opposition_count"], 0)


if __name__ == "__main__":
    unittest.main()
