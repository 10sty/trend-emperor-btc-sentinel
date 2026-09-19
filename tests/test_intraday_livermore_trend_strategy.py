from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from intraday_livermore_trend_strategy import (
    detect_intraday_livermore_event,
    evaluate_rr3_outcome,
    summarize_intraday_livermore_events,
)


def _anchor(direction: str) -> pd.Series:
    if direction == "long":
        return pd.Series(
            {
                "timestamp_utc": pd.Timestamp("2025-01-01 00:00:00"),
                "ma20": 110.0,
                "ma60": 105.0,
                "ma120": 100.0,
                "bullish_stack": True,
                "bearish_stack": False,
                "ma60_slope": 0.01,
                "ma120_slope": 0.01,
                "log_slope_120": 0.04,
                "ma_spread_pct": 0.02,
            }
        )
    return pd.Series(
        {
            "timestamp_utc": pd.Timestamp("2025-01-01 00:00:00"),
            "ma20": 90.0,
            "ma60": 95.0,
            "ma120": 100.0,
            "bullish_stack": False,
            "bearish_stack": True,
            "ma60_slope": -0.01,
            "ma120_slope": -0.01,
            "log_slope_120": -0.04,
            "ma_spread_pct": 0.02,
        }
    )


class IntradayLivermoreTrendStrategyTest(unittest.TestCase):
    def test_detects_long_breakout_with_anchor_support(self) -> None:
        previous = pd.Series(
            {
                "timestamp_utc": pd.Timestamp("2025-01-01 01:00:00"),
                "close": 99.0,
                "high": 100.0,
                "low": 98.5,
                "prior_high_20": 100.0,
                "above_ma20_group": True,
            }
        )
        row = pd.Series(
            {
                "timestamp_utc": pd.Timestamp("2025-01-01 02:00:00"),
                "close": 102.0,
                "high": 103.0,
                "low": 101.0,
                "ma20": 100.2,
                "ma60": 99.5,
                "ma120": 98.0,
                "ema20": 100.1,
                "bullish_stack": True,
                "bearish_stack": False,
                "ma20_slope": 0.012,
                "ma60_slope": 0.004,
                "prior_high_20": 100.0,
                "prior_low_20": 99.5,
                "above_ma20_group": True,
                "below_ma20_group": False,
                "volume_ratio": 1.3,
                "log_slope_120": 0.03,
                "ma_spread_pct": 0.015,
            }
        )

        event = detect_intraday_livermore_event(
            row,
            previous,
            "1h",
            {"4h": _anchor("long"), "1d": _anchor("long")},
        )

        self.assertIsNotNone(event)
        self.assertEqual(event["direction"], "long")
        self.assertEqual(event["setup_type"], "intraday_breakout_followthrough")
        self.assertGreaterEqual(event["setup_score"], 0.70)
        self.assertEqual(event["anchor_support_count"], 2)
        self.assertIn("市场证明", event["livermore_principle"])
        self.assertEqual(event["market_proof_stage"], "initial_breakout_probe")
        self.assertGreaterEqual(event["proof_strength_score"], 0.70)
        self.assertFalse(event["reaction_test_present"])
        self.assertFalse(event["pyramid_eligible_shadow"])
        self.assertIn("最小阻力线", " ".join(event["livermore_sequence"]))

    def test_pullback_restart_marks_revalidated_proof_chain(self) -> None:
        previous = pd.Series(
            {
                "timestamp_utc": pd.Timestamp("2025-01-01 01:00:00"),
                "close": 100.0,
                "high": 101.0,
                "low": 99.0,
                "prior_low_20": 99.0,
                "above_ma20_group": False,
                "near_ma60": True,
                "near_ma120": False,
            }
        )
        row = pd.Series(
            {
                "timestamp_utc": pd.Timestamp("2025-01-01 02:00:00"),
                "close": 102.0,
                "high": 103.0,
                "low": 100.5,
                "ma20": 100.2,
                "ma60": 99.5,
                "ma120": 98.0,
                "ema20": 100.1,
                "bullish_stack": True,
                "bearish_stack": False,
                "ma20_slope": 0.012,
                "ma60_slope": 0.004,
                "prior_high_20": 104.0,
                "prior_low_20": 99.0,
                "above_ma20_group": True,
                "below_ma20_group": False,
                "near_ma60": True,
                "near_ma120": False,
                "volume_ratio": 0.8,
                "log_slope_120": 0.03,
                "ma_spread_pct": 0.015,
            }
        )

        event = detect_intraday_livermore_event(
            row,
            previous,
            "1h",
            {"4h": _anchor("long"), "1d": _anchor("long")},
        )

        self.assertIsNotNone(event)
        self.assertEqual(event["setup_type"], "intraday_pullback_restart")
        self.assertEqual(event["market_proof_stage"], "reaction_revalidated")
        self.assertTrue(event["reaction_test_present"])
        self.assertTrue(event["pyramid_eligible_shadow"])

    def test_rejects_anchor_opposition(self) -> None:
        previous = pd.Series(
            {
                "timestamp_utc": pd.Timestamp("2025-01-01 01:00:00"),
                "close": 99.0,
                "high": 100.0,
                "low": 98.5,
                "prior_high_20": 100.0,
                "above_ma20_group": False,
            }
        )
        row = pd.Series(
            {
                "timestamp_utc": pd.Timestamp("2025-01-01 02:00:00"),
                "close": 102.0,
                "high": 103.0,
                "low": 101.0,
                "ma20": 100.2,
                "ma60": 99.5,
                "ma120": 98.0,
                "ema20": 100.1,
                "bullish_stack": True,
                "bearish_stack": False,
                "ma20_slope": 0.012,
                "ma60_slope": 0.004,
                "prior_high_20": 100.0,
                "prior_low_20": 99.5,
                "above_ma20_group": True,
                "below_ma20_group": False,
                "volume_ratio": 1.3,
                "log_slope_120": 0.03,
                "ma_spread_pct": 0.015,
            }
        )

        event = detect_intraday_livermore_event(
            row,
            previous,
            "1h",
            {"4h": _anchor("short"), "1d": _anchor("long")},
        )

        self.assertIsNone(event)

    def test_evaluates_three_to_one_outcome(self) -> None:
        df = pd.DataFrame(
            [
                {"timestamp_utc": pd.Timestamp("2025-01-01 00:00:00"), "open": 100, "high": 101, "low": 99, "close": 100},
                {"timestamp_utc": pd.Timestamp("2025-01-01 01:00:00"), "open": 100, "high": 104, "low": 99, "close": 103},
                {"timestamp_utc": pd.Timestamp("2025-01-01 02:00:00"), "open": 103, "high": 106.5, "low": 102, "close": 106},
            ]
        )
        event = {"timeframe": "1h", "direction": "long", "entry": 100.0, "invalidation": 98.0}

        outcome = evaluate_rr3_outcome(df, 0, event)

        self.assertEqual(outcome["outcome_label"], "win_rr3")
        self.assertEqual(outcome["target"], 106.0)
        self.assertEqual(outcome["bars_to_outcome"], 2)

    def test_summary_tracks_monthly_frequency_and_quality(self) -> None:
        events = [
            {
                "timestamp_utc": pd.Timestamp("2025-01-01 00:00:00"),
                "timeframe": "1h",
                "direction": "long",
                "setup_type": "intraday_pullback_restart",
                "future_outcome": {"outcome_label": "win_rr3", "return_pct": 0.03},
            },
            {
                "timestamp_utc": pd.Timestamp("2025-01-10 00:00:00"),
                "timeframe": "15m",
                "direction": "short",
                "setup_type": "intraday_breakout_followthrough",
                "future_outcome": {"outcome_label": "loss", "return_pct": -0.01},
            },
            {
                "timestamp_utc": pd.Timestamp("2025-02-01 00:00:00"),
                "timeframe": "1h",
                "direction": "long",
                "setup_type": "intraday_pullback_restart",
                "future_outcome": {"outcome_label": "timeout", "return_pct": 0.0},
            },
        ]

        summary = summarize_intraday_livermore_events(events)

        self.assertEqual(summary["total_events"], 3)
        self.assertEqual(summary["months_with_2_to_3_events"], 1)
        self.assertEqual(summary["rr3_wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertIn("intraday_pullback_restart", summary["setup_quality"])


if __name__ == "__main__":
    unittest.main()
