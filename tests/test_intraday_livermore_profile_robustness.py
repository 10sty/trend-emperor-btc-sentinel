from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from intraday_livermore_profile_robustness import (  # noqa: E402
    ProfileRobustnessRequest,
    evaluate_profile_robustness,
    run_profile_robustness,
)


def event(timestamp: str, *, r1: float, r3: float) -> dict:
    return {
        "timestamp_utc": pd.Timestamp(timestamp),
        "timeframe": "1h",
        "setup_type": "intraday_pullback_restart",
        "direction": "long",
        "setup_score": 0.9,
        "volume_ratio": 0.8,
        "risk_pct": 0.02,
        "ma_spread_pct": 0.02,
        "clock_slope_type": "stable_up",
        "market_regime": "bull_trend",
        "mfe_r": 2.5,
        "mae_r": 0.8,
        "target_results": {
            "1.0": {"gross_r": r1 + 0.1, "cost_r": 0.1, "net_r": r1, "label": "win" if r1 > 0 else "loss", "bars_held": 2},
            "3.0": {"gross_r": r3 + 0.1, "cost_r": 0.1, "net_r": r3, "label": "win" if r3 > 0 else "loss", "bars_held": 4},
        },
    }


class IntradayLivermoreProfileRobustnessTest(unittest.TestCase):
    def test_evaluate_profile_robustness_computes_targets_and_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            rows = [
                event("2024-01-01", r1=1.0, r3=3.0),
                event("2024-01-02", r1=-1.0, r3=-1.0),
                event("2024-02-01", r1=1.0, r3=-1.0),
            ]
            events_path.write_text("\n".join(json.dumps(row, default=str) for row in rows) + "\n", encoding="utf-8")

            result = evaluate_profile_robustness(
                ProfileRobustnessRequest(
                    events_path=events_path,
                    profile_name="pullback_restart_1h_long_volume_0_5_clock_risk_1pct_3r",
                    target_multiples=(1.0, 3.0),
                )
            )

            self.assertEqual(result["event_count"], 3)
            self.assertEqual(result["monthly_frequency"]["active_months"], 2)
            self.assertEqual(result["target_metrics"]["1.0"]["max_loss_streak"], 1)
            self.assertEqual(result["target_metrics"]["3.0"]["label_distribution"], {"loss": 2, "win": 1})

    def test_run_profile_robustness_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            events_path = tmp_path / "events.jsonl"
            events_path.write_text(
                json.dumps(event("2024-01-01", r1=1.0, r3=3.0), default=str) + "\n",
                encoding="utf-8",
            )

            result = run_profile_robustness(
                ProfileRobustnessRequest(
                    events_path=events_path,
                    profile_name="pullback_restart_1h_long_volume_0_5_clock_risk_1pct_3r",
                    output_dir=tmp_path,
                )
            )

            self.assertTrue(Path(result["summary_json_path"]).exists())
            self.assertTrue(Path(result["summary_md_path"]).exists())


if __name__ == "__main__":
    unittest.main()
