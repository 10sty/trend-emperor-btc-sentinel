from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from intraday_livermore_profile_optimizer import (  # noqa: E402
    LivermoreProfile,
    cost_stress_metrics,
    default_profiles,
    era_metrics,
    net_r_for_event,
    profile_matches,
    readiness_from_metrics,
    select_profile,
    summarize_profile,
    target_robustness_metrics,
)


def event(
    *,
    timeframe: str = "1h",
    setup: str = "intraday_breakout_followthrough",
    direction: str = "long",
    score: float = 0.98,
    volume_ratio: float = 1.5,
    risk_pct: float = 0.02,
    ma_spread_pct: float = 0.02,
    clock_slope_type: str = "stable_up",
    proof_strength_score: float = 0.85,
    market_proof_stage: str = "reaction_revalidated",
    reaction_test_present: bool = True,
    pyramid_eligible_shadow: bool = True,
    timestamp: str = "2021-01-01 00:00:00",
    r3: float = 1.0,
    r2: float = 0.5,
) -> dict:
    return {
        "timestamp_utc": pd.Timestamp(timestamp),
        "timeframe": timeframe,
        "setup_type": setup,
        "direction": direction,
        "setup_score": score,
        "volume_ratio": volume_ratio,
        "risk_pct": risk_pct,
        "ma_spread_pct": ma_spread_pct,
        "clock_slope_type": clock_slope_type,
        "proof_strength_score": proof_strength_score,
        "market_proof_stage": market_proof_stage,
        "reaction_test_present": reaction_test_present,
        "pyramid_eligible_shadow": pyramid_eligible_shadow,
        "target_results": {
            "2.0": {"gross_r": r2 + 0.1, "cost_r": 0.1, "net_r": r2},
            "3.0": {"gross_r": r3 + 0.1, "cost_r": 0.1, "net_r": r3},
        },
    }


class IntradayLivermoreProfileOptimizerTest(unittest.TestCase):
    def test_profile_filter_matches_setup_direction_and_score(self) -> None:
        profile = LivermoreProfile(
            name="x",
            setup_types=("intraday_breakout_followthrough",),
            directions=("long",),
            min_setup_score=0.9,
        )

        self.assertTrue(profile_matches(event(), profile))
        self.assertFalse(profile_matches(event(direction="short"), profile))
        self.assertFalse(profile_matches(event(score=0.8), profile))

    def test_profile_filter_matches_timeframe_when_specified(self) -> None:
        profile = LivermoreProfile(name="only_1h", timeframes=("1h",))
        unrestricted = LivermoreProfile(name="any_timeframe")

        self.assertTrue(profile_matches(event(timeframe="1h"), profile))
        self.assertFalse(profile_matches(event(timeframe="15m"), profile))
        self.assertTrue(profile_matches(event(timeframe="15m"), unrestricted))

    def test_institutional_profile_requires_volume_clock_score_and_tight_risk(self) -> None:
        profile = next(
            item
            for item in default_profiles()
            if item.name == "institutional_long_volume_1_3_up_clock_risk_2_5pct_score_0_90_3r"
        )

        self.assertTrue(profile_matches(event(timeframe="1h", score=0.95, volume_ratio=1.4, risk_pct=0.02), profile))
        self.assertTrue(profile_matches(event(timeframe="15m", score=0.95, volume_ratio=1.4, risk_pct=0.02), profile))
        self.assertFalse(profile_matches(event(direction="short", score=0.95, volume_ratio=1.4, risk_pct=0.02), profile))
        self.assertFalse(profile_matches(event(score=0.86, volume_ratio=1.4, risk_pct=0.02), profile))
        self.assertFalse(profile_matches(event(score=0.95, volume_ratio=1.2, risk_pct=0.02), profile))
        self.assertFalse(profile_matches(event(score=0.95, volume_ratio=1.4, risk_pct=0.03), profile))
        self.assertFalse(
            profile_matches(event(score=0.95, volume_ratio=1.4, risk_pct=0.02, clock_slope_type="sideways"), profile)
        )

    def test_institutional_stable_profile_excludes_acceleration_chase(self) -> None:
        profile = next(
            item
            for item in default_profiles()
            if item.name == "institutional_long_volume_1_3_stable_clock_risk_2_5pct_score_0_90_3r"
        )

        self.assertTrue(
            profile_matches(event(timeframe="15m", score=0.95, volume_ratio=1.4, risk_pct=0.02), profile)
        )
        self.assertFalse(
            profile_matches(
                event(timeframe="15m", score=0.95, volume_ratio=1.4, risk_pct=0.02, clock_slope_type="accelerating_up"),
                profile,
            )
        )

    def test_cross_source_profile_requires_score_risk_ma_and_up_clock(self) -> None:
        profile = next(
            item
            for item in default_profiles()
            if item.name == "institutional_cross_source_score_0_90_risk_1_2_3_5pct_ma_2_5pct_up_clock_3r"
        )

        self.assertTrue(profile_matches(event(timeframe="1h", score=0.95, risk_pct=0.02, ma_spread_pct=0.02), profile))
        self.assertFalse(profile_matches(event(timeframe="4h", score=0.95, risk_pct=0.02, ma_spread_pct=0.02), profile))
        self.assertFalse(profile_matches(event(direction="short", score=0.95, risk_pct=0.02, ma_spread_pct=0.02), profile))
        self.assertFalse(profile_matches(event(score=0.85, risk_pct=0.02, ma_spread_pct=0.02), profile))
        self.assertFalse(profile_matches(event(score=0.95, risk_pct=0.01, ma_spread_pct=0.02), profile))
        self.assertFalse(profile_matches(event(score=0.95, risk_pct=0.04, ma_spread_pct=0.02), profile))
        self.assertFalse(profile_matches(event(score=0.95, risk_pct=0.02, ma_spread_pct=0.03), profile))
        self.assertFalse(
            profile_matches(event(score=0.95, risk_pct=0.02, ma_spread_pct=0.02, clock_slope_type="sideways"), profile)
        )

    def test_cross_source_sample_balanced_profile_does_not_require_clock(self) -> None:
        profile = next(
            item
            for item in default_profiles()
            if item.name == "institutional_cross_source_score_0_90_volume_0_5_risk_1_5_3_5pct_ma_2_5pct_3r"
        )

        self.assertTrue(
            profile_matches(
                event(score=0.95, volume_ratio=0.6, risk_pct=0.02, ma_spread_pct=0.02, clock_slope_type="sideways"),
                profile,
            )
        )
        self.assertFalse(profile_matches(event(score=0.95, volume_ratio=0.4, risk_pct=0.02, ma_spread_pct=0.02), profile))
        self.assertFalse(profile_matches(event(score=0.95, volume_ratio=0.6, risk_pct=0.012, ma_spread_pct=0.02), profile))

    def test_livermore_proof_chain_profile_requires_revalidated_reaction(self) -> None:
        profile = next(
            item
            for item in default_profiles()
            if item.name == "livermore_proof_chain_1h_revalidated_long_3r"
        )

        self.assertTrue(
            profile_matches(
                event(
                    timeframe="1h",
                    setup="intraday_pullback_restart",
                    direction="long",
                    risk_pct=0.02,
                    proof_strength_score=0.85,
                    market_proof_stage="reaction_revalidated",
                    reaction_test_present=True,
                    pyramid_eligible_shadow=True,
                ),
                profile,
            )
        )
        self.assertFalse(
            profile_matches(
                event(
                    timeframe="1h",
                    setup="intraday_breakout_followthrough",
                    risk_pct=0.02,
                    market_proof_stage="initial_breakout_probe",
                    reaction_test_present=False,
                    pyramid_eligible_shadow=False,
                ),
                profile,
            )
        )
        self.assertFalse(
            profile_matches(
                event(
                    timeframe="1h",
                    setup="intraday_pullback_restart",
                    risk_pct=0.02,
                    proof_strength_score=0.70,
                ),
                profile,
            )
        )

    def test_livermore_cross_source_profile_requires_proof_and_compact_clock(self) -> None:
        profile = next(
            item
            for item in default_profiles()
            if item.name == "livermore_cross_source_proof_1h_long_3r"
        )

        self.assertTrue(
            profile_matches(
                event(
                    timeframe="1h",
                    direction="long",
                    volume_ratio=0.6,
                    risk_pct=0.02,
                    ma_spread_pct=0.02,
                    proof_strength_score=0.86,
                    market_proof_stage="reaction_revalidated",
                    clock_slope_type="stable_up",
                ),
                profile,
            )
        )
        self.assertFalse(profile_matches(event(timeframe="15m", proof_strength_score=0.9), profile))
        self.assertFalse(profile_matches(event(direction="short", proof_strength_score=0.9), profile))
        self.assertFalse(profile_matches(event(volume_ratio=0.4, proof_strength_score=0.9), profile))
        self.assertFalse(profile_matches(event(risk_pct=0.006, proof_strength_score=0.9), profile))
        self.assertFalse(profile_matches(event(ma_spread_pct=0.03, proof_strength_score=0.9), profile))
        self.assertFalse(profile_matches(event(proof_strength_score=0.8), profile))
        self.assertFalse(
            profile_matches(
                event(proof_strength_score=0.9, market_proof_stage="unknown"),
                profile,
            )
        )
        self.assertFalse(
            profile_matches(
                event(proof_strength_score=0.9, clock_slope_type="sideways"),
                profile,
            )
        )

    def test_livermore_reaction_revalidated_profile_excludes_initial_breakout(self) -> None:
        profile = next(
            item
            for item in default_profiles()
            if item.name == "livermore_reaction_revalidated_1h_long_3r"
        )

        self.assertTrue(
            profile_matches(
                event(
                    timeframe="1h",
                    setup="intraday_pullback_restart",
                    direction="long",
                    volume_ratio=0.7,
                    risk_pct=0.02,
                    ma_spread_pct=0.015,
                    proof_strength_score=1.0,
                    market_proof_stage="reaction_revalidated",
                    reaction_test_present=True,
                    pyramid_eligible_shadow=True,
                    clock_slope_type="stable_up",
                ),
                profile,
            )
        )
        self.assertFalse(
            profile_matches(
                event(
                    timeframe="1h",
                    setup="intraday_breakout_followthrough",
                    direction="long",
                    volume_ratio=2.0,
                    risk_pct=0.02,
                    ma_spread_pct=0.015,
                    proof_strength_score=1.0,
                    market_proof_stage="initial_breakout_probe",
                    reaction_test_present=False,
                    pyramid_eligible_shadow=False,
                    clock_slope_type="stable_up",
                ),
                profile,
            )
        )

    def test_profile_filter_matches_volume_risk_ma_spread_and_clock(self) -> None:
        profile = LivermoreProfile(
            name="institutional_breakout",
            min_volume_ratio=1.3,
            min_risk_pct=0.01,
            max_risk_pct=0.035,
            max_ma_spread_pct=0.04,
            clock_slope_types=("stable_up", "accelerating_up"),
        )

        self.assertTrue(profile_matches(event(), profile))
        self.assertFalse(profile_matches(event(volume_ratio=1.2), profile))
        self.assertFalse(profile_matches(event(risk_pct=0.005), profile))
        self.assertFalse(profile_matches(event(risk_pct=0.04), profile))
        self.assertFalse(profile_matches(event(ma_spread_pct=0.05), profile))
        self.assertFalse(profile_matches(event(clock_slope_type="sideways"), profile))

    def test_net_r_can_use_direction_specific_target(self) -> None:
        profile = LivermoreProfile(name="x", target_by_direction={"short": 2.0}, default_target=3.0)

        self.assertEqual(net_r_for_event(event(direction="long", r3=1.25), profile), 1.25)
        self.assertEqual(net_r_for_event(event(direction="short", r2=0.75), profile), 0.75)

    def test_cost_stress_uses_gross_r_and_cost_multiplier(self) -> None:
        profile = LivermoreProfile(name="x")
        item = event(r3=1.0)

        self.assertAlmostEqual(net_r_for_event(item, profile, cost_multiplier=2.0), 0.9)
        stress = cost_stress_metrics([item], profile)

        self.assertAlmostEqual(stress["2x_cost"]["expectancy_net_r"], 0.9)
        self.assertAlmostEqual(stress["3x_cost"]["expectancy_net_r"], 0.8)

    def test_target_robustness_reports_each_target_multiple(self) -> None:
        profile = LivermoreProfile(name="x")
        metrics = target_robustness_metrics(
            [
                event(r2=1.0, r3=-1.0),
                event(r2=-1.0, r3=2.0),
            ],
            profile,
        )

        self.assertIn("1r", metrics)
        self.assertIn("1.5r", metrics)
        self.assertIn("2r", metrics)
        self.assertIn("3r", metrics)
        self.assertEqual(metrics["2r"]["event_count"], 2)
        self.assertEqual(metrics["2r"]["expectancy_net_r"], 0.0)
        self.assertEqual(metrics["3r"]["expectancy_net_r"], 0.5)

    def test_era_metrics_split_early_middle_and_recent_periods(self) -> None:
        profile = LivermoreProfile(name="x")
        metrics = era_metrics(
            [
                event(timestamp="2017-01-01", r3=1.0),
                event(timestamp="2019-01-01", r3=0.5),
                event(timestamp="2022-01-01", r3=-1.0),
            ],
            profile,
        )

        self.assertEqual(metrics["2012_2017"]["event_count"], 1)
        self.assertEqual(metrics["2018_2020"]["event_count"], 1)
        self.assertEqual(metrics["2021_present"]["event_count"], 1)

    def test_summarize_profile_calculates_expectancy_drawdown_and_years(self) -> None:
        profile = LivermoreProfile(name="breakout", setup_types=("intraday_breakout_followthrough",))
        events = [
            event(timestamp="2021-01-01", r3=1.0),
            event(timestamp="2021-02-01", r3=-1.0),
            event(timestamp="2022-01-01", r3=2.0),
        ]

        result = summarize_profile(events, profile, recent_start_year=2021)

        self.assertEqual(result["metrics"]["event_count"], 3)
        self.assertAlmostEqual(result["metrics"]["expectancy_net_r"], 0.666667)
        self.assertEqual(result["metrics"]["positive_year_rate"], 0.5)
        self.assertGreaterEqual(result["metrics"]["max_drawdown_r"], 1.0)
        self.assertIn("target_robustness", result["metrics"])

    def test_readiness_blocks_small_or_weak_profiles(self) -> None:
        ready = readiness_from_metrics(
            {
                "event_count": 100,
                "expectancy_net_r": 0.25,
                "profit_factor": 1.5,
                "positive_year_rate": 0.7,
                "recent_expectancy_net_r": 0.1,
                "max_drawdown_r": 10,
                "worst_year_r": -2,
            }
        )
        weak = readiness_from_metrics(
            {
                "event_count": 10,
                "expectancy_net_r": 0.1,
                "profit_factor": 1.1,
                "positive_year_rate": 0.2,
                "recent_expectancy_net_r": -0.1,
                "max_drawdown_r": 10,
                "worst_year_r": -2,
            }
        )

        self.assertEqual(ready["verdict"], "live_pilot_candidate")
        self.assertEqual(weak["verdict"], "not_ready")
        self.assertIn("sample_below_80", weak["blockers"])

    def test_select_profile_prefers_ready_candidate(self) -> None:
        results = [
            {
                "profile": {"name": "weak"},
                "metrics": {
                    "event_count": 10,
                    "expectancy_net_r": 0.5,
                    "recent_expectancy_net_r": 0.5,
                    "profit_factor": 2,
                    "positive_year_rate": 1,
                    "worst_year_r": 1,
                    "max_drawdown_r": 1,
                },
                "readiness": {"verdict": "not_ready", "blockers": ["sample_below_80"], "warnings": []},
            },
            {
                "profile": {"name": "ready"},
                "metrics": {
                    "event_count": 100,
                    "expectancy_net_r": 0.25,
                    "recent_expectancy_net_r": 0.1,
                    "profit_factor": 1.5,
                    "positive_year_rate": 0.7,
                    "worst_year_r": -2,
                    "max_drawdown_r": 10,
                },
                "readiness": {"verdict": "live_pilot_candidate", "blockers": [], "warnings": []},
            },
        ]

        selected = select_profile(results)

        self.assertEqual(selected["selected_profile"], "ready")
        self.assertEqual(selected["verdict"], "live_pilot_candidate")

    def test_select_profile_prefers_recent_and_yearly_robustness(self) -> None:
        results = [
            {
                "profile": {"name": "higher_average"},
                "metrics": {
                    "event_count": 100,
                    "expectancy_net_r": 0.58,
                    "recent_expectancy_net_r": 0.20,
                    "profit_factor": 3.3,
                    "positive_year_rate": 0.8,
                    "worst_year_r": -0.8,
                    "max_drawdown_r": 3,
                },
                "readiness": {"verdict": "live_pilot_candidate", "blockers": [], "warnings": []},
            },
            {
                "profile": {"name": "more_robust"},
                "metrics": {
                    "event_count": 100,
                    "expectancy_net_r": 0.54,
                    "recent_expectancy_net_r": 0.27,
                    "profit_factor": 3.0,
                    "positive_year_rate": 0.9,
                    "worst_year_r": -0.3,
                    "max_drawdown_r": 3,
                },
                "readiness": {"verdict": "live_pilot_candidate", "blockers": [], "warnings": []},
            },
        ]

        selected = select_profile(results)

        self.assertEqual(selected["selected_profile"], "more_robust")

    def test_select_profile_does_not_prefer_zero_event_profile(self) -> None:
        results = [
            {
                "profile": {"name": "zero_event"},
                "metrics": {
                    "event_count": 0,
                    "expectancy_net_r": 0,
                    "recent_expectancy_net_r": 0,
                    "profit_factor": None,
                    "positive_year_rate": 0,
                    "worst_year_r": 0,
                    "max_drawdown_r": 0,
                },
                "readiness": {"verdict": "not_ready", "blockers": ["sample_below_80"], "warnings": []},
            },
            {
                "profile": {"name": "small_positive"},
                "metrics": {
                    "event_count": 3,
                    "expectancy_net_r": 0.05,
                    "recent_expectancy_net_r": 0.05,
                    "profit_factor": 1.1,
                    "positive_year_rate": 0.5,
                    "worst_year_r": -0.1,
                    "max_drawdown_r": 1,
                },
                "readiness": {"verdict": "not_ready", "blockers": ["sample_below_80"], "warnings": []},
            },
        ]

        selected = select_profile(results)

        self.assertEqual(selected["selected_profile"], "small_positive")


if __name__ == "__main__":
    unittest.main()
