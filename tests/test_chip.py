"""Deterministic software regressions, not validation of physical chip boundaries.

The default schematic starts in jetting: size changes precede downstream breakup
migration and eventually an unbroken thread. The optional ratio window and its
historical numeric defaults remain research priors, not confirmed dripping data.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from mcs_date.experiment import ChipModelConfig, ChipSimulator, FlowRegime
from mcs_date.experiment.calibration import CalibrationResult, learn_calibration


class TestChipSimulator:
    def setup_method(self) -> None:
        self.sim = ChipSimulator(noise=False)

    def test_default_mode_is_jetting_at_its_anchor(self) -> None:
        cfg = self.sim.config
        assert cfg.operating_mode == "jetting"
        features, regime, confidence = self.sim.observe(cfg.jetting_center_q1, cfg.jetting_center_q2)
        assert regime is FlowRegime.JETTING
        assert confidence > 0
        assert features.droplet_events > 0  # A jet can produce downstream droplets.
        assert features.breakup_distance_h == cfg.anchor_breakup_h
        assert features.junction_visible
        assert features.source == "simulation"

    @pytest.mark.parametrize("q1,q2,expected", [
        (50.0, 20.0, FlowRegime.THREADING),
        (70.0, 20.0, FlowRegime.JETTING),
        (90.0, 20.0, FlowRegime.JETTING),
        (100.0, 20.0, FlowRegime.JETTING),
    ])
    def test_historical_flow_points_are_not_forced_to_dripping(self, q1, q2, expected) -> None:
        # These are model predictions, not historical topology labels.
        assert expected is not FlowRegime.DRIPPING
        assert self.sim.regime_at(q1, q2) is expected
        assert self.sim.observe(q1, q2)[1] is expected

    def test_size_change_precedes_breakup_migration_then_unbroken_thread(self) -> None:
        cfg = self.sim.config
        # Along fixed Q1, this construction gives increasing log-ratio deviation.
        deviations = (0.0, 0.30, 0.70, 1.10)
        observations = [
            self.sim.observe(
                cfg.jetting_center_q1,
                cfg.jetting_center_q2 * math.exp(-rho * cfg.jetting_log_ratio_radius),
                window_s=10.0,
            )
            for rho in deviations
        ]
        anchor, early, migrating, late = [item[0] for item in observations]
        assert [item[1] for item in observations] == [
            FlowRegime.JETTING, FlowRegime.JETTING, FlowRegime.JETTING, FlowRegime.THREADING
        ]
        assert early.mean_diameter_ratio > anchor.mean_diameter_ratio * 1.05
        assert early.breakup_distance_h == anchor.breakup_distance_h == cfg.anchor_breakup_h
        assert cfg.anchor_breakup_h < migrating.breakup_distance_h < cfg.observation_extent_h
        assert all(item.droplet_events > 0 for item in (anchor, early, migrating))
        assert late.breakup_distance_h is None
        assert late.droplet_events == 0
        assert late.droplet_rate_hz == 0.0
        assert late.observation_extent_h == cfg.observation_extent_h

    @pytest.mark.parametrize("q1,q2", [(0.0, 20.0), (20.0, 0.0), (-1.0, 20.0),
                                             (math.nan, 20.0), (20.0, math.inf)])
    def test_unknown_for_invalid_flow(self, q1: float, q2: float) -> None:
        assert self.sim.regime_at(q1, q2) is FlowRegime.UNKNOWN

    def test_capillary_numbers_are_positive_and_scale_independently(self) -> None:
        ca_c, ca_d = self.sim.capillary_numbers(10.0, 10.0)
        oil_changed = self.sim.capillary_numbers(100.0, 10.0)
        water_changed = self.sim.capillary_numbers(10.0, 30.0)
        assert ca_c > 0 and ca_d > 0
        assert oil_changed == pytest.approx((10 * ca_c, ca_d))
        assert water_changed == pytest.approx((ca_c, 3 * ca_d))

    def test_thread_diameter_implements_model_square_root_law(self) -> None:
        assert self.sim.thread_diameter_ratio(80.0, 20.0) == pytest.approx(math.sqrt(0.25 / 2.0))

    def test_thread_diameter_is_clipped(self) -> None:
        assert self.sim.thread_diameter_ratio(80.0, 1000.0) == 1.0
        assert self.sim.thread_diameter_ratio(80.0, 0.0) == 0.0

    def test_candidate_size_implements_configured_affine_prior(self) -> None:
        cfg = self.sim.config
        # Lock the software formula, not the physical validity of a historical fit.
        baseline = self.sim.droplet_diameter_um(60.0, 20.0)
        assert baseline == cfg.size_intercept_um
        assert self.sim.droplet_diameter_um(70.0, 20.0) == pytest.approx(
            baseline + 10 * cfg.size_q1_coef_um_per_ulmin
        )
        assert self.sim.droplet_diameter_um(60.0, 25.0) == pytest.approx(
            baseline + 5 * cfg.size_q2_coef_um_per_ulmin
        )
        clipped = ChipSimulator(config=replace(cfg, size_intercept_um=0.5), noise=False)
        assert clipped.droplet_diameter_um(60.0, 20.0) == 1.0

    @pytest.mark.parametrize("q1,q2", [(70.0, 20.0), (90.0, 20.0), (180.0, 20.0), (40.0, 20.0)])
    def test_observed_features_agree_with_default_model(self, q1: float, q2: float) -> None:
        features, regime, confidence = self.sim.observe(q1, q2, window_s=12.0)
        assert regime is self.sim.regime_at(q1, q2)
        assert 0.0 < confidence <= 1.0
        assert features.window_s == 12.0
        assert features.sample_count == 60
        assert 0.0 <= features.stream_coverage <= 1.0
        assert features.q1 == q1 and features.q2 == q2

    def test_noiseless_observations_repeat_and_seeded_noise_is_reproducible(self) -> None:
        assert self.sim.observe(90.0, 20.0) == self.sim.observe(90.0, 20.0)
        first = ChipSimulator(seed=19)
        second = ChipSimulator(seed=19)
        points = [(90.0, 20.0), (100.0, 20.0), (180.0, 20.0)]
        first_sequence = [first.observe(*point) for point in points]
        assert first_sequence == [second.observe(*point) for point in points]
        assert first_sequence[0][0] != self.sim.observe(90.0, 20.0)[0]

    def test_calibration_import_does_not_silently_enable_ratio_window(self) -> None:
        calibration = CalibrationResult(ratio_low=2.6, ratio_high=4.0, ratio_thread=5.0)
        cfg = ChipModelConfig.from_calibration(calibration)
        assert cfg.operating_mode == "jetting"
        assert (cfg.ratio_low, cfg.ratio_high, cfg.ratio_thread) == (2.6, 4.0, 5.0)
        assert ChipSimulator(config=cfg, noise=False).regime_at(70.0, 20.0) is FlowRegime.JETTING


class TestOptionalRatioWindowPrior:
    @pytest.fixture
    def sim(self) -> ChipSimulator:
        calibration = CalibrationResult(ratio_low=2.6, ratio_high=4.0, ratio_thread=5.0)
        config = replace(ChipModelConfig.from_calibration(calibration), operating_mode="ratio_window")
        return ChipSimulator(config=config, noise=False)

    @pytest.mark.parametrize("ratio,expected", [
        (2.59, FlowRegime.TUBING),
        (2.6, FlowRegime.DRIPPING),
        (4.0, FlowRegime.DRIPPING),
        (4.01, FlowRegime.JETTING),
        (5.0, FlowRegime.JETTING),
        (5.01, FlowRegime.THREADING),
    ])
    def test_explicit_research_mode_implements_internal_edges(self, sim, ratio, expected) -> None:
        # Synthetic prior boundaries: success is not evidence about a real chip.
        assert sim.regime_at(ratio, 1.0) is expected
        features, observed, confidence = sim.observe(ratio * 20.0, 20.0)
        assert observed is expected
        assert confidence > 0
        assert features.source == "simulation"
        assert features.breakup_distance_h is None


def test_empty_history_learning_is_reproducible_and_keeps_priors(tmp_path: Path) -> None:
    first = learn_calibration(tmp_path)
    second = learn_calibration(tmp_path)
    assert first.as_dict() == second.as_dict()
    # Compatibility constants are explicitly unsupported priors, never physical truths.
    assert (first.ratio_low, first.ratio_high, first.ratio_thread) == (2.2, 4.185, 6.0)
    assert first.ratio_low_confidence == first.ratio_high_confidence == first.ratio_thread_confidence == "prior"
    assert first.calibration_status == "prior"
    assert not first.window_identified
    assert first.valid_label_points == 0
    assert first.ratio_low_interval is None and first.ratio_high_interval is None
    assert first.sources == ()
    assert first.warnings
    assert list(tmp_path.iterdir()) == []  # Read-only learning must not create host artifacts.
