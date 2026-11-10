"""芯片流型模型（仿真器）的物理一致性测试。"""

from __future__ import annotations

import math

import pytest

from mcs_date.experiment import ChipSimulator, FlowRegime


class TestChipSimulator:
    def setup_method(self) -> None:
        self.sim = ChipSimulator()

    def test_center_working_point_is_threading(self) -> None:
        assert self.sim.regime_at(80.0, 20.0) is FlowRegime.THREADING

    def test_low_dispersed_flow_drips(self) -> None:
        assert self.sim.regime_at(80.0, 1.0) is FlowRegime.DRIPPING
        assert self.sim.regime_at(80.0, 2.0) is FlowRegime.DRIPPING

    def test_mid_band_jets(self) -> None:
        assert self.sim.regime_at(80.0, 7.0) is FlowRegime.JETTING

    def test_high_flow_ratio_floods(self) -> None:
        assert self.sim.regime_at(80.0, 40.0) is FlowRegime.TUBING

    def test_very_low_oil_flow_displaces(self) -> None:
        assert self.sim.regime_at(1.0, 0.2) is FlowRegime.DISPLACEMENT

    def test_unknown_for_non_positive_flow(self) -> None:
        assert self.sim.regime_at(0.0, 20.0) is FlowRegime.UNKNOWN

    def test_capillary_numbers_are_positive_and_grow_with_flow(self) -> None:
        ca_low, _ = self.sim.capillary_numbers(10.0, 10.0)
        ca_high, _ = self.sim.capillary_numbers(100.0, 10.0)
        assert ca_low > 0
        assert ca_high > ca_low

    def test_thread_diameter_follows_square_root_law(self) -> None:
        # δ/h ≈ sqrt(Q2 / (2 Q1))，角色互换版文献 Eq.1
        assert self.sim.thread_diameter_ratio(80.0, 20.0) == pytest.approx(math.sqrt(0.25 / 2.0))

    def test_thread_diameter_is_clipped(self) -> None:
        assert 0.0 <= self.sim.thread_diameter_ratio(80.0, 1000.0) <= 1.0

    def test_boundaries_rise_overall_with_oil_flow(self) -> None:
        stations = [10.0, 20.0, 40.0, 80.0, 160.0]
        lower = []
        upper = []
        for q1 in stations:
            ca, _ = self.sim.capillary_numbers(q1, 10.0)
            lower.append(self.sim.lambda_low(ca) * q1)
            upper.append(self.sim.lambda_up(ca) * q1)
        # λ_low 随 Ca 单调下降（模型基本假设）
        ca_values = [self.sim.capillary_numbers(q1, 10.0)[0] for q1 in stations]
        lambdas = [self.sim.lambda_low(ca) for ca in ca_values]
        assert lambdas == sorted(lambdas, reverse=True)
        # 高流量段下边界整体抬升，且上下边界永不相交
        assert lower[-1] > lower[0]
        assert all(lo < hi for lo, hi in zip(lower, upper))

    def test_observed_features_are_consistent_with_regime(self) -> None:
        for q1, q2 in [(80.0, 20.0), (80.0, 7.0), (80.0, 1.0), (80.0, 40.0)]:
            features = self.sim.features(q1, q2)
            assert features.q1 == q1
            assert features.q2 == q2
            assert features.window_s > 0
            assert 0.0 <= features.stream_coverage <= 1.0
            assert features.sample_count > 0
