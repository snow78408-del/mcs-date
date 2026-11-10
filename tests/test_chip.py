"""芯片流型模型（仿真器）的物理一致性测试。

模型口径（用户 2026-11-10）：油水比 ``r = Q1/Q2`` 太低切割不足、太高
拖拽占优，只有中间窗口能形成液滴。默认参数来自历史标定：
``r_low=2.2``（先验）、``r_high=4.185``（历史夹逼）、``r_thread=6.0``。
"""

from __future__ import annotations

import math

import pytest

from mcs_date.experiment import ChipModelConfig, ChipSimulator, FlowRegime
from mcs_date.experiment.calibration import CalibrationResult, learn_calibration


class TestChipSimulator:
    def setup_method(self) -> None:
        self.sim = ChipSimulator()

    # -- 液滴窗口 -----------------------------------------------------------
    def test_working_point_forms_droplets(self) -> None:
        # 70/20 是历史恒流基准（r=3.5），落在窗口内
        assert self.sim.regime_at(70.0, 20.0) is FlowRegime.DRIPPING

    def test_ideal_droplet_point(self) -> None:
        # 用户评价“液滴效果较理想”的 50/20（r=2.5）
        assert self.sim.regime_at(50.0, 20.0) is FlowRegime.DRIPPING

    def test_low_ratio_gives_plugs(self) -> None:
        # r < 2.2：切割不足，宽大栓塞，无液滴
        assert self.sim.regime_at(40.0, 20.0) is FlowRegime.TUBING
        assert self.sim.regime_at(20.0, 10.0) is FlowRegime.TUBING

    def test_high_ratio_gives_jetting(self) -> None:
        # 用户实际常在的射流区：90~100/20（r=4.5~5）
        assert self.sim.regime_at(90.0, 20.0) is FlowRegime.JETTING
        assert self.sim.regime_at(100.0, 20.0) is FlowRegime.JETTING

    def test_very_high_ratio_gives_thread(self) -> None:
        assert self.sim.regime_at(180.0, 20.0) is FlowRegime.THREADING

    def test_window_edges(self) -> None:
        cfg = self.sim.config
        q2 = 20.0
        assert self.sim.regime_at(cfg.ratio_low * q2 * 0.99, q2) is FlowRegime.TUBING
        assert self.sim.regime_at(cfg.ratio_low * q2 * 1.01, q2) is FlowRegime.DRIPPING
        assert self.sim.regime_at(cfg.ratio_high * q2 * 0.99, q2) is FlowRegime.DRIPPING
        assert self.sim.regime_at(cfg.ratio_high * q2 * 1.01, q2) is FlowRegime.JETTING
        assert self.sim.regime_at(cfg.ratio_thread * q2 * 1.01, q2) is FlowRegime.THREADING

    def test_unknown_for_non_positive_flow(self) -> None:
        assert self.sim.regime_at(0.0, 20.0) is FlowRegime.UNKNOWN
        assert self.sim.regime_at(20.0, 0.0) is FlowRegime.UNKNOWN

    # -- 无量纲量 -----------------------------------------------------------
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

    def test_droplet_diameter_follows_history_fit(self) -> None:
        # 10/16 campaign：D ≈ 92.2 µm @ (70,20)，随 Q1/Q2 缓慢变化
        d70 = self.sim.droplet_diameter_um(70.0, 20.0)
        d50 = self.sim.droplet_diameter_um(50.0, 20.0)
        assert 88.0 < d70 < 96.0
        assert d50 > d70  # 降低 Q1 略增大液滴（历史拟合负系数）

    # -- 特征与分类一致性 ---------------------------------------------------
    def test_observed_features_are_consistent_with_regime(self) -> None:
        cases = [(70.0, 20.0), (90.0, 20.0), (180.0, 20.0), (40.0, 20.0)]
        for q1, q2 in cases:
            expected = self.sim.regime_at(q1, q2)
            features, regime, confidence = self.sim.observe(q1, q2)
            assert regime is expected, f"{q1}/{q2}: {regime} != {expected}"
            assert 0.0 < confidence <= 1.0
            assert features.window_s > 0
            assert features.sample_count > 0
            assert 0.0 <= features.stream_coverage <= 1.0

    def test_calibration_overrides_window(self) -> None:
        calibration = CalibrationResult(ratio_low=2.6, ratio_high=4.0, ratio_thread=5.0)
        sim = ChipSimulator(config=ChipModelConfig.from_calibration(calibration))
        assert sim.regime_at(50.0, 20.0) is FlowRegime.TUBING       # r=2.5 < 2.6
        assert sim.regime_at(70.0, 20.0) is FlowRegime.DRIPPING     # 2.6 <= 3.5 <= 4.0
        assert sim.regime_at(90.0, 20.0) is FlowRegime.JETTING      # 4.5 > 4.0
        assert sim.regime_at(110.0, 20.0) is FlowRegime.THREADING   # 5.5 > 5.0

    def test_learned_calibration_is_reproducible(self) -> None:
        first = learn_calibration()
        second = learn_calibration()
        assert first.ratio_high == second.ratio_high
        assert first.ratio_low <= first.ratio_high < first.ratio_thread
