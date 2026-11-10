"""离线芯片仿真：由 (Q1, Q2) 生成可测量的节点特征。

流型模型（用户 2026-11-10 口径 + 历史数据标定）
-----------------------------------------------
油相 CH1 从两侧进入（连续相，总流量 Q1），水相 CH2 从中心进入（分散相，
流量 Q2）。定义油水流量比::

    r = Q1 / Q2

* ``r < ratio_low``：连续相切割能力不足，水相保持宽大的栓塞/塞流，
  形成不了液滴 → :class:`~.phase.FlowRegime.TUBING`（栓塞）；
* ``ratio_low <= r <= ratio_high``：只有这一段能稳定形成液滴 →
  :class:`~.phase.FlowRegime.DRIPPING`（滴落）；
* ``ratio_high < r <= ratio_thread``：拖拽/剪切占优，液滴在通道下游才
  脱离 → :class:`~.phase.FlowRegime.JETTING`（射流）；
* ``r > ratio_thread``：水相被拉成稳定细丝、不破裂 →
  :class:`~.phase.FlowRegime.THREADING`（束流）。

窗口参数默认来自 :mod:`mcs_date.experiment.calibration` 对 MCS_QT 历史
数据的标定（``ratio_high≈4.19``，由 3.889 成滴与 4.505 长结构夹逼）。
``ratio_low`` 在真实设备上受泵安全约束 ``Q1 >= 2*Q2`` 限制，实测不可达，
只能先验取值，置信度标为 ``prior``。

尺寸模型沿用 10/16 campaign 的线性拟合::

    D[µm] = a + b*(Q1-60) + c*(Q2-20)

它只对液滴窗口内部有意义，用于让离线特征更接近历史观测。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .calibration import FALLBACK_SIZE, CalibrationResult
from .phase import FlowRegime, NodeFeatures, RegimeThresholds, classify_regime

MU_L_PER_M3 = 6.0e10  # 1 m³/s = 6e10 µL/min


@dataclass(frozen=True)
class ChipFluids:
    """默认流体对：硅油连续相 + 水分散相（可按实际试剂修改）。"""

    h_m: float = 100e-6          # 方通道边长
    mu_continuous: float = 0.050  # Pa·s，油相
    mu_dispersed: float = 0.0010  # Pa·s，水相
    sigma: float = 0.030          # N/m，界面张力


@dataclass(frozen=True)
class ChipModelConfig:
    """仿真芯片的液滴窗口参数（可由历史标定覆盖）。"""

    ratio_low: float = 2.2      # 低于此油水比：切割不足 → 栓塞（先验）
    ratio_high: float = 4.185   # 高于此油水比：拖拽占优 → 射流（历史夹逼）
    ratio_thread: float = 6.0   # 更高：稳定束流（先验）
    size_intercept_um: float = FALLBACK_SIZE[0]
    size_q1_coef_um_per_ulmin: float = FALLBACK_SIZE[1]
    size_q2_coef_um_per_ulmin: float = FALLBACK_SIZE[2]
    cross_section_um: float = 50.0   # 历史芯片名义截面（仅用于直径归一化）

    @classmethod
    def from_calibration(cls, calibration: CalibrationResult) -> "ChipModelConfig":
        return cls(
            ratio_low=calibration.ratio_low,
            ratio_high=calibration.ratio_high,
            ratio_thread=calibration.ratio_thread,
            size_intercept_um=calibration.size_intercept_um,
            size_q1_coef_um_per_ulmin=calibration.size_q1_coef_um_per_ulmin,
            size_q2_coef_um_per_ulmin=calibration.size_q2_coef_um_per_ulmin,
            cross_section_um=calibration.nominal_cross_section_um,
        )


@dataclass
class ChipSimulator:
    """由流量点生成流型与测量特征（带可复现噪声）。"""

    fluids: ChipFluids = field(default_factory=ChipFluids)
    config: ChipModelConfig = field(default_factory=ChipModelConfig)
    thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)
    seed: int = 7
    noise: bool = True

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -- 无量纲量 -----------------------------------------------------------
    def capillary_numbers(self, q1: float, q2: float) -> tuple[float, float]:
        h2 = self.fluids.h_m ** 2
        ca_c = self.fluids.mu_continuous * (q1 / MU_L_PER_M3) / h2 / self.fluids.sigma
        ca_d = self.fluids.mu_dispersed * (q2 / MU_L_PER_M3) / h2 / self.fluids.sigma
        return ca_c, ca_d

    @staticmethod
    def oil_water_ratio(q1: float, q2: float) -> float:
        """油水流量比 ``r = Q1/Q2``（用户口径的“流速比”）。"""

        return q1 / q2 if q2 > 0 else math.inf

    def thread_diameter_ratio(self, q1: float, q2: float) -> float:
        """稳定水丝直径 δ/h ≈ √(Q2/(2Q1)) = √(1/(2r))（Cubaud Eq.1 角色互换）。"""

        r = self.oil_water_ratio(q1, q2)
        if not math.isfinite(r) or r <= 0:
            return 0.0
        return max(0.0, min(1.0, math.sqrt(1.0 / (2.0 * r))))

    # -- 边界 ---------------------------------------------------------------
    def window(self) -> tuple[float, float, float]:
        return (self.config.ratio_low, self.config.ratio_high, self.config.ratio_thread)

    # -- 流型 ---------------------------------------------------------------
    def regime_at(self, q1: float, q2: float) -> FlowRegime:
        if q1 <= 0 or q2 <= 0:
            return FlowRegime.UNKNOWN
        r = self.oil_water_ratio(q1, q2)
        if r < self.config.ratio_low:
            return FlowRegime.TUBING
        if r <= self.config.ratio_high:
            return FlowRegime.DRIPPING
        if r <= self.config.ratio_thread:
            return FlowRegime.JETTING
        return FlowRegime.THREADING

    def droplet_diameter_um(self, q1: float, q2: float) -> float:
        raw = (
            self.config.size_intercept_um
            + self.config.size_q1_coef_um_per_ulmin * (q1 - 60.0)
            + self.config.size_q2_coef_um_per_ulmin * (q2 - 20.0)
        )
        return max(1.0, raw)

    # -- 测量特征 -----------------------------------------------------------
    def features(self, q1: float, q2: float, *, window_s: float = 180.0) -> NodeFeatures:
        """按流型生成一段观测窗口的特征（含测量噪声）。"""

        regime = self.regime_at(q1, q2)
        delta = self.thread_diameter_ratio(q1, q2)

        def jitter(scale: float) -> float:
            return self._rng.gauss(0.0, scale) if self.noise else 0.0

        if regime is FlowRegime.THREADING:
            coverage = 0.84 + jitter(0.02) - 0.10 * abs(delta - 0.30)
            diameter = max(0.05, delta + jitter(0.004))
            rate = max(0.0, 0.10 + jitter(0.06))
            cv = 0.06 + abs(jitter(0.02))
            plug = 0.85 + jitter(0.04)
        elif regime is FlowRegime.TUBING:
            coverage = 0.96 + jitter(0.01)
            diameter = 0.92 + jitter(0.03)
            rate = max(0.0, 0.05 + jitter(0.04))
            cv = 0.05 + abs(jitter(0.02))
            plug = 0.96 + jitter(0.01)
        elif regime is FlowRegime.JETTING:
            coverage = 0.42 + jitter(0.06) + 0.10 * (delta - 0.20)
            diameter = max(0.18, 0.40 + jitter(0.05))
            rate = max(2.0, 5.5 + jitter(1.4))
            cv = 0.14 + abs(jitter(0.03))
            plug = 0.30 + jitter(0.05)
        elif regime is FlowRegime.DRIPPING:
            diameter_ratio = self.droplet_diameter_um(q1, q2) / max(self.config.cross_section_um, 1e-6)
            coverage = 0.05 + abs(jitter(0.02))
            diameter = max(0.60, min(1.15, diameter_ratio + jitter(0.05)))
            rate = max(0.4, 1.0 + jitter(0.25))
            cv = 0.09 + abs(jitter(0.02))
            plug = 0.07 + abs(jitter(0.02))
        else:  # UNKNOWN：无有效测量
            coverage = diameter = rate = cv = plug = 0.0

        return NodeFeatures(
            q1=q1,
            q2=q2,
            droplet_rate_hz=max(0.0, rate),
            mean_diameter_ratio=max(0.0, diameter),
            diameter_cv=max(0.0, cv),
            stream_coverage=max(0.0, min(1.0, coverage)),
            max_plug_length_ratio=max(0.0, min(1.0, plug)),
            sample_count=int(window_s * 5),
            window_s=window_s,
        )

    def observe(self, q1: float, q2: float, *, window_s: float = 180.0) -> tuple[NodeFeatures, FlowRegime, float]:
        """生成特征并走一遍真实分类器，返回 (特征, 分类结果, 置信度)。"""

        features = self.features(q1, q2, window_s=window_s)
        regime, confidence = classify_regime(features, self.thresholds)
        return features, regime, confidence
