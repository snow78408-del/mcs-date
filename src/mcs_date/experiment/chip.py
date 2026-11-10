"""离线芯片仿真：由 (Q1, Q2) 生成可测量的节点特征。

默认模式模拟“尺寸变化 → 断裂位置后移 → 观测窗口内不破裂束流”。
二维射流区、变化幅度与迁移规律都是演示先验，不是历史数据学出的物理规律。
射流可以在下游切割出液滴，不等于不成滴。以下比值窗口仅保留为可选研究模式：
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

窗口参数2.2/4.185/6.0全部是仿真先验，不是已复核的芯片边界。
历史长度候选不能确认滴落，早期长结构不能确认无滴。低比一侧也可能是
射流而非栓塞，不能从r低就给真实视频命名tubing。允许从reviewed标定注入
演示参数，但仿真成图不能反过来验证该模型。

尺寸模型沿用 10/16 campaign 的线性拟合::

    D[µm] = a + b*(Q1-60) + c*(Q2-20)

它来自长度候选的低置信拟合，只能作为尺寸参考，不能用来判定滴落/射流。
文献角色互换后的水丝直径关系也仅作示意，不证明本芯片适用。
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

    h_m: float = 50e-6           # 名义方通道边长（仍需实际标定）
    mu_continuous: float = 0.050  # Pa·s，油相
    mu_dispersed: float = 0.0010  # Pa·s，水相
    sigma: float = 0.030          # N/m，界面张力


@dataclass(frozen=True)
class ChipModelConfig:
    """仿真芯片的液滴窗口参数（可由历史标定覆盖）。"""

    ratio_low: float = 2.2      # 低于此油水比：切割不足 → 栓塞（先验）
    ratio_high: float = 4.185   # 高于此油水比：射流（仅离线先验）
    ratio_thread: float = 6.0   # 更高：稳定束流（先验）
    size_intercept_um: float = FALLBACK_SIZE[0]
    size_q1_coef_um_per_ulmin: float = FALLBACK_SIZE[1]
    size_q2_coef_um_per_ulmin: float = FALLBACK_SIZE[2]
    cross_section_um: float = 50.0   # 历史芯片名义截面（仅用于直径归一化）
    operating_mode: str = "jetting"  # 默认射流退化示意；ratio_window仅作可选旧研究模型
    jetting_center_q1: float = 90.0
    jetting_center_q2: float = 20.0
    jetting_log_flow_radius: float = 0.70
    jetting_log_ratio_radius: float = 0.65
    migration_start_fraction: float = 0.35
    anchor_breakup_h: float = 2.0
    observation_extent_h: float = 20.0

    def __post_init__(self) -> None:
        if self.operating_mode not in {"jetting", "ratio_window"}:
            raise ValueError("operating_mode必须为jetting或ratio_window")
        for name in ("ratio_low", "ratio_high", "ratio_thread", "cross_section_um",
                     "jetting_center_q1", "jetting_center_q2", "jetting_log_flow_radius",
                     "jetting_log_ratio_radius", "anchor_breakup_h", "observation_extent_h"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name}必须为有限正数")
        if not self.ratio_low < self.ratio_high < self.ratio_thread:
            raise ValueError("窗口先验必须有序")
        if not 0 <= self.migration_start_fraction < 1 or self.anchor_breakup_h >= self.observation_extent_h:
            raise ValueError("断裂位置示意参数无效")

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
        if not math.isfinite(q1) or not math.isfinite(q2) or q1 <= 0 or q2 <= 0:
            return FlowRegime.UNKNOWN
        r = self.oil_water_ratio(q1, q2)
        if self.config.operating_mode == "jetting":
            return FlowRegime.JETTING if self.jetting_deviation(q1, q2) < 1.0 else FlowRegime.THREADING
        if r < self.config.ratio_low:
            return FlowRegime.TUBING
        if r <= self.config.ratio_high:
            return FlowRegime.DRIPPING
        if r <= self.config.ratio_thread:
            return FlowRegime.JETTING
        return FlowRegime.THREADING

    def jetting_deviation(self, q1: float, q2: float) -> float:
        """二维射流退化示意；半径/中心均为先验，不是由历史标签学出的边界。"""
        cfg = self.config
        a = math.log(q1 / cfg.jetting_center_q1) / cfg.jetting_log_flow_radius
        r0 = cfg.jetting_center_q1 / cfg.jetting_center_q2
        b = math.log((q1 / q2) / r0) / cfg.jetting_log_ratio_radius
        return math.hypot(a, b)

    def breakup_distance_h(self, q1: float, q2: float) -> float | None:
        cfg = self.config
        if cfg.operating_mode != "jetting" or self.regime_at(q1, q2) is not FlowRegime.JETTING:
            return None
        rho = self.jetting_deviation(q1, q2)
        # 先改变尺寸；偏离一定幅度后切割位置再明显向下游迁移。
        progress = max(0.0, (rho - cfg.migration_start_fraction) / (1 - cfg.migration_start_fraction))
        return cfg.anchor_breakup_h + (cfg.observation_extent_h - cfg.anchor_breakup_h) * progress ** 2

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

        if self.config.operating_mode == "jetting":
            if regime is FlowRegime.JETTING:
                cfg = self.config
                ratio_delta = math.log((q1 / q2) / (cfg.jetting_center_q1 / cfg.jetting_center_q2))
                diameter = self.droplet_diameter_um(q1, q2) / cfg.cross_section_um * max(0.3, 1 + 0.5 * ratio_delta)
                cv = 0.08 + 0.18 * self.jetting_deviation(q1, q2) ** 2
            elif regime is FlowRegime.THREADING:
                rate = 0.0

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
            source="simulation",
            droplet_events=int(rate * window_s) if regime in {FlowRegime.DRIPPING, FlowRegime.JETTING} else 0,
            junction_visible=self.config.operating_mode == "jetting",
            breakup_distance_h=self.breakup_distance_h(q1, q2) if self.config.operating_mode == "jetting" else None,
            observation_extent_h=self.config.observation_extent_h if self.config.operating_mode == "jetting" else None,
        )

    def observe(self, q1: float, q2: float, *, window_s: float = 180.0) -> tuple[NodeFeatures, FlowRegime, float]:
        """生成特征并走一遍真实分类器，返回 (特征, 分类结果, 置信度)。"""

        features = self.features(q1, q2, window_s=window_s)
        regime, confidence = classify_regime(features, self.thresholds)
        return features, regime, confidence
