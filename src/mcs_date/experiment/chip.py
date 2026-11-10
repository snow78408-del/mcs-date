"""离线芯片仿真：由 (Q1, Q2) 生成可测量的节点特征。

用途：在没有相机/泵/芯片时验证扫描策略、边界拟合与二相图动画；也作为
真实实验的“标定先验”。模型采用 Cubaud & Mason (2008) 的无量纲框架
（Ca_i = μ_i J_i/σ12，J_i = Q_i/h²，λ = Q2/Q1，束流直径 δ/h ≈ √(λ/2)），
但 **注意流路角色是反的**：

* 文献 L1 是较粘的中心相、L2 是较稀的鞘相；
* 本芯片 CH1 油相从两侧进入（连续/鞘相，总流量 Q1），CH2 水相从中心
  进入（分散/中心相，流量 Q2）。

因此阈值（``ca_thread_min``、``lambda_low``、``lambda_up`` 等）是待标定
参数，不是论文中的普适常数；真实芯片上应通过扫描实验重新拟合。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

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
    """仿真芯片的流型边界参数（可标定）。"""

    ca_drip_crit: float = 0.010     # Ca_continuous 低于此值：只有滴落
    ca_thread_min: float = 0.080    # Ca_continuous 高于此值才可能出现稳定束流
    lambda_low_base: float = 0.040  # 束流所需最小流量比下限
    lambda_low_gain: float = 0.350  # 油相不足时所需流量比的上浮量
    lambda_low_scale: float = 0.040  # 上浮随 Ca_continuous 衰减的尺度
    lambda_up: float = 0.460        # 高于此流量比：栓塞/充满通道
    thread_lambda_scale: float = 0.280
    ca_thread_ref: float = 0.100
    thread_lambda_exp: float = 0.350
    ca_wetting_max: float = 0.004   # 极低油相毛细数：浸润/指进


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
    def flow_ratio(q1: float, q2: float) -> float:
        return q2 / q1 if q1 > 0 else math.inf

    def thread_diameter_ratio(self, q1: float, q2: float) -> float:
        """稳定束流直径 δ/h ≈ √(Q2/(2Q1))（文献 Eq.1 的角色互换版）。"""
        lam = self.flow_ratio(q1, q2)
        if not math.isfinite(lam) or lam <= 0:
            return 0.0
        return max(0.0, min(1.0, math.sqrt(lam / 2.0)))

    # -- 边界 ---------------------------------------------------------------
    def lambda_low(self, ca_continuous: float) -> float:
        """给定油相毛细数时，形成连续束流所需的最小流量比 λ。"""
        return self.config.lambda_low_base + self.config.lambda_low_gain * math.exp(
            -ca_continuous / self.config.lambda_low_scale
        )

    def lambda_thread(self, ca_continuous: float) -> float:
        """滴落/射流 → 稳定束流的流量比阈值。"""
        raw = self.config.thread_lambda_scale * (
            self.config.ca_thread_ref / max(ca_continuous, 1e-6)
        ) ** self.config.thread_lambda_exp
        low = self.lambda_low(ca_continuous)
        return max(low, min(self.config.lambda_up * 0.9, raw))

    def lambda_up(self, _ca_continuous: float = 0.0) -> float:
        return self.config.lambda_up

    # -- 流型 ---------------------------------------------------------------
    def regime_at(self, q1: float, q2: float) -> FlowRegime:
        if q1 <= 0 or q2 <= 0:
            return FlowRegime.UNKNOWN
        ca_c, _ca_d = self.capillary_numbers(q1, q2)
        lam = self.flow_ratio(q1, q2)
        if lam >= self.lambda_up(ca_c):
            return FlowRegime.TUBING
        if ca_c < self.config.ca_wetting_max:
            return FlowRegime.DISPLACEMENT
        if ca_c < self.config.ca_drip_crit:
            return FlowRegime.DRIPPING
        if lam < self.lambda_low(ca_c):
            return FlowRegime.DRIPPING
        if ca_c >= self.config.ca_thread_min and lam >= self.lambda_thread(ca_c):
            return FlowRegime.THREADING
        return FlowRegime.JETTING

    # -- 测量特征 -----------------------------------------------------------
    def features(self, q1: float, q2: float, *, window_s: float = 180.0) -> NodeFeatures:
        """按流型生成一段观测窗口的特征（含测量噪声）。"""
        regime = self.regime_at(q1, q2)
        delta = self.thread_diameter_ratio(q1, q2)

        def jitter(scale: float) -> float:
            return self._rng.gauss(0.0, scale) if self.noise else 0.0

        if regime is FlowRegime.THREADING:
            coverage = 0.84 + jitter(0.03) - 0.10 * abs(delta - 0.35)
            diameter = max(0.05, delta + jitter(0.005))
            rate = max(0.0, 0.10 + jitter(0.08))
            cv = 0.06 + abs(jitter(0.02))
            plug = 0.85 + jitter(0.05)
        elif regime is FlowRegime.TUBING:
            coverage = 0.97 + jitter(0.01)
            diameter = 0.95 + jitter(0.04)
            rate = max(0.0, 0.05 + jitter(0.05))
            cv = 0.05 + abs(jitter(0.02))
            plug = 0.98 + jitter(0.01)
        elif regime is FlowRegime.JETTING:
            coverage = 0.45 + jitter(0.07) + 0.15 * (delta - 0.2)
            diameter = max(0.15, 0.42 + jitter(0.05))
            rate = max(1.8, 6.0 + jitter(1.6))
            cv = 0.12 + abs(jitter(0.03))
            plug = 0.30 + jitter(0.05)
        elif regime is FlowRegime.DRIPPING:
            coverage = 0.06 + abs(jitter(0.03))
            diameter = max(0.65, 1.05 + jitter(0.12))
            rate = max(0.1, 0.8 + jitter(0.25))
            cv = 0.10 + abs(jitter(0.03))
            plug = 0.08 + abs(jitter(0.03))
        elif regime is FlowRegime.DISPLACEMENT:
            coverage = 0.20 + jitter(0.08)
            diameter = max(0.2, 0.80 + jitter(0.25))
            rate = max(0.05, 0.5 + jitter(0.3))
            cv = 0.55 + abs(jitter(0.10))
            plug = 0.55 + jitter(0.15)
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
