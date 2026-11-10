"""扫描实验的数据模型：工作点、区间边界及未解决的证据缺口。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .phase import FlowRegime, NodeFeatures, RegimeThresholds


@dataclass(frozen=True)
class NodeRecord:
    """单节点观测；started_s 是稳定等待完成、观测实际开始的计划时间。"""

    index: int
    q1: float
    q2: float
    regime: FlowRegime
    confidence: float
    features: NodeFeatures
    started_s: float
    duration_s: float
    purpose: str = ""

    @property
    def elapsed_s(self) -> float:
        return self.started_s + self.duration_s

    @property
    def flow_ratio(self) -> float:
        """总油/水体积流量比；不是未校准的局部速度比。"""
        return self.q1 / self.q2 if self.q2 > 0 else math.inf


@dataclass(frozen=True)
class BoundaryBracket:
    """两次有效观测夹出的区间；中点仅用于画图，不是精确阈值。"""

    station_q1: float
    kind: str
    direction: str
    inside: NodeRecord  # 液滴边界始终规范化为窗口内
    outside: NodeRecord
    boundary_q2: float
    relative_width: float  # 区间端点比 max(Q2)/min(Q2)，不是其平方根
    at_domain_edge: bool = False
    resolution: str = "bracketed"  # bracketed / uncertain（区间内出现未知观测）

    @property
    def boundary_ratio(self) -> float:
        return self.station_q1 / self.boundary_q2 if self.boundary_q2 > 0 else math.inf

    @property
    def q2_interval(self) -> tuple[float, float]:
        return tuple(sorted((self.inside.q2, self.outside.q2)))


@dataclass(frozen=True)
class ScanConfig:
    """射流起点向两侧探索；先验不会决定是否必须找到边界。"""

    center_q1: float = 90.0
    center_q2: float = 20.0
    node_duration_s: float = 180.0
    startup_s: float = 30.0
    step_ratio: float = 1.60
    min_ratio_step: float = 1.15
    bisection_rounds: int = 3
    max_expand_steps: int = 7
    q1_min: float = 2.0
    q1_max: float = 200.0
    q2_min: float = 0.2
    q2_max: float = 100.0
    station_step_ratio: float = 1.80
    max_stations: int = 7
    max_nodes: int = 140
    safety_q1_over_q2: float = 2.0
    window_search_rounds: int = 1  # 粗扫跨过窄窗时，有限次插入探查点
    max_duration_s: float = 14400.0  # 计划及执行器时间预算；真实采样源须协作遵守截止
    target_regime: str = "jetting"  # 实验默认只探索射流；dripping仅作可选研究模式

    def validate(self) -> None:
        if self.target_regime not in {"jetting", "dripping"}:
            raise ValueError("target_regime必须为jetting或dripping")
        for name in (
            "center_q1", "center_q2", "node_duration_s", "step_ratio", "min_ratio_step",
            "q1_min", "q1_max", "q2_min", "q2_max", "station_step_ratio",
            "safety_q1_over_q2", "max_duration_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if not math.isfinite(self.startup_s) or self.startup_s < 0:
            raise ValueError("startup_s 必须为有限非负数")
        for name in ("bisection_rounds", "window_search_rounds", "max_expand_steps", "max_stations", "max_nodes"):
            value = getattr(self, name)
            minimum = 0 if name in {"bisection_rounds", "window_search_rounds"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
        if self.q1_min > self.q1_max or self.q2_min > self.q2_max:
            raise ValueError("流量下限不能高于上限")
        if not self.q1_min <= self.center_q1 <= self.q1_max or not self.q2_min <= self.center_q2 <= self.q2_max:
            raise ValueError("中心工作点必须在配置的流量范围内")
        if self.safety_q1_over_q2 < 2:
            raise ValueError("不得放宽泵的固定安全约束 Q1 >= 2*Q2")
        if self.center_q1 < self.safety_q1_over_q2 * self.center_q2:
            raise ValueError("中心工作点违反 Q1 >= 2*Q2 安全约束")
        if min(self.step_ratio, self.min_ratio_step, self.station_step_ratio) <= 1:
            raise ValueError("步幅比例必须 > 1")


@dataclass
class ScanResult:
    config: ScanConfig
    center_node: NodeRecord
    nodes: list[NodeRecord] = field(default_factory=list)
    brackets: list[BoundaryBracket] = field(default_factory=list)
    thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)
    metadata: dict[str, Any] = field(default_factory=dict)
    stopped_reason: str = ""

    @property
    def total_duration_s(self) -> float:
        return max((n.elapsed_s for n in self.nodes), default=0.0)

    def boundaries(self, kind: str) -> list[BoundaryBracket]:
        return [b for b in self.brackets if b.kind == kind]


def classify_boundary_kind(a: FlowRegime, b: FlowRegime) -> str:
    """仅流型的辅助分类；真正的上下边界由观测 Q2 的位置决定。"""
    if FlowRegime.UNKNOWN in (a, b) or a is b:
        return "other"
    if FlowRegime.DRIPPING in (a, b):
        return "droplet_upper" if FlowRegime.TUBING in (a, b) else "droplet_lower"
    if {a, b} == {FlowRegime.JETTING, FlowRegime.THREADING}:
        return "thread_onset"
    return "other"


def regime_group(regime: FlowRegime) -> str:
    if regime is FlowRegime.DRIPPING:
        return "droplet"
    if regime in (FlowRegime.JETTING, FlowRegime.THREADING):
        return "stream"
    if regime is FlowRegime.TUBING:
        return "plug"
    if regime is FlowRegime.DISPLACEMENT:
        return "displacement"
    return "unknown"
