"""扫描实验的数据模型（节点记录、边界区间、配置与结果）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .phase import FlowRegime, NodeFeatures, RegimeThresholds


@dataclass(frozen=True)
class NodeRecord:
    """一次实验节点：固定 Q1/Q2，观测一段时间后得到的分类结果。"""

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
        """油水流量比 ``r = Q1/Q2``（用户口径的“流速比”）。"""

        return self.q1 / self.q2 if self.q2 > 0 else float("inf")


@dataclass(frozen=True)
class BoundaryBracket:
    """一个被夹逼出来的流型边界区间。"""

    station_q1: float
    kind: str                 # droplet_upper / droplet_lower / thread_onset
    direction: str            # low / high（相对该站基准点）
    inside: NodeRecord        # 液滴区一侧（窗口内）
    outside: NodeRecord       # 另一流型一侧
    boundary_q2: float
    relative_width: float
    at_domain_edge: bool = False

    @property
    def boundary_ratio(self) -> float:
        """边界处的油水比 ``Q1/Q2``。"""

        return self.station_q1 / self.boundary_q2 if self.boundary_q2 > 0 else float("inf")


@dataclass(frozen=True)
class ScanConfig:
    """蜘蛛式扫描配置。"""

    center_q1: float = 70.0
    center_q2: float = 20.0
    node_duration_s: float = 180.0     # 每个节点观测约 3 分钟
    startup_s: float = 30.0            # 换流量后的稳定等待（不计入观测窗口）
    step_ratio: float = 1.60           # 粗扫步幅（乘法）
    min_ratio_step: float = 1.15       # 终止夹逼的最小相对区间宽度
    bisection_rounds: int = 3
    max_expand_steps: int = 7
    q1_min: float = 2.0
    q1_max: float = 200.0
    q2_min: float = 0.2
    q2_max: float = 100.0
    station_step_ratio: float = 1.80   # 蜘蛛腿沿 Q1 左右扩展的步幅
    max_stations: int = 7
    max_nodes: int = 140
    safety_q1_over_q2: float = 2.0     # 固定安全约束 Q1 >= 2*Q2

    def validate(self) -> None:
        if self.center_q1 <= 0 or self.center_q2 <= 0:
            raise ValueError("中心工作点流量必须为正")
        if self.center_q1 < self.safety_q1_over_q2 * self.center_q2:
            raise ValueError(
                f"中心工作点违反 Q1 >= {self.safety_q1_over_q2:g}*Q2："
                f"Q1={self.center_q1}, Q2={self.center_q2}"
            )
        if self.step_ratio <= 1.0 or self.min_ratio_step <= 1.0:
            raise ValueError("步幅比例必须 > 1")
        if self.bisection_rounds < 0:
            raise ValueError("bisection_rounds 不能为负")
        if self.node_duration_s <= 0:
            raise ValueError("node_duration_s 必须为正")


@dataclass
class ScanResult:
    """一次完整扫描的产物。"""

    config: ScanConfig
    center_node: NodeRecord
    nodes: list[NodeRecord] = field(default_factory=list)
    brackets: list[BoundaryBracket] = field(default_factory=list)
    thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)
    metadata: dict[str, Any] = field(default_factory=dict)
    stopped_reason: str = ""

    @property
    def total_duration_s(self) -> float:
        if not self.nodes:
            return 0.0
        return max(n.elapsed_s for n in self.nodes)

    def boundaries(self, kind: str) -> list[BoundaryBracket]:
        return [b for b in self.brackets if b.kind == kind]


def classify_boundary_kind(a: FlowRegime, b: FlowRegime) -> str:
    """把两侧流型映射成液滴窗口边界类型。

    * ``droplet_upper``：栓塞 ↔ 滴落（Q2 较高一侧，r = ratio_low）；
    * ``droplet_lower``：滴落 ↔ 射流（Q2 较低一侧，r = ratio_high）；
    * ``thread_onset``：射流 ↔ 束流。
    """

    g1, g2 = regime_group(a), regime_group(b)
    groups = {g1, g2}
    if groups == {"droplet", "plug"}:
        return "droplet_upper"
    if groups == {"droplet", "stream"}:
        if (a is FlowRegime.DRIPPING and b is FlowRegime.JETTING) or (
            a is FlowRegime.JETTING and b is FlowRegime.DRIPPING
        ):
            return "droplet_lower"
        return "thread_onset"
    if a is FlowRegime.JETTING and b is FlowRegime.THREADING:
        return "thread_onset"
    if a is FlowRegime.THREADING and b is FlowRegime.JETTING:
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
