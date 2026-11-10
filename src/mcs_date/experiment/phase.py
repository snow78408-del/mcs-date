"""流型定义、节点观测量与基于观测量窗口的流型分类。

分类器只吃测量特征，不从流量比推断实测流型。真实滴落/射流识别还要求
结点可见、断裂位置和去重生成事件；目前 VisionPipeline 本身尚不提供全部
这些字段。缺形态证据时返回UNKNOWN；启发式只能显式选择或用于仿真。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class FlowRegime(str, Enum):
    """方形微通道十字结处的两相流型（术语沿用 Cubaud & Mason 2008）。"""

    DRIPPING = "dripping"          # 液滴在结处生成，无连续束流
    JETTING = "jetting"            # 形成射流，在距结 < 20h 内断裂成液滴
    THREADING = "threading"        # 稳定束流/thread，在 20h 内不断裂
    TUBING = "tubing"              # 分散相几乎充满通道（栓塞流）
    DISPLACEMENT = "displacement"  # 低毛细数下的浸润/指进，束流失稳
    UNKNOWN = "unknown"

    @property
    def label_zh(self) -> str:
        return {
            FlowRegime.DRIPPING: "滴落",
            FlowRegime.JETTING: "射流",
            FlowRegime.THREADING: "束流",
            FlowRegime.TUBING: "栓塞/充满",
            FlowRegime.DISPLACEMENT: "浸润/指进",
            FlowRegime.UNKNOWN: "未知",
        }[self]

    @property
    def is_stream(self) -> bool:
        """是否为“形成束流”的流型（射流与稳定束流都算）。"""
        return self in (FlowRegime.JETTING, FlowRegime.THREADING)


@dataclass(frozen=True)
class NodeFeatures:
    """一个实验节点（固定 Q1/Q2、一段观测窗口）内提取的测量特征。"""

    q1: float
    q2: float
    droplet_rate_hz: float = 0.0          # 单位时间新增液滴数
    mean_diameter_ratio: float = 0.0      # 平均液滴直径 / 通道宽度
    diameter_cv: float = 0.0              # 直径变异系数
    stream_coverage: float = 0.0          # 沿流动方向的连续液流覆盖率 [0,1]
    max_plug_length_ratio: float = 0.0    # 最长栓长 / 观测长度
    sample_count: int = 0
    window_s: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)
    source: str = "unspecified"  # simulation / camera / recorded_video / heuristic
    droplet_events: int | None = None  # 去重越线/生成事件，绝非逐帧检测数相加
    junction_visible: bool = False
    breakup_distance_h: float | None = None
    observation_extent_h: float | None = None

    @property
    def flow_ratio(self) -> float:
        """油水流量比 ``r = Q1/Q2``（用户口径的“流速比”）。"""

        return self.q1 / self.q2 if self.q2 > 0 else math.inf


@dataclass(frozen=True)
class RegimeThresholds:
    """流型判据阈值（可通过真实实验数据校准）。"""

    # 束流：连续液流覆盖率高、几乎不再生成离散液滴
    threading_min_coverage: float = 0.70
    threading_max_rate_hz: float = 1.0
    threading_max_diameter_ratio: float = 0.75
    # 栓塞：覆盖率极高、液段几乎填满通道
    tubing_min_coverage: float = 0.90
    tubing_min_diameter_ratio: float = 0.75
    tubing_max_rate_hz: float = 0.5
    # 射流：中等覆盖率 + 稳定的高频小液滴
    jetting_min_coverage: float = 0.15
    jetting_max_coverage: float = 0.70
    jetting_min_rate_hz: float = 1.5
    jetting_min_diameter_ratio: float = 0.15
    jetting_max_diameter_ratio: float = 0.75
    # 滴落：几乎无连续液流 + 低频大液滴
    dripping_max_coverage: float = 0.15
    dripping_min_diameter_ratio: float = 0.60
    dripping_max_rate_hz: float = 2.0
    # 浸润/指进：尺寸高度不均且覆盖率低
    displacement_max_coverage: float = 0.35
    displacement_min_cv: float = 0.35
    displacement_max_rate_hz: float = 1.0
    min_droplet_events: int = 3
    max_uniform_diameter_cv: float = 0.20
    junction_breakup_max_h: float = 1.0  # 待芯片标注校准，不是文献通用常数
    threading_observation_h: float = 20.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def classify_regime(
    features: NodeFeatures,
    thresholds: RegimeThresholds | None = None,
) -> tuple[FlowRegime, float]:
    """按特征判定流型，返回 ``(流型, 置信度)``。

    置信度定义为命中判据偏离最近阈值的相对余量（0~1），仅用于排序和
    实验日志，不作为统计显著性。
    """
    th = thresholds or RegimeThresholds()
    numeric = (features.q1, features.q2, features.window_s, features.sample_count,
               features.stream_coverage, features.droplet_rate_hz, features.mean_diameter_ratio,
               features.diameter_cv, features.max_plug_length_ratio)
    if (not all(math.isfinite(v) for v in numeric) or features.sample_count <= 0
            or features.window_s <= 0 or features.q1 <= 0 or features.q2 <= 0
            or not 0 <= features.stream_coverage <= 1 or features.droplet_rate_hz < 0
            or features.mean_diameter_ratio < 0 or features.diameter_cv < 0
            or not 0 <= features.max_plug_length_ratio <= 1):
        return FlowRegime.UNKNOWN, 0.0
    for value in (features.breakup_distance_h, features.observation_extent_h):
        if value is not None and (not math.isfinite(value) or value < 0):
            return FlowRegime.UNKNOWN, 0.0
    if features.droplet_events is not None and (
        isinstance(features.droplet_events, bool) or not isinstance(features.droplet_events, int)
        or features.droplet_events < 0
    ):
        return FlowRegime.UNKNOWN, 0.0

    formation = assess_formation(features, th)
    if features.junction_visible and features.breakup_distance_h is not None:
        if (not formation["droplets_observed"] or formation["droplet_events"] < th.min_droplet_events
                or features.observation_extent_h is not None
                and features.breakup_distance_h > features.observation_extent_h):
            return FlowRegime.UNKNOWN, 0.0
        if features.breakup_distance_h <= th.junction_breakup_max_h:
            if not formation["uniformity_candidate"]:
                return FlowRegime.UNKNOWN, 0.0
            return FlowRegime.DRIPPING, 0.8  # 规则评分，不是标注正确率
        if features.breakup_distance_h < th.threading_observation_h:
            return FlowRegime.JETTING, 0.8
        return FlowRegime.UNKNOWN, 0.0
    if (features.junction_visible and features.observation_extent_h is not None
            and features.observation_extent_h >= th.threading_observation_h
            and features.stream_coverage >= th.threading_min_coverage
            and features.droplet_events == 0 and features.droplet_rate_hz == 0):
        return FlowRegime.THREADING, 0.8
    if features.junction_visible and features.observation_extent_h is not None:
        # 视野不够、丢帧或没有可靠断裂证据时，不落回启发式宣称束流。
        return FlowRegime.UNKNOWN, 0.0
    # 没有结点/断裂位置的真实视频不可凭“大、低频”就宣称dripping。
    # 保留显式 heuristic 模式给旧研究脚本，仿真也必须打来源标识。
    if features.source not in {"simulation", "heuristic"}:
        return FlowRegime.UNKNOWN, 0.0

    coverage = clamp01(features.stream_coverage)
    rate = max(0.0, features.droplet_rate_hz)
    diameter = max(0.0, features.mean_diameter_ratio)
    cv = max(0.0, features.diameter_cv)

    # 1) 栓塞/充满通道
    if (
        coverage >= th.tubing_min_coverage
        and diameter >= th.tubing_min_diameter_ratio
        and rate <= th.tubing_max_rate_hz
    ):
        margin = min(
            coverage - th.tubing_min_coverage,
            diameter - th.tubing_min_diameter_ratio,
        )
        return FlowRegime.TUBING, _confidence(margin, 0.10)

    # 2) 稳定束流
    if (
        coverage >= th.threading_min_coverage
        and rate <= th.threading_max_rate_hz
        and diameter <= th.threading_max_diameter_ratio
    ):
        margin = min(
            coverage - th.threading_min_coverage,
            th.threading_max_diameter_ratio - diameter,
        )
        return FlowRegime.THREADING, _confidence(margin, 0.10)

    # 3) 射流（覆盖不足但液滴细小且高频）
    if (
        th.jetting_min_coverage <= coverage <= th.jetting_max_coverage
        and rate >= th.jetting_min_rate_hz
        and th.jetting_min_diameter_ratio <= diameter <= th.jetting_max_diameter_ratio
    ):
        margin = min(
            coverage - th.jetting_min_coverage,
            rate - th.jetting_min_rate_hz,
        )
        return FlowRegime.JETTING, _confidence(margin, 0.12)

    # 4) 滴落（低频大液滴，基本无连续束流）
    if (
        coverage <= th.dripping_max_coverage
        and diameter >= th.dripping_min_diameter_ratio
        and 0 < rate <= th.dripping_max_rate_hz
        and bool(formation["uniformity_candidate"])
    ):
        margin = min(
            th.dripping_max_coverage - coverage,
            diameter - th.dripping_min_diameter_ratio,
        )
        return FlowRegime.DRIPPING, _confidence(margin, 0.10)

    # 5) 浸润/指进（尺寸高度离散）
    if coverage <= th.displacement_max_coverage and cv >= th.displacement_min_cv:
        return FlowRegime.DISPLACEMENT, _confidence(cv - th.displacement_min_cv, 0.10)

    return FlowRegime.UNKNOWN, 0.0


def assess_formation(features: NodeFeatures, thresholds: RegimeThresholds | None = None) -> dict[str, object]:
    """将“确有离散液滴/均一候选”与“滴落/射流形态”分开，射流也可能成滴。

    均一候选仅表示事件数与尺寸CV达标，不代表最佳点或已验证长期稳定。
    """
    th = thresholds or RegimeThresholds()
    events = features.droplet_events
    if events is None and features.source in {"simulation", "heuristic"}:
        if math.isfinite(features.droplet_rate_hz) and math.isfinite(features.window_s):
            events = max(0, int(features.droplet_rate_hz * features.window_s))
    valid = (features.sample_count > 0 and math.isfinite(features.window_s)
             and features.window_s > 0 and math.isfinite(features.droplet_rate_hz)
             and features.droplet_rate_hz >= 0
             and isinstance(events, int) and not isinstance(events, bool) and events >= 0)
    observed = bool(events and features.droplet_rate_hz > 0) if valid else None
    uniform = bool(valid and observed and events >= th.min_droplet_events
                   and math.isfinite(features.mean_diameter_ratio) and features.mean_diameter_ratio > 0
                   and math.isfinite(features.diameter_cv)
                   and 0 <= features.diameter_cv <= th.max_uniform_diameter_cv)
    return {"droplet_events": events, "droplets_observed": observed,
            "uniformity_candidate": uniform,
            "best_operating_point_verified": False}


def _confidence(margin: float, scale: float) -> float:
    if not math.isfinite(margin):
        return 0.0
    return clamp01(0.5 + margin / (2.0 * scale)) * 0.98


def merge_features(samples: list[NodeFeatures]) -> NodeFeatures:
    """把同一节点的多段特征按时间加权合并。"""
    if not samples:
        raise ValueError("samples 不能为空")
    first = samples[0]
    if any(not math.isfinite(s.window_s) or s.window_s <= 0 for s in samples):
        raise ValueError("合并窗口必须为有限正时长")
    if any(not math.isclose(s.q1, first.q1) or not math.isclose(s.q2, first.q2) for s in samples):
        raise ValueError("不能把不同流量节点合并成一个工作点")
    total = sum(max(s.window_s, 1e-9) for s in samples)
    weights = [max(s.window_s, 1e-9) / total for s in samples]

    def weighted(attr: str) -> float:
        return sum(getattr(s, attr) * w for s, w in zip(samples, weights))

    first = samples[0]
    return NodeFeatures(
        q1=weighted("q1"),
        q2=weighted("q2"),
        droplet_rate_hz=weighted("droplet_rate_hz"),
        mean_diameter_ratio=weighted("mean_diameter_ratio"),
        diameter_cv=weighted("diameter_cv"),
        stream_coverage=weighted("stream_coverage"),
        max_plug_length_ratio=max(s.max_plug_length_ratio for s in samples),
        sample_count=sum(s.sample_count for s in samples),
        window_s=sum(s.window_s for s in samples),
        extra={k: v for k, v in first.extra.items()},
        source=first.source if all(s.source == first.source for s in samples) else "unspecified",
        droplet_events=sum(s.droplet_events for s in samples) if all(s.droplet_events is not None for s in samples) else None,
        junction_visible=all(s.junction_visible for s in samples),
        breakup_distance_h=first.breakup_distance_h if all(s.breakup_distance_h == first.breakup_distance_h for s in samples) else None,
        observation_extent_h=min(s.observation_extent_h for s in samples) if all(s.observation_extent_h is not None for s in samples) else None,
    )
