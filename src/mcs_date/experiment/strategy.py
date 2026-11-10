"""蜘蛛式扫描策略：从一个稳定工作点出发，向左/右扩展 Q1 站位，
在每个站位上沿 Q2 上下夹逼流型边界。"""

from __future__ import annotations

import math
from typing import Callable

from .models import (
    BoundaryBracket,
    NodeRecord,
    ScanConfig,
    ScanResult,
    classify_boundary_kind,
)
from .phase import FlowRegime, NodeFeatures, RegimeThresholds, classify_regime

Measurement = Callable[[float, float, float], NodeFeatures]
ProgressCallback = Callable[[NodeRecord], None]


class _BudgetExceeded(RuntimeError):
    pass


class SpiderScanner:
    """在 (Q1, Q2) 平面上执行“中心站 + 左右腿 + 上下夹逼”的扫描。

    每个节点 = 换流量 + 稳定等待 + 一段约 3 分钟的观测窗口。扫描只依赖
    :class:`Measurement` 返回的特征，因此离线仿真与真实硬件共用同一策略。
    """

    def __init__(
        self,
        measure: Measurement,
        config: ScanConfig | None = None,
        thresholds: RegimeThresholds | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.measure = measure
        self.config = config or ScanConfig()
        self.config.validate()
        self.thresholds = thresholds or RegimeThresholds()
        self.progress = progress
        self._nodes: list[NodeRecord] = []
        self._brackets: list[BoundaryBracket] = []
        self._time_s = 0.0

    # -- 公共入口 -----------------------------------------------------------
    def run(self) -> ScanResult:
        self._nodes = []
        self._brackets = []
        self._time_s = 0.0
        stopped_reason = ""
        center: NodeRecord | None = None

        stations = self._stations()
        try:
            for station_index, q1 in enumerate(stations):
                base = self._station_base(q1, station_index)
                if center is None:
                    center = base
                for direction in ("low", "high"):
                    self._scan_direction(base, direction)
        except _BudgetExceeded:
            stopped_reason = f"达到节点上限 {self.config.max_nodes}"
        except RuntimeError as exc:
            stopped_reason = f"扫描中止: {exc}"

        if center is None:
            raise RuntimeError("扫描未产生任何节点")

        return ScanResult(
            config=self.config,
            center_node=center,
            nodes=list(self._nodes),
            brackets=list(self._brackets),
            thresholds=self.thresholds,
            stopped_reason=stopped_reason,
        )

    # -- 站位 ---------------------------------------------------------------
    def _stations(self) -> list[float]:
        cfg = self.config
        stations = [cfg.center_q1]
        step = cfg.station_step_ratio
        for k in range(1, cfg.max_stations + 1):
            for candidate in (cfg.center_q1 / step ** k, cfg.center_q1 * step ** k):
                if cfg.q1_min <= candidate <= cfg.q1_max:
                    if all(abs(candidate - s) / s > 0.05 for s in stations):
                        stations.append(candidate)
            if len(stations) >= cfg.max_stations:
                break
        return stations[: cfg.max_stations]

    def _station_base(self, q1: float, station_index: int) -> NodeRecord:
        cfg = self.config
        ratio = cfg.center_q2 / cfg.center_q1
        q2 = min(max(ratio * q1, cfg.q2_min), self._q2_ceiling(q1))
        return self._evaluate(q1, q2, purpose=f"station_base#{station_index}")

    # -- 单方向扫描 ---------------------------------------------------------
    def _scan_direction(self, base: NodeRecord, direction: str) -> None:
        cfg = self.config
        step = cfg.step_ratio if direction == "high" else 1.0 / cfg.step_ratio
        q2 = base.q2
        current = base
        for _ in range(cfg.max_expand_steps):
            # 单调性启发：沿 Q2 方向只会更深入某一侧极端流型，提前收手省节点。
            #   Q2 升高 → r=Q1/Q2 降低 → 栓塞（TUBING）
            #   Q2 降低 → r 升高      → 束流（THREADING）
            if direction == "low" and current.regime is FlowRegime.THREADING:
                break
            if direction == "high" and current.regime is FlowRegime.TUBING:
                break
            candidate = self._clamp_q2(base.q1, q2 * step)
            if direction == "high" and candidate <= q2 * 1.001:
                break  # 到达 Q2 上限或安全约束边界
            if direction == "low" and candidate >= q2 * 0.999:
                break
            node = self._evaluate(base.q1, candidate, purpose=f"expand_{direction}")
            if node.regime is not current.regime and node.regime is not FlowRegime.UNKNOWN:
                kind = classify_boundary_kind(current.regime, node.regime)
                self._maybe_bisect(base.q1, current, node, direction, kind)
            q2, current = candidate, node

    # -- 夹逼 ---------------------------------------------------------------
    def _maybe_bisect(
        self,
        station_q1: float,
        inside: NodeRecord,
        outside: NodeRecord,
        direction: str,
        kind: str,
    ) -> None:
        """对液滴窗口边界做几何二分；束流边界只记录粗区间。"""
        if kind not in {"droplet_upper", "droplet_lower"}:
            self._record_bracket(station_q1, direction, kind, inside, outside)
            return

        best_inside, best_outside = inside, outside
        rounds = 0
        while rounds < self.config.bisection_rounds:
            low, high = sorted((best_inside.q2, best_outside.q2))
            if high / low <= self.config.min_ratio_step:
                break
            mid = math.sqrt(low * high)
            candidate = self._clamp_q2(station_q1, mid)
            if min(abs(candidate - low), abs(candidate - high)) < 0.005:
                break
            node = self._evaluate(station_q1, candidate, purpose="bisect")
            if node.regime is best_inside.regime:
                best_inside = node
            else:
                best_outside = node
            rounds += 1

        self._record_bracket(station_q1, direction, kind, best_inside, best_outside)

    def _record_bracket(
        self,
        station_q1: float,
        direction: str,
        kind: str,
        inside: NodeRecord,
        outside: NodeRecord,
    ) -> None:
        boundary = math.sqrt(inside.q2 * outside.q2)
        width = math.sqrt(max(inside.q2, outside.q2) / max(min(inside.q2, outside.q2), 1e-9))
        outer_q2 = max(inside.q2, outside.q2)
        self._brackets.append(
            BoundaryBracket(
                station_q1=station_q1,
                kind=kind,
                direction=direction,
                inside=inside,
                outside=outside,
                boundary_q2=boundary,
                relative_width=width,
                at_domain_edge=abs(outer_q2 - self._q2_ceiling(station_q1)) < 1e-9,
            )
        )

    # -- 节点执行 -----------------------------------------------------------
    def _evaluate(self, q1: float, q2: float, *, purpose: str) -> NodeRecord:
        cfg = self.config
        if len(self._nodes) >= cfg.max_nodes:
            raise _BudgetExceeded
        q1 = max(cfg.q1_min, min(cfg.q1_max, q1))
        q2 = self._clamp_q2(q1, q2)
        features = self.measure(q1, q2, cfg.node_duration_s)
        regime, confidence = classify_regime(features, self.thresholds)
        node = NodeRecord(
            index=len(self._nodes) + 1,
            q1=q1,
            q2=q2,
            regime=regime,
            confidence=confidence,
            features=features,
            started_s=self._time_s,
            duration_s=cfg.node_duration_s,
            purpose=purpose,
        )
        self._time_s += cfg.startup_s + cfg.node_duration_s
        self._nodes.append(node)
        if self.progress is not None:
            self.progress(node)
        return node

    # -- 约束 ---------------------------------------------------------------
    def _q2_ceiling(self, q1: float) -> float:
        return min(self.config.q2_max, q1 / self.config.safety_q1_over_q2)

    def _clamp_q2(self, q1: float, q2: float) -> float:
        ceiling = self._q2_ceiling(q1)
        if ceiling < self.config.q2_min:
            raise RuntimeError(
                f"Q1={q1:g} 下 Q2 的安全上限 {ceiling:g} 小于下限 {self.config.q2_min:g}"
            )
        return min(max(q2, self.config.q2_min), ceiling)


def summarize_regimes(result: ScanResult) -> dict[FlowRegime, int]:
    counts: dict[FlowRegime, int] = {}
    for node in result.nodes:
        counts[node.regime] = counts.get(node.regime, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0].value))


def station_boundaries(result: ScanResult) -> dict[float, dict[str, BoundaryBracket]]:
    """按站位汇总 ``droplet_upper`` / ``droplet_lower`` 边界。"""
    table: dict[float, dict[str, BoundaryBracket]] = {}
    for bracket in result.brackets:
        if bracket.kind not in {"droplet_upper", "droplet_lower"}:
            continue
        entry = table.setdefault(bracket.station_q1, {})
        current = entry.get(bracket.kind)
        if current is None or bracket.relative_width < current.relative_width:
            entry[bracket.kind] = bracket
    return table
