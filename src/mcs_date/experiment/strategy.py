"""有界蜘蛛式扫描：从射流等稳定工作点探索，而不是验证预设的两条直线。"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

from .models import BoundaryBracket, NodeRecord, ScanConfig, ScanResult, classify_boundary_kind
from .phase import FlowRegime, NodeFeatures, RegimeThresholds, assess_formation, classify_regime

Measurement = Callable[[float, float, float], NodeFeatures]
ProgressCallback = Callable[[NodeRecord], None]


class _BudgetExceeded(RuntimeError):
    pass


class SpiderScanner:
    """粗扫 + 有限插点找窗 + 有效观测夹逼，未知点绝不当作窗口外证据。

    默认每节点观察180s。流型/窗口只由测量决定；测不到、安全域外、预算不足
    会作为缺口输出。硬件/测量异常向上传播，由执行器负责停机。
    """

    def __init__(self, measure: Measurement, config: ScanConfig | None = None,
                 thresholds: RegimeThresholds | None = None,
                 progress: ProgressCallback | None = None) -> None:
        self.measure = measure
        self.config = config or ScanConfig()
        self.config.validate()
        self.thresholds = thresholds or RegimeThresholds()
        self.progress = progress
        self._nodes: list[NodeRecord] = []
        self._brackets: list[BoundaryBracket] = []
        self._cache: dict[tuple[float, float], NodeRecord] = {}
        self._unresolved: list[dict[str, object]] = []
        self._time_s = 0.0
        self.target = FlowRegime(self.config.target_regime)
        self.boundary_prefix = "jetting" if self.target is FlowRegime.JETTING else "droplet"

    @property
    def node_count(self) -> int:
        """已成功记录的有效节点数；执行器仅在有节点时把预算不足转成partial。"""
        return len(self._nodes)

    def run(self) -> ScanResult:
        self._nodes, self._brackets, self._unresolved = [], [], []
        self._cache = {}
        self._time_s = 0.0
        stopped_reason = ""
        center: NodeRecord | None = None
        stations = self._stations()
        try:
            for station_index, q1 in enumerate(stations):
                if self._q2_ceiling(q1) < self.config.q2_min:
                    continue
                base = self._station_base(q1, station_index)
                if center is None:
                    center = base
                # 以给定的优质射流点为锚向两侧展开，不预设一定能到滴落区。
                directions = ("low", "high") if self.target is FlowRegime.JETTING else ("high", "low")
                for direction in directions:
                    self._scan_direction(base, direction)
        except _BudgetExceeded:
            stopped_reason = "达到节点或计划时间预算，未覆盖区域仍未知"
        if center is None:
            if stopped_reason:
                raise RuntimeError(f"预算不足以完成任何节点，未产生有效观测: {stopped_reason}")
            raise RuntimeError("扫描未产生任何节点")
        metadata = {
            "data_origin": "unspecified",
            "ratio_definition": "Q1_continuous_total / Q2_dispersed",
            "classification_basis": "feature_heuristic_not_validated_topology",
            "boundary_estimate": "geometric_midpoint_of_observed_interval_not_exact_threshold",
            "unresolved_intervals": self._unresolved,
            "station_status": self._station_status(stations, bool(stopped_reason)),
            "scan_target": self.target.value,
            "warning": "射流也可在下游形成液滴；默认探索射流到束流，不预设能到滴落。未发现边界不代表边界不存在。",
            "evolution_observables": ["mean_diameter_ratio", "diameter_cv", "breakup_distance_h"],
        }
        return ScanResult(self.config, center, list(self._nodes), list(self._brackets),
                          self.thresholds, metadata, stopped_reason)

    def _stations(self) -> list[float]:
        cfg = self.config
        stations = [cfg.center_q1]
        for k in range(1, cfg.max_stations + 1):
            for candidate in (cfg.center_q1 / cfg.station_step_ratio ** k,
                              cfg.center_q1 * cfg.station_step_ratio ** k):
                if cfg.q1_min <= candidate <= cfg.q1_max:
                    if all(abs(candidate - s) / s > 0.05 for s in stations):
                        stations.append(candidate)
            if len(stations) >= cfg.max_stations:
                break
        return stations[:cfg.max_stations]

    def _station_base(self, q1: float, station_index: int) -> NodeRecord:
        q2 = self._clamp_q2(q1, self.config.center_q2 / self.config.center_q1 * q1)
        return self._evaluate(q1, q2, purpose=f"station_base#{station_index}")

    def _scan_direction(self, base: NodeRecord, direction: str) -> None:
        cfg = self.config
        step = cfg.step_ratio if direction == "high" else 1 / cfg.step_ratio
        current = base
        for _ in range(cfg.max_expand_steps):
            candidate = self._clamp_q2(base.q1, current.q2 * step)
            if math.isclose(candidate, current.q2, rel_tol=1e-6):
                break
            node = self._evaluate(base.q1, candidate, purpose=f"expand_{direction}")
            self._inspect_interval(current, node, direction, cfg.window_search_rounds)
            current = node

    def _inspect_interval(self, a: NodeRecord, b: NodeRecord, direction: str, depth: int) -> None:
        if FlowRegime.UNKNOWN in (a.regime, b.regime):
            self._unresolved.append({"q1": a.q1, "indices": [a.index, b.index],
                                     "reason": "unknown_observation"})
            return
        a_drop, b_drop = a.regime is self.target, b.regime is self.target
        if a_drop != b_drop:
            self._maybe_bisect(a if a_drop else b, b if a_drop else a, direction)
            return
        if a_drop and b_drop:
            return
        # 即使两个粗节点都不滴落，也可能跨过中间窄窗（两侧甚至均为jetting）。
        lo, hi = sorted((a.q2, b.q2))
        if depth > 0 and hi / lo > self.config.min_ratio_step:
            mid = self._evaluate(a.q1, math.sqrt(lo * hi), purpose="window_search")
            self._inspect_interval(a, mid, direction, depth - 1)
            self._inspect_interval(mid, b, direction, depth - 1)
        elif a.regime is not b.regime:
            kind = classify_boundary_kind(a.regime, b.regime)
            if kind == "thread_onset":
                self._record_bracket(a, b, direction, kind)

    def _maybe_bisect(self, inside: NodeRecord, outside: NodeRecord, direction: str) -> None:
        resolution = "bracketed"
        for _ in range(self.config.bisection_rounds):
            lo, hi = sorted((inside.q2, outside.q2))
            if hi / lo <= self.config.min_ratio_step:
                break
            node = self._evaluate(inside.q1, math.sqrt(lo * hi), purpose="bisect")
            if node.regime is FlowRegime.UNKNOWN:
                # 保留有效两端的粗区间，但不得将未知中点推成“非滴落”。
                resolution = "uncertain"
                self._unresolved.append({"q1": inside.q1,
                                         "indices": [inside.index, node.index, outside.index],
                                         "reason": "unknown_during_refinement"})
                break
            if node.regime is self.target:
                inside = node
            else:
                outside = node
        side = "lower" if outside.q2 < inside.q2 else "upper"
        kind = f"{self.boundary_prefix}_{side}"
        self._record_bracket(inside, outside, direction, kind, resolution)

    def _record_bracket(self, inside: NodeRecord, outside: NodeRecord, direction: str,
                        kind: str, resolution: str = "bracketed") -> None:
        lo, hi = sorted((inside.q2, outside.q2))
        if any(b.kind == kind and {b.inside.index, b.outside.index} == {inside.index, outside.index}
               for b in self._brackets):
            return
        self._brackets.append(BoundaryBracket(
            station_q1=inside.q1, kind=kind, direction=direction, inside=inside, outside=outside,
            boundary_q2=math.sqrt(lo * hi), relative_width=hi / lo,
            at_domain_edge=math.isclose(hi, self._q2_ceiling(inside.q1), rel_tol=1e-9)
            or math.isclose(lo, self.config.q2_min, rel_tol=1e-9), resolution=resolution,
        ))

    def _evaluate(self, q1: float, q2: float, *, purpose: str) -> NodeRecord:
        key = (round(q1, 10), round(q2, 10))
        if key in self._cache:
            return self._cache[key]
        cfg = self.config
        if (len(self._nodes) >= cfg.max_nodes
                or self._time_s + cfg.startup_s + cfg.node_duration_s > cfg.max_duration_s):
            raise _BudgetExceeded
        features = self.measure(q1, q2, cfg.node_duration_s)
        if not all(math.isfinite(v) for v in (features.q1, features.q2)):
            raise ValueError("测量返回非有限流量")
        if not math.isclose(features.q1, q1, rel_tol=0.005, abs_tol=0.01) or not math.isclose(features.q2, q2, rel_tol=0.005, abs_tol=0.01):
            raise ValueError("测量流量与请求节点不匹配，不能用于该点的边界")
        if not math.isclose(features.window_s, cfg.node_duration_s, rel_tol=1e-6):
            raise ValueError("测量窗口与节点时长不一致")
        regime, confidence = classify_regime(features, self.thresholds)
        origin = self._nodes[0].features if self._nodes else features
        size_change = (features.mean_diameter_ratio / origin.mean_diameter_ratio - 1
                       if origin.mean_diameter_ratio > 0 and features.mean_diameter_ratio > 0
                       and assess_formation(origin, self.thresholds)["droplets_observed"]
                       and assess_formation(features, self.thresholds)["droplets_observed"] else None)
        shift = (features.breakup_distance_h - origin.breakup_distance_h
                 if features.breakup_distance_h is not None and origin.breakup_distance_h is not None else None)
        stage = ("unbroken_in_observation_window" if regime is FlowRegime.THREADING else
                 "unknown" if regime is FlowRegime.UNKNOWN else
                 "breakup_migration" if shift is not None and abs(shift) >= 1.0 else
                 "size_change" if size_change is not None and abs(size_change) >= 0.10 else
                 "near_anchor")
        features = replace(features, extra={**features.extra,
                           "diameter_change_from_anchor": size_change,
                           "breakup_shift_h_from_anchor": shift, "evolution_stage": stage,
                           "stage_thresholds_are_heuristic": True})
        node = NodeRecord(len(self._nodes) + 1, q1, q2, regime, confidence, features,
                          self._time_s + cfg.startup_s, cfg.node_duration_s, purpose)
        self._time_s += cfg.startup_s + cfg.node_duration_s
        self._nodes.append(node)
        self._cache[key] = node
        if self.progress is not None:
            self.progress(node)
        return node

    def _station_status(self, stations: list[float], budget_exhausted: bool) -> list[dict[str, object]]:
        statuses: list[dict[str, object]] = []
        for q1 in stations:
            nodes = [n for n in self._nodes if math.isclose(n.q1, q1)]
            droplets = [n for n in nodes if n.regime is self.target]
            entries = [b for b in self._brackets if math.isclose(b.station_q1, q1)]
            sides: dict[str, str] = {}
            for kind in (f"{self.boundary_prefix}_lower", f"{self.boundary_prefix}_upper"):
                found = [b for b in entries if b.kind == kind]
                if any(b.resolution == "bracketed" for b in found):
                    status = "bracketed"
                elif found:
                    status = "uncertain"
                elif not nodes:
                    status = "not_scanned" if self._q2_ceiling(q1) >= self.config.q2_min else "outside_safe_domain"
                elif droplets and (kind == f"{self.boundary_prefix}_upper" and math.isclose(max(n.q2 for n in droplets), self._q2_ceiling(q1))
                                   or kind == f"{self.boundary_prefix}_lower" and math.isclose(min(n.q2 for n in droplets), self.config.q2_min)):
                    status = "censored_by_safe_domain"
                elif budget_exhausted:
                    status = "budget_exhausted"
                elif any(n.regime is FlowRegime.UNKNOWN for n in nodes):
                    status = "unknown_observations"
                else:
                    status = "not_observed_at_search_resolution"
                sides[kind] = status
            statuses.append({"q1": q1, "nodes": len(nodes), "target_observed": bool(droplets),
                             "target_regime": self.target.value,
                             "droplet_observed": any(n.regime is FlowRegime.DRIPPING for n in nodes),
                             "safe_q2_range": [self.config.q2_min, self._q2_ceiling(q1)], **sides})
        return statuses

    def _q2_ceiling(self, q1: float) -> float:
        return min(self.config.q2_max, q1 / self.config.safety_q1_over_q2)

    def _clamp_q2(self, q1: float, q2: float) -> float:
        ceiling = self._q2_ceiling(q1)
        if ceiling < self.config.q2_min:
            raise ValueError("该Q1站位没有安全可执行的Q2")
        return min(max(q2, self.config.q2_min), ceiling)


def summarize_regimes(result: ScanResult) -> dict[FlowRegime, int]:
    counts: dict[FlowRegime, int] = {}
    for node in result.nodes:
        counts[node.regime] = counts.get(node.regime, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0].value))


def station_boundaries(result: ScanResult) -> dict[float, dict[str, BoundaryBracket]]:
    """仅返回无未知中点的有效夹逼；未找见的边界不补齐、不外推。"""
    table: dict[float, dict[str, BoundaryBracket]] = {}
    for bracket in result.brackets:
        if bracket.kind not in {"droplet_upper", "droplet_lower", "jetting_lower", "jetting_upper"} or bracket.resolution != "bracketed":
            continue
        entry = table.setdefault(bracket.station_q1, {})
        current = entry.get(bracket.kind)
        if current is None or bracket.relative_width < current.relative_width:
            entry[bracket.kind] = bracket
    return table
