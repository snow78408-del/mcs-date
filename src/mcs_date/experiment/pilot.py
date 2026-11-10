"""小规模射流→束流试验方案：围绕操作者认可的射流锚点走四条有限射线。

设计口径（用户方案）：
- 锚点由操作者选定，不把默认流量当成“最佳点”；
- 先观察液滴尺寸变化，再观察断裂位置向下游移动，最后在观测域内看到不破裂束流；
- R = Q1/Q2 与 S = Q1+Q2 各向两侧走有限步（默认 25%），每节点约 3 分钟；
- 回测锚点检查漂移；安全域/流量域截断的射线显式标注 censored，不伪造边界。

这里只生成执行清单并记录观测；演化顺序不被强制，未看到边界不代表不存在。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .models import NodeRecord, ScanConfig, ScanResult
from .phase import FlowRegime, RegimeThresholds
from .strategy import Measurement, ProgressCallback, SpiderScanner, _BudgetExceeded

RAYS: tuple[str, ...] = ("R_down", "R_up", "S_down", "S_up")


@dataclass(frozen=True)
class PilotConfig:
    """四条射线的小规模试验配置；默认 13–14 个节点。"""

    anchor_q1: float = 90.0
    anchor_q2: float = 20.0
    node_duration_s: float = 180.0
    startup_s: float = 30.0
    step_fraction: float = 0.25
    steps_per_ray: int = 3
    q1_min: float = 2.0
    q1_max: float = 200.0
    q2_min: float = 0.2
    q2_max: float = 100.0
    safety_q1_over_q2: float = 2.0
    revisit_anchor: bool = True
    target_regime: str = "jetting"

    def validate(self) -> None:
        if self.target_regime not in {"jetting", "dripping"}:
            raise ValueError("target_regime必须为jetting或dripping")
        for name in ("anchor_q1", "anchor_q2", "node_duration_s", "step_fraction",
                     "q1_min", "q1_max", "q2_min", "q2_max", "safety_q1_over_q2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if not math.isfinite(self.startup_s) or self.startup_s < 0:
            raise ValueError("startup_s 必须为有限非负数")
        if isinstance(self.steps_per_ray, bool) or not isinstance(self.steps_per_ray, int) or self.steps_per_ray < 1:
            raise ValueError("steps_per_ray 必须是 >= 1 的整数")
        if not 0.05 <= self.step_fraction <= 0.5:
            raise ValueError("step_fraction 应在 5%–50%（方案默认 20%–30%）")
        if self.q1_min > self.q1_max or self.q2_min > self.q2_max:
            raise ValueError("流量下限不能高于上限")
        if not self.q1_min <= self.anchor_q1 <= self.q1_max or not self.q2_min <= self.anchor_q2 <= self.q2_max:
            raise ValueError("锚点必须在配置的流量范围内")
        if self.safety_q1_over_q2 < 2:
            raise ValueError("不得放宽泵的固定安全约束 Q1 >= 2*Q2")
        if self.anchor_q1 < self.safety_q1_over_q2 * self.anchor_q2:
            raise ValueError("锚点违反 Q1 >= 2*Q2 安全约束")

    @property
    def estimated_duration_s(self) -> float:
        planned = self.planned_nodes
        return planned * self.node_duration_s + (planned - 1) * self.startup_s

    @property
    def planned_nodes(self) -> int:
        return len(build_pilot_points(self)[0])


@dataclass(frozen=True)
class PilotPoint:
    q1: float
    q2: float
    ray: str
    step: int

    @property
    def oil_water_ratio(self) -> float:
        return self.q1 / self.q2

    @property
    def total_flow(self) -> float:
        return self.q1 + self.q2


def _points_for(config: PilotConfig) -> tuple[list[PilotPoint], dict[str, dict[str, object]]]:
    points = [PilotPoint(config.anchor_q1, config.anchor_q2, "anchor", 0)]
    state = {"R": config.anchor_q1 / config.anchor_q2, "S": config.anchor_q1 + config.anchor_q2}
    status: dict[str, dict[str, object]] = {}
    growth = 1.0 + config.step_fraction

    for ray in RAYS:
        taken = 0
        censored: str | None = None
        for step in range(1, config.steps_per_ray + 1):
            factor = growth ** step
            ratio = state["R"] * factor if ray == "R_up" else state["R"] / factor if ray == "R_down" else state["R"]
            total = state["S"] * factor if ray == "S_up" else state["S"] / factor if ray == "S_down" else state["S"]
            q1 = total * ratio / (1.0 + ratio)
            q2 = total / (1.0 + ratio)
            if q1 < config.q1_min:
                censored = "q1_min"
            elif q2 < config.q2_min:
                censored = "q2_min"
            elif q1 > config.q1_max:
                censored = "q1_max"
            elif q2 > config.q2_max:
                censored = "q2_max"
            elif q1 < config.safety_q1_over_q2 * q2:
                censored = "safety_q1_over_q2"
            if censored is not None:
                break
            candidate = PilotPoint(round(q1, 9), round(q2, 9), ray, step)
            if any(abs(math.log(candidate.q1 / p.q1)) < 0.02 and abs(math.log(candidate.q2 / p.q2)) < 0.02
                   for p in points):
                continue
            points.append(candidate)
            taken += 1
        status[ray] = {"points": taken, "censored_by": censored,
                       "last_q1": points[-1].q1 if taken else None,
                       "last_q2": points[-1].q2 if taken else None}

    if config.revisit_anchor:
        points.append(PilotPoint(config.anchor_q1, config.anchor_q2, "anchor_revisit", 0))
    return points, status


def build_pilot_points(config: PilotConfig) -> tuple[list[PilotPoint], dict[str, dict[str, object]]]:
    """返回执行清单与每条射线的截断原因；截断不是边界证据。"""
    config.validate()
    return _points_for(config)


def pilot_scan_config(config: PilotConfig, planned_nodes: int) -> ScanConfig:
    """把试验清单映射到执行器/结果模型需要的 ScanConfig 字段。"""
    return ScanConfig(
        center_q1=config.anchor_q1,
        center_q2=config.anchor_q2,
        node_duration_s=config.node_duration_s,
        startup_s=config.startup_s,
        step_ratio=1.0 + config.step_fraction,
        min_ratio_step=1.0 + config.step_fraction / 2.0,
        bisection_rounds=0,
        max_expand_steps=1,
        q1_min=config.q1_min,
        q1_max=config.q1_max,
        q2_min=config.q2_min,
        q2_max=config.q2_max,
        station_step_ratio=1.0 + config.step_fraction,
        max_stations=1,
        max_nodes=planned_nodes,
        safety_q1_over_q2=config.safety_q1_over_q2,
        window_search_rounds=0,
        max_duration_s=planned_nodes * (config.startup_s + config.node_duration_s) + 60.0,
        target_regime=config.target_regime,
    )


class PilotScanner(SpiderScanner):
    """固定清单扫描器：复用分类/演化字段，不插入自适应补点。"""

    def __init__(self, measure: Measurement, config: PilotConfig | None = None,
                 thresholds: RegimeThresholds | None = None,
                 progress: ProgressCallback | None = None) -> None:
        self.pilot_config = config or PilotConfig()
        self.pilot_config.validate()
        self.points, self.ray_status = build_pilot_points(self.pilot_config)
        scan_config = pilot_scan_config(self.pilot_config, len(self.points))
        super().__init__(measure, scan_config, thresholds, progress)
        self._crossings: list[dict[str, object]] = []

    def run(self) -> ScanResult:
        self._nodes, self._brackets, self._unresolved = [], [], []
        self._cache = {}
        self._time_s = 0.0
        self._crossings = []
        stopped_reason = ""
        center = None
        previous: dict[str, object] = {}
        try:
            for point in self.points:
                if point.ray == "anchor_revisit":
                    # 回测必须重新观测同一流量，检查漂移/滞回；不能复用锚点缓存。
                    self._cache.pop((round(point.q1, 10), round(point.q2, 10)), None)
                node = self._evaluate(point.q1, point.q2, purpose=f"pilot:{point.ray}#{point.step}")
                if center is None:
                    center = node
                before = previous.get(point.ray)
                if before is not None:
                    self._inspect_pair(before, node, point.ray)
                previous[point.ray] = node
                if self.progress is not None:
                    self.progress(node)
        except _BudgetExceeded:
            stopped_reason = "达到节点或计划时间预算，未覆盖区域仍未知"
        if center is None:
            if stopped_reason:
                raise RuntimeError(f"预算不足以完成任何节点，未产生有效观测: {stopped_reason}")
            raise RuntimeError("试验清单未产生任何节点")
        metadata = {
            "data_origin": "unspecified",
            "experiment_design": "pilot_four_rays_not_adaptive_window_search",
            "scan_target": self.target.value,
            "ratio_definition": "Q1_continuous_total / Q2_dispersed",
            "pilot_plan": [{"q1": p.q1, "q2": p.q2, "ray": p.ray, "step": p.step,
                            "oil_water_ratio": p.oil_water_ratio, "total_flow": p.total_flow}
                           for p in self.points],
            "ray_status": self.ray_status,
            "ray_crossings": self._crossings,
            "unresolved_intervals": self._unresolved,
            "planned_nodes": len(self.points),
            "estimated_duration_s": self.pilot_config.estimated_duration_s,
            "classification_basis": "feature_heuristic_not_validated_topology",
            "boundary_estimate": "first_observed_target_to_thread_transition_along_a_ray_not_exact",
            "warning": "未看到束流不代表不存在；censored 射线只是被安全域/流量域截断，不是边界证据。",
            "evolution_observables": ["mean_diameter_ratio", "diameter_cv", "breakup_distance_h"],
        }
        return ScanResult(self.config, center, list(self._nodes), list(self._brackets),
                          self.thresholds, metadata, stopped_reason)

    def _inspect_pair(self, before, after, ray: str) -> None:
        if FlowRegime.UNKNOWN in (before.regime, after.regime):
            self._unresolved.append({"ray": ray, "indices": [before.index, after.index],
                                     "reason": "unknown_observation"})
            return
        before_is_target = before.regime is self.target
        after_is_target = after.regime is self.target
        if before_is_target == after_is_target:
            return
        inside, outside = (before, after) if before_is_target else (after, before)
        if outside.regime is not FlowRegime.THREADING:
            self._unresolved.append({"ray": ray, "indices": [before.index, after.index],
                                     "reason": f"outside_is_{outside.regime.value}_not_thread"})
            return
        self._crossings.append({
            "ray": ray, "inside_index": inside.index, "outside_index": outside.index,
            "inside_q1": inside.q1, "inside_q2": inside.q2,
            "outside_q1": outside.q1, "outside_q2": outside.q2,
            "resolution": "bracketed",
            "midpoint_q1": math.sqrt(inside.q1 * outside.q1),
            "midpoint_q2": math.sqrt(inside.q2 * outside.q2),
        })


def run_offline_pilot(config: PilotConfig | None = None, simulator=None,
                      logger: Callable[[str], None] | None = None) -> ScanResult:
    """只用仿真跑通方案；结果不代表芯片真实物理。"""
    from ..pump.service import PumpController, SimulatedPumpBackend
    from .chip import ChipModelConfig, ChipSimulator
    from .runner import ExperimentRunner, SimulatedMeasurement, VirtualClock, _attach_cleanup_error

    pilot_config = config or PilotConfig()
    pilot_config.validate()
    planned = len(build_pilot_points(pilot_config)[0])
    scan_config = pilot_scan_config(pilot_config, planned)
    sim = simulator or ChipSimulator(config=ChipModelConfig(
        operating_mode="jetting" if pilot_config.target_regime == "jetting" else "ratio_window",
        jetting_center_q1=pilot_config.anchor_q1,
        jetting_center_q2=pilot_config.anchor_q2,
    ))
    pump = PumpController(SimulatedPumpBackend(flow_quantum=None))
    sink = logger or (lambda _msg: None)
    runner = ExperimentRunner(pump, SimulatedMeasurement(sim), VirtualClock(), sink,
                              scan_config=scan_config)
    scanner = PilotScanner(runner._measure_node, pilot_config, progress=lambda node: sink(
        f"[PILOT] {node.purpose} Q1={node.q1:.2f} Q2={node.q2:.2f} -> {node.regime.value}"))
    runner.scanner = scanner
    primary: BaseException | None = None
    try:
        pump.connect()
        result = runner.run()
        result.metadata.update({"data_origin": "simulation", "source": "simulation",
                                "model_kind": sim.config.operating_mode + "_prior",
                                "physical_boundaries_verified": False})
        return result
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            pump.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            _attach_cleanup_error(primary, cleanup)
