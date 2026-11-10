"""执行器将泵、测量源与扫描串联。真实视觉MeasurementSource仍须标定接入。

scan CLI只运行仿真；不能把不存在的VisionMeasurement当已完成闭环。
自定义真实采样源须使用单调时钟、事件去重、节点同步和可超时的采集调用。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from ..pump.service import PumpController
from .chip import ChipSimulator
from .models import ScanConfig, ScanResult
from .phase import NodeFeatures
from .strategy import SpiderScanner, _BudgetExceeded

Logger = Callable[[str], None]


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class VirtualClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._t += float(seconds)


class MeasurementSource(Protocol):
    def features(self, q1: float, q2: float, *, window_s: float, clock: Clock) -> NodeFeatures:
        """必须实际观测整个window_s；I/O必须有限超时，不能返回旧帧冒充本节点。"""


@dataclass
class SimulatedMeasurement:
    simulator: ChipSimulator = field(default_factory=ChipSimulator)

    def features(self, q1: float, q2: float, *, window_s: float, clock: Clock) -> NodeFeatures:
        features = self.simulator.features(q1, q2, window_s=window_s)
        clock.sleep(window_s)
        return features


def _attach_cleanup_error(primary: BaseException, cleanup: BaseException) -> None:
    message = f"停机/资源释放另行失败，设备状态未确认: {cleanup!r}"
    if hasattr(primary, "add_note"):
        primary.add_note(message)
    else:  # Python3.10仍保留清理故障供日志/调用者检查。
        setattr(primary, "cleanup_error", message)


@dataclass
class ExperimentRunner:
    pump: PumpController
    measurement: MeasurementSource
    clock: Clock = field(default_factory=RealClock)
    logger: Logger = field(default_factory=lambda: (lambda _msg: None))
    scanner: SpiderScanner | None = None
    scan_config: ScanConfig = field(default_factory=ScanConfig)
    _deadline: float = field(default=float("inf"), init=False)

    def build_scanner(self) -> SpiderScanner:
        if self.scanner is not None:
            return self.scanner
        return SpiderScanner(
            measure=self._measure_node, config=self.scan_config,
            progress=lambda node: self.logger(
                f"[NODE {node.index:03d}] Q1={node.q1:.2f} Q2={node.q2:.2f} "
                f"r={node.flow_ratio:.3f} -> {node.regime.value} ({node.purpose})"),
        )

    def _deadline_hit(self, required: float) -> None:
        """节点开始前预算不足：已采到节点则交回partial，否则明确失败。

        已完成的节点必须能持久化；预算不足不能让整场结果消失，也不能让
        未开始的节点伪装成有效观测。
        """
        if self.clock.now() + required <= self._deadline:
            return
        completed = self.scanner.node_count if self.scanner is not None else 0
        message = f"剩余执行时间不足以完整观测本节点（已完成{completed}节点），停止扫描并停泵"
        if completed > 0:
            raise _BudgetExceeded(message)
        raise TimeoutError(message)

    def _measure_node(self, q1: float, q2: float, window_s: float) -> NodeFeatures:
        required = self.scan_config.startup_s + window_s
        self._deadline_hit(required)
        actual = self.pump.apply(q1, q2)
        self._deadline_hit(required)
        self.clock.sleep(self.scan_config.startup_s)
        observe_start = self.clock.now()
        features = self.measurement.features(q1, q2, window_s=window_s, clock=self.clock)
        elapsed = self.clock.now() - observe_start
        if elapsed + 1e-6 < window_s:
            raise ValueError("测量源未完成要求的观测窗口，拒绝记录为有效节点")
        if self.clock.now() > self._deadline + 1e-6:
            raise TimeoutError("采样源超过时间预算，结果不能当正常完成")
        # 参数回读不等于物理流量传感器；保留回读以便事后核查。
        from dataclasses import replace
        return replace(features, extra={**features.extra,
                       "parameter_readback_q1": actual[0], "parameter_readback_q2": actual[1],
                       "observe_start_monotonic": observe_start,
                       "observe_end_monotonic": self.clock.now()})

    def run(self) -> ScanResult:
        self.scan_config.validate()
        started = self.clock.now()
        self._deadline = started + self.scan_config.max_duration_s
        self.scanner = self.build_scanner()
        self.logger(f"[SCAN] 从 {self.scan_config.center_q1:g}/{self.scan_config.center_q2:g} 开始探索")
        primary: BaseException | None = None
        try:
            result = self.scanner.run()
        except BaseException as exc:
            primary = exc
            self.logger(f"[ERROR] 扫描失败，非正常完成: {exc}")
            raise
        finally:
            try:
                self.pump.stop()
            except BaseException as cleanup:
                self.logger(f"[SAFETY][ERROR] 停机未确认！需要检查设备/独立急停: {cleanup}")
                if primary is None:
                    raise
                _attach_cleanup_error(primary, cleanup)
            else:
                self.logger("[SAFETY] 停机调用已成功返回（硬件后端要求RSE确认）")
        result.metadata.update({"actual_elapsed_s": self.clock.now() - started,
                                "clock_kind": type(self.clock).__name__,
                                "status": "partial" if result.stopped_reason else "completed",
                                "stop_call_succeeded": True})
        self.logger(f"[SCAN] {result.metadata['status']}: {len(result.nodes)} 节点，计划 {result.total_duration_s/3600:.2f}h")
        return result


def run_offline_scan(scan_config: ScanConfig | None = None, simulator: ChipSimulator | None = None,
                     logger: Logger | None = None) -> ScanResult:
    """只跑仿真；它找出的窗口仅重建模型，不证明本芯片真实物理。"""
    from ..pump.service import SimulatedPumpBackend
    from .chip import ChipModelConfig

    cfg = scan_config or ScanConfig()
    sim = simulator or ChipSimulator(config=ChipModelConfig(
        operating_mode="jetting" if cfg.target_regime == "jetting" else "ratio_window",
        jetting_center_q1=cfg.center_q1, jetting_center_q2=cfg.center_q2))
    pump = PumpController(SimulatedPumpBackend(flow_quantum=None))
    runner = ExperimentRunner(pump, SimulatedMeasurement(sim), VirtualClock(),
                              logger or (lambda _msg: None), scan_config=cfg)
    primary: BaseException | None = None
    try:
        pump.connect()
        result = runner.run()
        result.metadata.update({"data_origin": "simulation", "model_kind": sim.config.operating_mode + "_prior",
                                "model_config": asdict(sim.config), "random_seed": sim.seed,
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
