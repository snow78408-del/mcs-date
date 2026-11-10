"""实验执行器：把泵、测量源、时钟和蜘蛛式扫描串起来。

- 真实硬件：``pump=SerialPumpBackend``、``measurement=VisionMeasurement``、
  ``clock=RealClock``；
- 离线演示：``pump=SimulatedPumpBackend``、``measurement=SimulatedMeasurement``、
  ``clock=VirtualClock``（整场实验瞬间跑完，便于生成动画和回归测试）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..pump.service import PumpController, PumpError
from .chip import ChipSimulator
from .models import ScanConfig, ScanResult
from .phase import NodeFeatures
from .strategy import SpiderScanner

Logger = Callable[[str], None]


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class RealClock:
    """真实时钟：按墙钟等待。"""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class VirtualClock:
    """虚拟时钟：``sleep`` 只推进内部时间，不真的等待。"""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._t += float(seconds)


class MeasurementSource(Protocol):
    def features(self, q1: float, q2: float, *, window_s: float, clock: Clock) -> NodeFeatures:
        """在给定流量下观测 ``window_s`` 秒并返回特征。"""


@dataclass
class SimulatedMeasurement:
    """离线测量源：直接由芯片模型生成特征。"""

    simulator: ChipSimulator = field(default_factory=ChipSimulator)

    def features(self, q1: float, q2: float, *, window_s: float, clock: Clock) -> NodeFeatures:
        return self.simulator.features(q1, q2, window_s=window_s)


@dataclass
class ExperimentRunner:
    """执行一次完整扫描实验并保证异常时安全停泵。"""

    pump: PumpController
    measurement: MeasurementSource
    clock: Clock = field(default_factory=RealClock)
    logger: Logger = field(default_factory=lambda: (lambda _msg: None))
    scanner: SpiderScanner | None = None
    scan_config: ScanConfig = field(default_factory=ScanConfig)

    def build_scanner(self) -> SpiderScanner:
        return SpiderScanner(
            measure=self._measure_node,
            config=self.scan_config,
            progress=lambda node: self.logger(
                f"[NODE {node.index:03d}] Q1={node.q1:7.2f} Q2={node.q2:7.2f} "
                f"λ={node.flow_ratio:5.3f} -> {node.regime.value:10s} "
                f"(conf {node.confidence:.2f}, {node.purpose})"
            ),
        )

    def _measure_node(self, q1: float, q2: float, window_s: float) -> NodeFeatures:
        try:
            self.pump.apply(q1, q2)
        except PumpError:
            self.logger(f"[SAFETY] 施加 Q1={q1:g} Q2={q2:g} 失败，停泵并中止")
            self.pump.stop()
            raise
        self.clock.sleep(self.scan_config.startup_s)
        return self.measurement.features(q1, q2, window_s=window_s, clock=self.clock)

    def run(self) -> ScanResult:
        self.scanner = self.build_scanner()
        self.logger(
            f"[SCAN] 开始：中心 Q1={self.scan_config.center_q1:g} "
            f"Q2={self.scan_config.center_q2:g}，节点 {self.scan_config.node_duration_s:g}s"
        )
        try:
            result = self.scanner.run()
        finally:
            self.pump.stop()
            self.logger("[SAFETY] 扫描结束，已停泵")
        self.logger(
            f"[SCAN] 完成：{len(result.nodes)} 个节点，"
            f"{len(result.brackets)} 个边界区间，耗时 {result.total_duration_s / 3600.0:.2f} h（计划）"
        )
        return result


def run_offline_scan(
    scan_config: ScanConfig | None = None,
    simulator: ChipSimulator | None = None,
    logger: Logger | None = None,
) -> ScanResult:
    """离线跑一次完整扫描（虚拟时钟，秒级返回）。"""
    from ..pump.service import SimulatedPumpBackend

    cfg = scan_config or ScanConfig()
    sim = simulator or ChipSimulator()
    pump = PumpController(SimulatedPumpBackend())
    pump.connect()
    runner = ExperimentRunner(
        pump=pump,
        measurement=SimulatedMeasurement(simulator=sim),
        clock=VirtualClock(),
        logger=logger or (lambda _msg: None),
        scan_config=cfg,
    )
    try:
        return runner.run()
    finally:
        pump.close()
