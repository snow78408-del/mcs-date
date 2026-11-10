"""TS 注射泵控制层（CH1=连续油相，CH2=分散水相）。

分两层：
- :class:`SerialPumpBackend` 真实串口后端，按 TS 1200/8/E/1 规约写
  WSS/WSP/WSE 并用 RSS/RSP/RSE 回读校验（逻辑移植自 MCS_QT
  ``backend/pump_hardware/service.py``，保留关键安全语义）；
- :class:`SimulatedPumpBackend` 纯仿真后端，离线跑扫描实验时替代硬件。

两者都实现 :class:`PumpBackend`，实验层只依赖这个协议。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from . import protocol
from .client import PumpClient, PumpClientError
from .config import PumpHardwareConfig, SerialConfig
from .invariants import STRICT_Q1_Q2_RATIO, q1_is_at_least_twice_q2
from .models import ChannelParams, RunState, SystemSetup


class PumpError(RuntimeError):
    """泵控制失败。"""


class PumpSafetyError(PumpError):
    """请求违反固定安全约束（如 Q1 < 2*Q2）。"""


class PumpBackend(Protocol):
    """实验层使用的泵后端最小协议。"""

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def apply(self, q1: float, q2: float) -> None:
        """写入并启动两通道流量（µL/min）。"""

    def read(self) -> tuple[float, float]:
        """回读两通道流量（µL/min）。"""

    def stop(self) -> None: ...

    @property
    def connected(self) -> bool: ...


# ---------------------------------------------------------------------------
# 流路参数换算（与 MCS_QT 的 WSP 生成规则一致）
# ---------------------------------------------------------------------------

_VOLUME_UNIT_TO_UL = {1: 0.001, 2: 0.01, 3: 0.1, 4: 1.0, 5: 10.0}
_TIME_UNIT_TO_MIN = {1: 1.0 / 600.0, 2: 0.1, 3: 6.0}
_DEFAULT_SYRINGE_CODE = 0x18
_DEFAULT_DISPENSE_VALUE = 1000
_DEFAULT_DISPENSE_UNIT = 4
_DEFAULT_INFUSE_TIME_UNIT = 2
_DEFAULT_WITHDRAW_TIME_VALUE = 100
_DEFAULT_WITHDRAW_TIME_UNIT = 1
_DEFAULT_REPEAT_COUNT = 10
_DEFAULT_INTERVAL_VALUE = 50
_MIN_INFUSE_TIME_VALUE = 10
_MAX_PARAM_VALUE = 9999


def flow_from_channel_params(params: ChannelParams) -> float:
    """把 RSP 回读参数换算成 µL/min。"""
    volume_ul = float(params.dispense_value) * _VOLUME_UNIT_TO_UL[int(params.dispense_unit)]
    time_min = float(params.infuse_time_value) * _TIME_UNIT_TO_MIN[int(params.infuse_time_unit)]
    if time_min <= 0.0:
        raise PumpError("回读灌注时间非正")
    return volume_ul / time_min


def _infuse_time_value(dispense_value: int, dispense_unit: int, infuse_unit: int, flow: float) -> int:
    volume_ul = max(1, dispense_value) * _VOLUME_UNIT_TO_UL[dispense_unit]
    raw = volume_ul / max(flow, 1e-6) / _TIME_UNIT_TO_MIN[infuse_unit]
    return max(1, min(_MAX_PARAM_VALUE, int(round(raw))))


def channel_params_for_flow(channel: int, flow_ul_min: float) -> ChannelParams:
    """按目标流量生成 WSP 参数（沿用 MCS_QT 的默认注射器/单位档位）。"""
    dispense_value = _DEFAULT_DISPENSE_VALUE
    dispense_unit = _DEFAULT_DISPENSE_UNIT
    infuse_unit = _DEFAULT_INFUSE_TIME_UNIT
    if _infuse_time_value(dispense_value, dispense_unit, infuse_unit, flow_ul_min) > _MAX_PARAM_VALUE:
        dispense_value = max(1, min(
            _MAX_PARAM_VALUE,
            int(flow_ul_min * _MAX_PARAM_VALUE * _TIME_UNIT_TO_MIN[infuse_unit]
                / _VOLUME_UNIT_TO_UL[dispense_unit]),
        ))
    infuse_value = _infuse_time_value(dispense_value, dispense_unit, infuse_unit, flow_ul_min)
    if infuse_value < _MIN_INFUSE_TIME_VALUE:
        dispense_value = max(1, min(
            _MAX_PARAM_VALUE,
            int(round(flow_ul_min * _MIN_INFUSE_TIME_VALUE * _TIME_UNIT_TO_MIN[infuse_unit]
                      / _VOLUME_UNIT_TO_UL[dispense_unit])),
        ))
        infuse_value = max(
            _MIN_INFUSE_TIME_VALUE,
            _infuse_time_value(dispense_value, dispense_unit, infuse_unit, flow_ul_min),
        )
    return ChannelParams(
        channel=channel,
        mode=1,
        syringe_code=_DEFAULT_SYRINGE_CODE,
        dispense_value=dispense_value,
        dispense_unit=dispense_unit,
        infuse_time_value=infuse_value,
        infuse_time_unit=infuse_unit,
        withdraw_time_value=_DEFAULT_WITHDRAW_TIME_VALUE,
        withdraw_time_unit=_DEFAULT_WITHDRAW_TIME_UNIT,
        repeat_count=_DEFAULT_REPEAT_COUNT,
        interval_value=_DEFAULT_INTERVAL_VALUE,
    )


# ---------------------------------------------------------------------------
# 仿真后端
# ---------------------------------------------------------------------------


@dataclass
class SimulatedPumpBackend:
    """无硬件时替代泵：记录设定值并返回一致回读。"""

    max_flow_ul_min: float = 200.0
    flow_quantum: float = 0.01
    log: list[str] = field(default_factory=list)
    _q1: float = 0.0
    _q2: float = 0.0
    _connected: bool = False

    def connect(self) -> None:
        self._connected = True
        self.log.append("sim connect")

    def close(self) -> None:
        self._q1 = self._q2 = 0.0
        self._connected = False
        self.log.append("sim close")

    def apply(self, q1: float, q2: float) -> None:
        if not self._connected:
            raise PumpError("泵未连接")
        for name, value in (("q1", q1), ("q2", q2)):
            if not math.isfinite(value) or value <= 0.0:
                raise PumpSafetyError(f"{name} 必须为正数，得到 {value!r}")
            if value > self.max_flow_ul_min:
                raise PumpSafetyError(f"{name}={value} 超过仿真泵上限 {self.max_flow_ul_min}")
        if not q1_is_at_least_twice_q2(q1, q2):
            raise PumpSafetyError(f"安全约束 Q1>=2*Q2 不满足: Q1={q1}, Q2={q2}")
        self._q1 = round(q1 / self.flow_quantum) * self.flow_quantum
        self._q2 = round(q2 / self.flow_quantum) * self.flow_quantum
        self.log.append(f"apply q1={self._q1:.2f} q2={self._q2:.2f}")

    def read(self) -> tuple[float, float]:
        return self._q1, self._q2

    def stop(self) -> None:
        self._q1 = self._q2 = 0.0
        self.log.append("stop")

    @property
    def connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# 真实串口后端
# ---------------------------------------------------------------------------


class SerialPumpBackend:
    """真实 TS 泵后端。"""

    def __init__(
        self,
        serial_config: SerialConfig | None = None,
        hardware_config: PumpHardwareConfig | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.serial_config = serial_config or SerialConfig()
        self.hardware_config = hardware_config or PumpHardwareConfig()
        self._log = logger or (lambda _msg: None)
        self.client = PumpClient(self.serial_config, self.hardware_config, self._log)
        self._setup: SystemSetup | None = None

    # -- 连接 ---------------------------------------------------------------
    def connect(self) -> None:
        self.client.connect()
        rss = self._query("RSS", protocol.pdu_rss())
        self._setup = protocol.parse_rss_pdu(rss)
        self._log(f"[PUMP] 已连接 {self.serial_config.port}, setup={self._setup}")

    def close(self) -> None:
        try:
            if self.client.is_connected():
                self.client.disconnect()
        finally:
            self._setup = None

    @property
    def connected(self) -> bool:
        return self.client.is_connected()

    # -- 底层收发 -----------------------------------------------------------
    def _query(self, expect: str, pdu: bytes, *, allow_no_reply: bool = False,
               retries: int = 2) -> bytes | None:
        try:
            reply = self.client.send_pdu(pdu, expect_cmd=expect, allow_no_reply=allow_no_reply)
        except PumpClientError as exc:
            raise PumpError(f"{expect} 通信失败: {exc}") from exc
        if reply is None:
            if allow_no_reply:
                return None
            raise PumpError(f"{expect} 无应答")
        return reply.pdu

    def _write(self, cmd: str, pdu: bytes) -> None:
        self._query(cmd, pdu, allow_no_reply=True)

    # -- 流量 ---------------------------------------------------------------
    def _set_flow_params(self, params: ChannelParams) -> None:
        self._write("WSP", protocol.pdu_wsp(
            channel=params.channel,
            mode=params.mode,
            syringe_code=params.syringe_code,
            dispense_value=params.dispense_value,
            dispense_unit=params.dispense_unit,
            infuse_time_value=params.infuse_time_value,
            infuse_time_unit=params.infuse_time_unit,
            withdraw_time_value=params.withdraw_time_value,
            withdraw_time_unit=params.withdraw_time_unit,
            repeat_count=params.repeat_count,
            interval_value=params.interval_value,
        ))

    def _read_channel(self, channel: int) -> float:
        last: Exception | None = None
        for _ in range(3):
            raw = self._query("RSP", protocol.pdu_rsp(channel), retries=1)
            try:
                params = protocol.parse_rsp_pdu(raw or b"")
                if params.channel != channel:
                    raise PumpError(f"RSP 通道号不一致: {params.channel} != {channel}")
                return flow_from_channel_params(params)
            except Exception as exc:  # 回读损坏时重试，而不是回退到指令值
                last = exc
                time.sleep(0.12)
        raise PumpError(f"CH{channel} 回读解析失败: {last}")

    def apply(self, q1: float, q2: float) -> None:
        if not self.connected:
            raise PumpError("泵未连接")
        if not q1_is_at_least_twice_q2(q1, q2):
            raise PumpSafetyError(f"安全约束 Q1>=2*Q2 不满足: Q1={q1}, Q2={q2}")

        # 1) 先停泵再改参数，任何一步失败都不启动。
        self._write("WSE", protocol.pdu_wse(sys_runstate=0x00, q_runstate=0x00))
        time.sleep(self.hardware_config.post_write_delay)

        # 2) 使能 CH1/CH2，禁止通道参数拷贝。
        if self._setup is None:
            self._setup = protocol.parse_rss_pdu(self._query("RSS", protocol.pdu_rss()))
        setup = SystemSetup(
            enable_mask=0x03,
            copy_mask=0x00,
            delay_values=list(self._setup.delay_values),
            delay_units=list(self._setup.delay_units),
        )
        self._write("WSS", protocol.pdu_wss(
            copy_mask=setup.copy_mask,
            enable_mask=setup.enable_mask,
            delay_values=setup.delay_values,
            delay_units=setup.delay_units,
        ))
        time.sleep(self.hardware_config.post_write_delay)

        # 3) 写两通道参数并回读校验。
        for index, flow in ((1, q1), (2, q2)):
            params = channel_params_for_flow(index, flow)
            self._set_flow_params(params)
            time.sleep(self.hardware_config.inter_channel_update_delay)
            actual = self._read_channel(index)
            tolerance = max(0.01, flow * 0.002)
            if abs(actual - flow) > tolerance:
                raise PumpError(
                    f"CH{index} 回读流量 {actual:.4f} 与目标 {flow:.4f} 偏差超过 {tolerance:.4f}"
                )

        # 4) 启动：sys_runstate = 0x01 | enable_mask<<1。
        target_sys = 0x01 | (0x03 << 1)
        self._write("WSE", protocol.pdu_wse(sys_runstate=target_sys, q_runstate=0x00))
        self._verify_running()

    def _verify_running(self) -> None:
        last: RunState | None = None
        for _ in range(8):
            raw = self._query("RSE", protocol.pdu_rse())
            try:
                state = protocol.parse_rse_pdu(raw or b"")
            except Exception as exc:
                raise PumpError(f"RSE 解析失败: {exc}") from exc
            last = state
            if state.system_running and all(state.channel_running[:2]):
                return
            time.sleep(0.12)
        raise PumpError(f"启动后回读未确认运行: {last}")

    def read(self) -> tuple[float, float]:
        return self._read_channel(1), self._read_channel(2)

    def stop(self) -> None:
        if not self.connected:
            return
        self._write("WSE", protocol.pdu_wse(sys_runstate=0x00, q_runstate=0x00))
        time.sleep(self.hardware_config.post_write_delay)
        # 停机回读，失败只记录：上层仍会调用 close()。
        try:
            raw = self._query("RSE", protocol.pdu_rse())
            state = protocol.parse_rse_pdu(raw or b"")
            if state.system_running:
                self._log(f"[PUMP][WARN] 停泵后回读仍在运行: {state}")
        except Exception as exc:
            self._log(f"[PUMP][WARN] 停泵回读失败: {exc}")


# ---------------------------------------------------------------------------
# 控制器
# ---------------------------------------------------------------------------


class PumpController:
    """实验层唯一入口：施加流量、回读、停机，并强制执行固定安全约束。"""

    def __init__(self, backend: PumpBackend, logger: Callable[[str], None] | None = None) -> None:
        self.backend = backend
        self._log = logger or (lambda _msg: None)
        self._target: tuple[float, float] | None = None

    def connect(self) -> None:
        self.backend.connect()

    def close(self) -> None:
        try:
            self.backend.stop()
        finally:
            self.backend.close()

    def apply(self, q1: float, q2: float) -> tuple[float, float]:
        q1, q2 = float(q1), float(q2)
        if not q1_is_at_least_twice_q2(q1, q2):
            raise PumpSafetyError(
                f"Q1/Q2 必须满足 Q1 >= {STRICT_Q1_Q2_RATIO:g} * Q2（当前 Q1={q1}, Q2={q2}）"
            )
        self.backend.apply(q1, q2)
        self._target = (q1, q2)
        actual = self.backend.read()
        self._log(f"[PUMP] apply q1={q1:.3f} q2={q2:.3f} -> readback {actual}")
        return actual

    def read(self) -> tuple[float, float]:
        return self.backend.read()

    def stop(self) -> None:
        self.backend.stop()
        self._target = None

    @property
    def target(self) -> tuple[float, float] | None:
        return self._target
