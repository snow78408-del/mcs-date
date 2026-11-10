"""TS 注射泵控制（CH1=连续油相，CH2=分散水相）。"""

from __future__ import annotations

from .config import PumpHardwareConfig, SerialConfig
from .invariants import (
    STRICT_Q1_Q2_GAP_UL_MIN,
    STRICT_Q1_Q2_RATIO,
    q1_is_at_least_twice_q2,
    q1_is_strictly_above_q2,
)
from .models import ChannelParams, PumpChannelState, RunState, SystemSetup
from .service import (
    PumpBackend,
    PumpController,
    PumpError,
    PumpSafetyError,
    SerialPumpBackend,
    SimulatedPumpBackend,
    channel_params_for_flow,
    flow_from_channel_params,
)

__all__ = [
    "ChannelParams",
    "PumpBackend",
    "PumpChannelState",
    "PumpController",
    "PumpError",
    "PumpHardwareConfig",
    "PumpSafetyError",
    "RunState",
    "STRICT_Q1_Q2_GAP_UL_MIN",
    "STRICT_Q1_Q2_RATIO",
    "SerialConfig",
    "SerialPumpBackend",
    "SimulatedPumpBackend",
    "SystemSetup",
    "channel_params_for_flow",
    "flow_from_channel_params",
    "q1_is_at_least_twice_q2",
    "q1_is_strictly_above_q2",
]
