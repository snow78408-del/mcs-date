from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .cameras.models import (
    CameraCapabilities as UnifiedCameraCapabilities,
    CameraDeviceInfo,
    CameraFeatureCapability,
    CameraStatus as UnifiedCameraStatus,
    CameraTestResult as UnifiedCameraTestResult,
    FrameData,
)


HIKROBOT_DEVICE_TYPE = "hikrobot_industrial_camera"
USB_CAMERA_DEVICE_TYPE = "usb_camera"


@dataclass(slots=True)
class CameraDevice:
    device_id: str
    device_name: str
    device_type: str
    transport_type: str = ""
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    user_defined_name: str = ""
    current_ip: str = ""
    available: bool = True
    error: str = ""
    device_index: int | None = None
    device_path: str = ""
    sdk_path: str = ""
    raw_info: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        # raw_info may contain vendor SDK SWIG/ctypes handles. Avoid asdict(),
        # which deep-copies every field before raw_info can be removed.
        return {item.name: getattr(self, item.name) for item in fields(self) if item.name != "raw_info"}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CameraDevice":
        return cls(
            device_id=str(raw.get("device_id", "") or ""),
            device_name=str(raw.get("device_name", "") or ""),
            device_type=str(raw.get("device_type", "") or ""),
            transport_type=str(raw.get("transport_type", "") or ""),
            manufacturer=str(raw.get("manufacturer", "") or ""),
            model=str(raw.get("model", "") or ""),
            serial_number=str(raw.get("serial_number", "") or ""),
            user_defined_name=str(raw.get("user_defined_name", "") or ""),
            current_ip=str(raw.get("current_ip", "") or ""),
            available=bool(raw.get("available", True)),
            error=str(raw.get("error", "") or ""),
            device_index=raw.get("device_index"),
            device_path=str(raw.get("device_path", "") or ""),
            sdk_path=str(raw.get("sdk_path", "") or ""),
            raw_info=raw.get("raw_info"),
        )


@dataclass(slots=True)
class CameraConfig:
    exposure_time: float | None = None
    gain: float | None = None
    frame_rate: float | None = None
    width: int | None = None
    height: int | None = None
    offset_x: int | None = None
    offset_y: int | None = None
    pixel_format: str | None = None
    trigger_mode: str = "Off"
    mvs_sdk_path: str = ""


@dataclass(slots=True)
class CameraParameterInfo:
    name: str
    exists: bool
    readable: bool = False
    writable: bool = False
    min_value: float | int | None = None
    max_value: float | int | None = None
    current_value: float | int | str | None = None
    error: str = ""


@dataclass(slots=True)
class CameraTestResult:
    ok: bool
    device_id: str
    message: str = ""
    error: str = ""
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    frame_rate: float = 0.0
    frames_read: int = 0
    error_code: int | None = None
    allowed_ranges: dict[str, dict[str, Any]] | None = None


@dataclass(slots=True)
class CameraStatus:
    device_id: str = ""
    device_type: str = ""
    is_open: bool = False
    is_streaming: bool = False
    camera_connected: bool = False
    latest_frame_timestamp: float = 0.0
    frame_id: int = 0
    consecutive_frame_failures: int = 0
    last_camera_error: str = ""
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    frame_rate: float = 0.0


@dataclass(slots=True)
class FramePacket:
    frame: Any
    timestamp: float
    frame_id: int
    valid: bool = True
    error: str = ""
