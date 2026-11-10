"""相机数据结构（仅保留 models，不含厂商 SDK 适配器）。"""

from __future__ import annotations

from .models import (
    DEVICE_TYPE_INDUSTRIAL,
    DEVICE_TYPE_UNKNOWN,
    DEVICE_TYPE_USB,
    CameraBackendStatus,
    CameraCapabilities,
    CameraDeviceInfo,
    CameraDiscoveryResult,
    CameraFeatureCapability,
    CameraStatus,
    CameraTestResult,
    FrameData,
)

__all__ = [
    "DEVICE_TYPE_INDUSTRIAL",
    "DEVICE_TYPE_UNKNOWN",
    "DEVICE_TYPE_USB",
    "CameraBackendStatus",
    "CameraCapabilities",
    "CameraDeviceInfo",
    "CameraDiscoveryResult",
    "CameraFeatureCapability",
    "CameraStatus",
    "CameraTestResult",
    "FrameData",
]
