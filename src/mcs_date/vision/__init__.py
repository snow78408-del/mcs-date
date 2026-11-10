"""液滴识别分析子系统。

代码移植自 ``E:/MCS_QT/backend/vision``（只读复制，原工程未改动），保留原有
检测/跟踪/度量实现，去掉了对相机 SDK 的强制依赖：视频、OpenCV 相机与
合成图像都通过 ``FramePacket`` 进入同一个 ``VisionPipeline``。
"""

from __future__ import annotations

from .config import PipelineConfig, default_config
from .detector import DetectionResult, DropletDetector
from .models import FramePacket
from .pipeline import VisionPipeline, VisionResult
from .tracker import DropletTrack, TrackingResult

__all__ = [
    "DetectionResult",
    "DropletDetector",
    "DropletTrack",
    "FramePacket",
    "PipelineConfig",
    "TrackingResult",
    "VisionPipeline",
    "VisionResult",
    "default_config",
]
