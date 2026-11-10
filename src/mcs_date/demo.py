"""离线演示：让三部分系统在没有相机、泵和显示器的环境里也能完整跑通。

1. 视觉：合成一段方形微通道视频，走真实的 ``VisionPipeline`` 检测液滴；
2. 泵：用 ``SimulatedPumpBackend`` 演示设定流量、回读校验和安全约束；
3. 实验：用 ``run_offline_scan`` 跑蜘蛛式扫描，生成二相图数据和动画。

真实硬件路径（相机/串口）分别在 ``mcs_date.vision`` 与 ``mcs_date.pump`` 中，
本模块只负责把“怎么用”串起来，方便交接和回归。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .experiment.animate import animate_scan
from .experiment.models import ScanConfig
from .experiment.report import save_nodes_csv, save_result
from .experiment.runner import run_offline_scan
from .pump import PumpController, PumpSafetyError, SimulatedPumpBackend
from .vision import VisionPipeline, default_config

Logger = Callable[[str], None]


# ---------------------------------------------------------------------------
# 第一部分：视频采集 + 液滴识别
# ---------------------------------------------------------------------------


def render_channel_frame(
    frame_index: int,
    *,
    width: int = 640,
    height: int = 480,
    droplet_radius: int = 22,
    droplet_count: int = 2,
) -> np.ndarray:
    """画一帧合成微通道图像：亮通道 + 深色液滴（与 MCS_QT 视频极性一致）。"""
    frame = np.full((height, width, 3), 18, dtype=np.uint8)
    # 通道内壁
    wall_top, wall_bottom = int(height * 0.30), int(height * 0.70)
    cv2.rectangle(frame, (0, wall_top), (width - 1, wall_bottom), (186, 186, 186), -1)
    # 轻微壁面阴影，让边缘不是一条死线
    cv2.line(frame, (0, wall_top), (width - 1, wall_top), (120, 120, 120), 2)
    cv2.line(frame, (0, wall_bottom), (width - 1, wall_bottom), (120, 120, 120), 2)
    speed = 11.0
    span = width + 2 * droplet_radius * droplet_count
    for index in range(droplet_count):
        phase = (index * span / droplet_count) + frame_index * speed
        cx = int(phase % span) - droplet_radius
        cy = (wall_top + wall_bottom) // 2
        if -droplet_radius < cx < width + droplet_radius:
            cv2.circle(frame, (cx, cy), droplet_radius, (46, 46, 46), -1)
            cv2.circle(frame, (cx, cy), droplet_radius, (70, 70, 70), 2)
    return frame


def wall_lines(width: int = 640, height: int = 480) -> list[dict[str, float]]:
    """返回两条管壁（归一化坐标，rectified_roi 约定 x,y ∈ [0,1]）。"""
    return [
        {"x1": 0.0, "y1": 0.30, "x2": 1.0, "y2": 0.30},
        {"x1": 0.0, "y1": 0.70, "x2": 1.0, "y2": 0.70},
    ]


def demo_vision(
    out_dir: str | Path = "out",
    *,
    frames: int = 24,
    logger: Logger = print,
) -> dict[str, Any]:
    """在合成视频上运行视觉管线，输出标注拼图和统计。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = VisionPipeline(default_config())
    walls = wall_lines()
    annotated: list[np.ndarray] = []
    detected_frames = 0
    radii: list[float] = []
    diameters: list[float] = []
    for index in range(frames):
        frame = render_channel_frame(index)
        result = pipeline.process_frame(frame, timestamp=index / 10.0, current_wall_lines=walls)
        detections = result.detections
        if len(detections.centers) > 0:
            detected_frames += 1
        for center, radius in zip(detections.centers, detections.radii):
            radii.append(float(radius))
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(frame, (cx, cy), int(radius), (40, 200, 60), 2)
            cv2.circle(frame, (cx, cy), 2, (40, 60, 220), -1)
        for diameter, valid in zip(detections.equivalent_diameters_px, detections.diameter_valid):
            if valid:
                diameters.append(float(diameter))
        cv2.putText(
            frame,
            f"frame {index:02d}  dets={len(detections.centers)}  "
            f"region={result.channel_region.status}",
            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA,
        )
        annotated.append(frame)

    # 抽取三帧拼成一张图，方便直接查看
    picks = [annotated[0], annotated[len(annotated) // 2], annotated[-1]]
    montage = np.hstack(picks)
    png_path = out_dir / "demo_vision.png"
    cv2.imwrite(str(png_path), montage)

    stats = {
        "frames": frames,
        "detected_frames": detected_frames,
        "detections_per_frame": round(
            sum(1 for _ in range(detected_frames)) / max(frames, 1), 3
        ),
        "mean_radius_px": round(float(np.mean(radii)), 2) if radii else None,
        "mean_equivalent_diameter_px": round(float(np.mean(diameters)), 2) if diameters else None,
        "png": str(png_path),
    }
    logger(
        f"[VISION] {frames} 帧，检出 {detected_frames} 帧，平均半径 "
        f"{stats['mean_radius_px']} px，输出 {png_path}"
    )
    return stats


# ---------------------------------------------------------------------------
# 第二部分：泵连接与控制
# ---------------------------------------------------------------------------


def demo_pump(*, q1: float = 80.0, q2: float = 20.0, logger: Logger = print) -> dict[str, Any]:
    """用仿真泵演示设定、回读、安全约束与停泵。"""
    pump = PumpController(SimulatedPumpBackend())
    pump.connect()
    try:
        actual = pump.apply(q1, q2)
        readback = pump.read()
        safety_ok = False
        safety_error = ""
        try:
            pump.apply(q1 * 0.3, q2)  # 故意违反 Q1 >= 2*Q2
        except PumpSafetyError as exc:
            safety_ok = True
            safety_error = str(exc)
        pump.stop()
        stopped = pump.read()
    finally:
        pump.close()
    stats = {
        "target": (q1, q2),
        "applied": tuple(round(value, 3) for value in actual),
        "readback": tuple(round(value, 3) for value in readback),
        "safety_guard": safety_ok,
        "safety_error": safety_error,
        "stopped_readback": tuple(round(value, 3) for value in stopped),
    }
    logger(
        f"[PUMP] 设定 ({q1:g}, {q2:g}) -> 回读 {stats['readback']}，"
        f"安全约束{'生效' if safety_ok else '未生效'}，停泵后 {stats['stopped_readback']}"
    )
    return stats


# ---------------------------------------------------------------------------
# 第三部分：蜘蛛式扫描 + 二相图
# ---------------------------------------------------------------------------


def demo_experiment(
    out_dir: str | Path = "out",
    *,
    scan_config: ScanConfig | None = None,
    logger: Logger = print,
) -> dict[str, Any]:
    """离线跑一次二相图扫描，保存数据并生成动画。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = scan_config or ScanConfig()
    result = run_offline_scan(config, logger=logger)
    json_path = save_result(result, out_dir / "scan_result.json")
    csv_path = save_nodes_csv(result, out_dir / "nodes.csv")
    artifacts = animate_scan(result, out_dir)
    summary = {
        "nodes": len(result.nodes),
        "brackets": len(result.brackets),
        "planned_hours": round(result.total_duration_s / 3600.0, 2),
        "stopped_reason": result.stopped_reason,
        "json": str(json_path),
        "csv": str(csv_path),
        "gif": str(artifacts.gif_path),
        "png": str(artifacts.png_path),
        "frames": artifacts.frame_count,
    }
    logger(
        f"[SCAN] {summary['nodes']} 节点 / {summary['brackets']} 边界，"
        f"计划 {summary['planned_hours']} h，动画 {summary['gif']}"
    )
    return summary


def run_demo(out_dir: str | Path = "out", logger: Logger = print) -> dict[str, Any]:
    """一次跑通三部分，返回汇总结果。"""
    out_dir = Path(out_dir)
    vision = demo_vision(out_dir, logger=logger)
    pump = demo_pump(logger=logger)
    experiment = demo_experiment(out_dir, logger=logger)
    return {"vision": vision, "pump": pump, "experiment": experiment}
