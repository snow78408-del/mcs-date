"""Command line entry point for ``python -m mcs_date``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mcs-date",
        description="MCS 二相图实验系统：视频液滴分析、泵控制、蜘蛛式扫描与相图动画。",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="离线跑通视觉/泵/二相图三部分")
    demo.add_argument("--out", default="out", help="输出目录（默认 out）")
    demo.add_argument("--quiet", action="store_true", help="只输出汇总")

    scan = subparsers.add_parser("scan", help="跑一次蜘蛛式扫描（离线仿真）")
    scan.add_argument("--out", default="out", help="输出目录（默认 out）")
    scan.add_argument("--center-q1", type=float, default=80.0, help="中心工作点 Q1 (µL/min)")
    scan.add_argument("--center-q2", type=float, default=20.0, help="中心工作点 Q2 (µL/min)")
    scan.add_argument("--node-seconds", type=float, default=180.0, help="每节点观测秒数")
    scan.add_argument("--no-animate", action="store_true", help="只保存数据，不生成动画")
    scan.add_argument("--quiet", action="store_true", help="只输出汇总")

    animate = subparsers.add_parser("animate", help="根据已保存的扫描结果生成动画")
    animate.add_argument("--result", required=True, help="scan_result.json 路径")
    animate.add_argument("--out", default="out", help="输出目录（默认 out）")

    vision = subparsers.add_parser("vision", help="对视频文件/相机做液滴识别")
    vision.add_argument("--video", help="视频文件路径")
    vision.add_argument("--camera", type=int, help="相机索引（实时采集）")
    vision.add_argument("--synthetic", action="store_true", help="用内置合成视频演示")
    vision.add_argument("--max-frames", type=int, default=120, help="最多处理帧数")
    vision.add_argument("--wall-top", type=float, help="上壁归一化 y（缺省自动标定）")
    vision.add_argument("--wall-bottom", type=float, help="下壁归一化 y")
    vision.add_argument("--out", default="out", help="输出目录（默认 out）")

    pump = subparsers.add_parser("pump", help="设定/回读泵流量")
    pump.add_argument("--q1", type=float, default=80.0, help="连续相（油）总流量 µL/min")
    pump.add_argument("--q2", type=float, default=20.0, help="分散相（水）流量 µL/min")
    pump.add_argument("--port", help="串口，例如 COM3；缺省用仿真泵")
    return parser


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    logger = (lambda _msg: None) if args.quiet else print
    summary = run_demo(args.out, logger=logger)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    from .experiment.animate import animate_scan
    from .experiment.models import ScanConfig
    from .experiment.report import save_nodes_csv, save_result
    from .experiment.runner import run_offline_scan

    logger = (lambda _msg: None) if args.quiet else print
    config = ScanConfig(
        center_q1=args.center_q1,
        center_q2=args.center_q2,
        node_duration_s=args.node_seconds,
    )
    config.validate()
    result = run_offline_scan(config, logger=logger)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_result(result, out_dir / "scan_result.json")
    csv_path = save_nodes_csv(result, out_dir / "nodes.csv")
    payload: dict[str, object] = {
        "nodes": len(result.nodes),
        "brackets": len(result.brackets),
        "json": str(json_path),
        "csv": str(csv_path),
    }
    if not args.no_animate:
        artifacts = animate_scan(result, out_dir)
        payload.update({"gif": str(artifacts.gif_path), "png": str(artifacts.png_path)})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_animate(args: argparse.Namespace) -> int:
    from .experiment.animate import animate_scan
    from .experiment.report import load_result

    result = load_result(args.result)
    artifacts = animate_scan(result, args.out)
    print(
        json.dumps(
            {"gif": str(artifacts.gif_path), "png": str(artifacts.png_path),
             "frames": artifacts.frame_count},
            ensure_ascii=False, indent=2,
        )
    )
    return 0


def _iter_frames(args: argparse.Namespace) -> "Sequence[object]":
    from .demo import render_channel_frame

    if args.synthetic:
        return [render_channel_frame(i) for i in range(min(args.max_frames, 60))]
    if args.video:
        import cv2

        capture = cv2.VideoCapture(args.video)
        if not capture.isOpened():
            raise SystemExit(f"无法打开视频: {args.video}")
        frames = []
        while len(frames) < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        return frames
    if args.camera is not None:
        import cv2

        capture = cv2.VideoCapture(args.camera)
        if not capture.isOpened():
            raise SystemExit(f"无法打开相机: {args.camera}")
        frames = []
        while len(frames) < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        return frames
    raise SystemExit("请指定 --video、--camera 或 --synthetic")


def _cmd_vision(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from .vision import VisionPipeline, default_config

    frames = _iter_frames(args)
    if not frames:
        raise SystemExit("没有可处理的帧")
    pipeline = VisionPipeline(default_config())
    walls = None
    if args.wall_top is not None and args.wall_bottom is not None:
        walls = [
            {"x1": 0.0, "y1": args.wall_top, "x2": 1.0, "y2": args.wall_top},
            {"x1": 0.0, "y1": args.wall_bottom, "x2": 1.0, "y2": args.wall_bottom},
        ]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detections: list[int] = []
    preview = None
    for index, frame in enumerate(frames):
        result = pipeline.process_frame(frame, timestamp=index / 10.0, current_wall_lines=walls)
        count = len(result.detections.centers)
        detections.append(count)
        if index == len(frames) // 2:
            preview = frame.copy()
            for center, radius in zip(result.detections.centers, result.detections.radii):
                cv2.circle(preview, (int(center[0]), int(center[1])), int(radius),
                           (40, 200, 60), 2)
                cv2.circle(preview, (int(center[0]), int(center[1])), 2, (40, 60, 220), -1)
            cv2.putText(preview, f"frame {index}  dets={count}", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    if preview is None:
        preview = frames[-1].copy()
    preview_path = out_dir / "vision_preview.png"
    cv2.imwrite(str(preview_path), preview)
    payload = {
        "frames": len(frames),
        "frames_with_detection": sum(1 for count in detections if count > 0),
        "total_detections": sum(detections),
        "mean_detections_per_frame": round(float(np.mean(detections)), 3),
        "preview": str(preview_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_pump(args: argparse.Namespace) -> int:
    from .pump import PumpController, SerialPumpBackend, SimulatedPumpBackend

    backend = SimulatedPumpBackend()
    if args.port:
        from .pump import SerialConfig

        backend = SerialPumpBackend(serial_config=SerialConfig(port=args.port))
    pump = PumpController(backend)
    pump.connect()
    try:
        actual = pump.apply(args.q1, args.q2)
        readback = pump.read()
        print(
            json.dumps(
                {"target": [args.q1, args.q2], "applied": list(actual), "readback": list(readback)},
                ensure_ascii=False, indent=2,
            )
        )
    finally:
        pump.stop()
        pump.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        print(f"\nmcs-date {__version__}: 用 demo 快速跑通三部分，例如 `python -m mcs_date demo --out out`。")
        return 0
    handlers = {
        "demo": _cmd_demo,
        "scan": _cmd_scan,
        "animate": _cmd_animate,
        "vision": _cmd_vision,
        "pump": _cmd_pump,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
