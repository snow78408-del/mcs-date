"""Command line entry point for ``python -m mcs_date``."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterator

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

    learn = subparsers.add_parser("learn", help="从 MCS_QT 历史数据学习液滴窗口/尺寸模型")
    learn.add_argument("--out", default="out", help="输出目录（默认 out）")
    learn.add_argument("--history-root", default=None, help="MCS_QT_Data 目录（缺省读环境变量）")
    learn.add_argument(
        "--observations", help="显式人工复核 JSON：[[Q1,Q2,label,confidence,source], ...]；"
        "label 可用 droplet/no_droplet_high/no_droplet_low，jetting 本身不代表无液滴",
    )
    learn.add_argument("--quiet", action="store_true", help="只输出 JSON")

    scan = subparsers.add_parser("scan", help="跑一次蜘蛛式扫描（离线仿真）")
    scan.add_argument("--out", default="out", help="输出目录（默认 out）")
    scan.add_argument("--center-q1", type=float, default=90.0, help="中心工作点 Q1 (uL/min)")
    scan.add_argument("--center-q2", type=float, default=20.0, help="中心工作点 Q2 (uL/min)")
    scan.add_argument(
        "--target-regime", choices=("jetting", "dripping"), default="jetting",
        help="从最佳射流点向周围探索（默认 jetting）；dripping 为旧式滴落窗口模式",
    )
    scan.add_argument("--node-seconds", type=float, default=180.0, help="每节点观测秒数")
    scan.add_argument("--max-nodes", type=int, default=140, help="节点预算（默认 140）")
    scan.add_argument("--max-hours", type=float, default=4.0, help="仿真计划时间预算（小时，默认 4）")
    scan.add_argument("--search-rounds", type=int, default=1, help="窄窗口补充探查轮数（默认 1）")
    scan.add_argument("--history-root", default=None, help="只读历史校准目录；scan 始终只运行仿真")
    scan.add_argument("--no-animate", action="store_true", help="只保存数据，不生成动画")
    scan.add_argument("--quiet", action="store_true", help="只输出汇总")

    pilot = subparsers.add_parser(
        "pilot", help="按小规模四射线方案跑射流→束流试验（离线仿真）",
    )
    pilot.add_argument("--out", default="out", help="输出目录（默认 out）")
    pilot.add_argument("--anchor-q1", type=float, default=90.0, help="操作者认可的射流锚点 Q1 (uL/min)")
    pilot.add_argument("--anchor-q2", type=float, default=20.0, help="操作者认可的射流锚点 Q2 (uL/min)")
    pilot.add_argument(
        "--target-regime", choices=("jetting", "dripping"), default="jetting",
        help="默认探索射流到束流；dripping 仅作可选研究模式",
    )
    pilot.add_argument("--node-seconds", type=float, default=180.0, help="每节点观测秒数（方案约 3 分钟）")
    pilot.add_argument("--step", type=float, default=0.25, help="射线乘法步幅（方案 20%–30%，默认 25%）")
    pilot.add_argument("--steps-per-ray", type=int, default=3, help="每条射线最多几个节点（默认 3）")
    pilot.add_argument("--max-hours", type=float, default=1.5, help="仿真计划时间预算（小时，默认 1.5）")
    pilot.add_argument("--no-animate", action="store_true", help="只保存数据，不生成动画")
    pilot.add_argument("--quiet", action="store_true", help="只输出汇总")

    animate = subparsers.add_parser("animate", help="根据已保存的扫描结果生成动画")
    animate.add_argument("--result", required=True, help="scan_result.json 路径")
    animate.add_argument("--out", default="out", help="输出目录（默认 out）")

    vision = subparsers.add_parser("vision", help="对视频文件/相机做液滴识别")
    source = vision.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="视频文件路径")
    source.add_argument("--camera", type=int, help="相机索引（实时采集）")
    source.add_argument("--synthetic", action="store_true", help="用内置合成视频演示")
    vision.add_argument("--max-frames", type=int, default=120, help="最多处理帧数")
    vision.add_argument("--wall-top", type=float, help="上壁归一化 y（缺省自动标定）")
    vision.add_argument("--wall-bottom", type=float, help="下壁归一化 y")
    vision.add_argument("--out", default="out", help="输出目录（默认 out）")

    pump = subparsers.add_parser("pump", help="设定/回读泵流量")
    pump.add_argument("--q1", type=float, default=80.0, help="连续相（油）总流量 uL/min")
    pump.add_argument("--q2", type=float, default=20.0, help="分散相（水）流量 uL/min")
    pump.add_argument("--port", help="串口，例如 COM3；缺省用仿真泵")
    return parser


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    logger = (lambda _msg: None) if args.quiet else print
    summary = run_demo(args.out, logger=logger)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_learn(args: argparse.Namespace) -> int:
    from .experiment.calibration import learn_calibration, summarize_calibration

    observations = []
    if args.observations:
        observations = json.loads(Path(args.observations).read_text(encoding="utf-8-sig"))
        if not isinstance(observations, list):
            raise ValueError("--observations 必须是人工复核观测数组")
    calibration = learn_calibration(args.history_root, extra_observations=observations)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = calibration.save(out_dir / "calibration.json")
    payload = summarize_calibration(calibration)
    payload["path"] = str(path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    from .experiment.animate import animate_scan
    from .experiment.calibration import learn_calibration
    from .experiment.chip import ChipModelConfig, ChipSimulator
    from .experiment.models import ScanConfig
    from .experiment.report import save_nodes_csv, save_result
    from .experiment.runner import run_offline_scan

    logger = (lambda _msg: None) if args.quiet else print
    config = ScanConfig(
        center_q1=args.center_q1,
        center_q2=args.center_q2,
        target_regime=args.target_regime,
        node_duration_s=args.node_seconds,
        max_nodes=args.max_nodes,
        max_duration_s=args.max_hours * 3600.0,
        window_search_rounds=args.search_rounds,
    )
    config.validate()
    calibration = learn_calibration(args.history_root)
    model_config = replace(
        ChipModelConfig.from_calibration(calibration),
        operating_mode="jetting" if args.target_regime == "jetting" else "ratio_window",
        jetting_center_q1=args.center_q1,
        jetting_center_q2=args.center_q2,
    )
    simulator = ChipSimulator(config=model_config)
    result = run_offline_scan(config, simulator=simulator, logger=logger)
    result.metadata.update({"source": "simulation", "calibration": calibration.as_dict()})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_result(result, out_dir / "scan_result.json")
    csv_path = save_nodes_csv(result, out_dir / "nodes.csv")
    calibration_path = calibration.save(out_dir / "calibration.json")
    payload: dict[str, object] = {
        "nodes": len(result.nodes),
        "brackets": len(result.brackets),
        "source": "simulation",
        "target_regime": config.target_regime,
        "window_identified": calibration.window_identified,
        "model_ratio_window": [calibration.ratio_low, calibration.ratio_high],
        "calibration_status": calibration.calibration_status,
        "json": str(json_path),
        "csv": str(csv_path),
        "calibration": str(calibration_path),
    }
    if not args.no_animate:
        artifacts = animate_scan(result, out_dir)
        payload.update({"gif": str(artifacts.gif_path), "png": str(artifacts.png_path),
                        "evolution_trace": str(artifacts.evolution_path)})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_pilot(args: argparse.Namespace) -> int:
    from .experiment.animate import animate_scan
    from .experiment.pilot import PilotConfig, run_offline_pilot
    from .experiment.report import save_nodes_csv, save_result

    logger = (lambda _msg: None) if args.quiet else print
    config = PilotConfig(
        anchor_q1=args.anchor_q1,
        anchor_q2=args.anchor_q2,
        target_regime=args.target_regime,
        node_duration_s=args.node_seconds,
        step_fraction=args.step,
        steps_per_ray=args.steps_per_ray,
    )
    config.validate()
    if config.estimated_duration_s > args.max_hours * 3600.0:
        raise ValueError(
            f"计划耗时 {config.estimated_duration_s/3600:.2f}h 超过 --max-hours {args.max_hours:g}h；"
            "请减少节点/步幅或提高预算"
        )
    result = run_offline_pilot(config, logger=logger)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_result(result, out_dir / "pilot_result.json")
    csv_path = save_nodes_csv(result, out_dir / "nodes.csv")
    observed = {node.purpose: node for node in result.nodes}
    plan_path = out_dir / "pilot_plan.csv"
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "ray", "step", "q1_ul_min", "q2_ul_min", "oil_water_ratio", "total_flow",
            "observed_regime", "evolution_stage", "mean_diameter_ratio",
            "breakup_distance_h",
        ])
        writer.writeheader()
        for plan in result.metadata["pilot_plan"]:
            node = observed.get(f"pilot:{plan['ray']}#{plan['step']}")
            features = node.features if node is not None else None
            writer.writerow({
                "ray": plan["ray"], "step": plan["step"],
                "q1_ul_min": plan["q1"], "q2_ul_min": plan["q2"],
                "oil_water_ratio": round(plan["oil_water_ratio"], 4),
                "total_flow": round(plan["total_flow"], 4),
                "observed_regime": node.regime.value if node is not None else "not_run",
                "evolution_stage": features.extra.get("evolution_stage", "") if features else "",
                "mean_diameter_ratio": features.mean_diameter_ratio if features else "",
                "breakup_distance_h": features.breakup_distance_h if features else "",
            })
    payload: dict[str, object] = {
        "design": "four_rays_small_scale",
        "source": "simulation",
        "target_regime": result.config.target_regime,
        "planned_nodes": result.metadata["planned_nodes"],
        "executed_nodes": len(result.nodes),
        "estimated_hours": round(config.estimated_duration_s / 3600.0, 2),
        "ray_status": result.metadata["ray_status"],
        "ray_crossings": len(result.metadata["ray_crossings"]),
        "json": str(json_path),
        "csv": str(csv_path),
        "plan": str(plan_path),
    }
    if not args.no_animate:
        artifacts = animate_scan(result, out_dir)
        payload.update({"gif": str(artifacts.gif_path), "png": str(artifacts.png_path),
                        "evolution_trace": str(artifacts.evolution_path)})
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
             "frames": artifacts.frame_count, "evolution_trace": str(artifacts.evolution_path)},
            ensure_ascii=False, indent=2,
        )
    )
    return 0


def _validate_vision_args(args: argparse.Namespace) -> None:
    if args.max_frames <= 0:
        raise ValueError("--max-frames 必须 > 0")
    if sum((bool(args.video), args.camera is not None, bool(args.synthetic))) != 1:
        raise ValueError("请且仅请指定 --video、--camera 或 --synthetic 之一")
    if (args.wall_top is None) != (args.wall_bottom is None):
        raise ValueError("--wall-top 和 --wall-bottom 必须同时指定")
    if args.wall_top is not None and not 0 <= args.wall_top < args.wall_bottom <= 1:
        raise ValueError("管壁必须满足 0 <= wall-top < wall-bottom <= 1")


def _iter_frames(args: argparse.Namespace) -> Iterator[tuple[object, float, str]]:
    """Yield one acquisition at a time; closing this generator releases the source."""
    _validate_vision_args(args)
    if args.synthetic:
        from .demo import render_channel_frame

        for index in range(args.max_frames):
            yield render_channel_frame(index), index / 10.0, "synthetic_index_10fps"
        return

    import cv2

    capture = cv2.VideoCapture(args.video if args.video else args.camera)
    try:
        if not capture.isOpened():
            raise ValueError(f"无法打开视频/相机: {args.video if args.video else args.camera}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if args.video else 0.0
        last_timestamp = None
        for index in range(args.max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if args.video:
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                basis = "video_media_pts"
                if (not math.isfinite(timestamp) or timestamp < 0
                        or (last_timestamp is not None and timestamp <= last_timestamp)):
                    if not math.isfinite(fps) or fps <= 0:
                        raise ValueError("视频 PTS 不可用/不递增且没有有效 FPS；无法确定采集时间")
                    timestamp = index / fps
                    if last_timestamp is not None:
                        timestamp = max(timestamp, last_timestamp + 1.0 / fps)
                    basis = "video_index_fps_fallback"
            else:
                timestamp = time.monotonic()
                basis = "camera_monotonic"
                if last_timestamp is not None and timestamp <= last_timestamp:
                    timestamp = math.nextafter(last_timestamp, math.inf)
            last_timestamp = timestamp
            yield frame, timestamp, basis
    finally:
        capture.release()


def _cmd_vision(args: argparse.Namespace) -> int:
    _validate_vision_args(args)
    import cv2

    from .vision import VisionPipeline, default_config

    pipeline = VisionPipeline(default_config())
    walls = None
    if args.wall_top is not None:
        walls = [
            {"x1": 0.0, "y1": args.wall_top, "x2": 1.0, "y2": args.wall_top},
            {"x1": 0.0, "y1": args.wall_bottom, "x2": 1.0, "y2": args.wall_bottom},
        ]
    frame_count = frames_with_detection = total_detections = 0
    timestamp_bases: set[str] = set()
    result = None
    # Explicit close also releases a suspended capture when processing raises.
    with closing(_iter_frames(args)) as frames:
        for frame, timestamp, basis in frames:
            result = pipeline.process_frame(frame, timestamp=timestamp, current_wall_lines=walls)
            count = len(result.detections.centers)
            frame_count += 1
            frames_with_detection += int(count > 0)
            total_detections += count
            timestamp_bases.add(basis)
    if result is None:
        raise ValueError("没有可处理的帧")
    # Detection coordinates belong to the (possibly rectified) analysis frame,
    # not necessarily to annotated_frame, which the pipeline may leave raw.
    preview = result.analysis_frame.copy()
    for center, radius in zip(result.detections.centers, result.detections.radii):
        cv2.circle(preview, (int(center[0]), int(center[1])), int(radius), (40, 200, 60), 2)
        cv2.circle(preview, (int(center[0]), int(center[1])), 2, (40, 60, 220), -1)
    cv2.putText(preview, f"frame {frame_count - 1}  detections/frame={count}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = out_dir / "vision_preview.png"
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"无法保存预览: {preview_path}")
    payload = {
        "frames": frame_count,
        "frames_with_detection": frames_with_detection,
        "total_detections": total_detections,
        "detection_count_basis": "sum_of_per_frame_detections_not_unique_droplet_generation",
        "topology": "unknown",
        "mean_detections_per_frame": round(total_detections / frame_count, 3),
        "timestamp_basis": sorted(timestamp_bases),
        "preview": str(preview_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_pump(args: argparse.Namespace) -> int:
    from .pump import (PumpController, PumpSafetyError, SerialPumpBackend,
                       SimulatedPumpBackend, q1_is_at_least_twice_q2)

    # Reject invalid targets before even opening a serial connection.
    if (not all(math.isfinite(q) and q > 0 for q in (args.q1, args.q2))
            or not q1_is_at_least_twice_q2(args.q1, args.q2)):
        raise PumpSafetyError("流量必须为有限正数且满足 Q1 >= 2*Q2")
    if args.port:
        from .pump import SerialConfig, channel_params_for_flow, flow_from_channel_params

        quantized = tuple(flow_from_channel_params(channel_params_for_flow(channel, flow))
                          for channel, flow in ((1, args.q1), (2, args.q2)))
        if not q1_is_at_least_twice_q2(*quantized):
            raise PumpSafetyError("量化后流量违反 Q1 >= 2*Q2")
        backend = SerialPumpBackend(serial_config=SerialConfig(port=args.port))
    else:
        backend = SimulatedPumpBackend()
        if max(args.q1, args.q2) > backend.max_flow_ul_min:
            raise PumpSafetyError("目标流量超过仿真泵上限")
    pump = PumpController(backend)
    try:
        pump.connect()
        actual = pump.apply(args.q1, args.q2)
        readback = pump.read()
    except BaseException as exc:
        try:
            pump.close()
        except BaseException as cleanup:
            note = f"泵清理失败，安全状态未确认: {cleanup}"
            if hasattr(exc, "add_note"):
                exc.add_note(note)
            else:  # Python 3.10
                exc.args = (*exc.args, note)
        raise
    else:
        pump.close()
    # A successful readback is not success until stop/close has also succeeded.
    print(json.dumps(
        {"target": [args.q1, args.q2], "applied": list(actual), "readback": list(readback)},
        ensure_ascii=False, indent=2,
    ))
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
        "learn": _cmd_learn,
        "scan": _cmd_scan,
        "pilot": _cmd_pilot,
        "animate": _cmd_animate,
        "vision": _cmd_vision,
        "pump": _cmd_pump,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt as exc:
        print("操作已取消", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"mcs-date: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
