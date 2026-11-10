"""命令行入口测试（全部走离线/合成路径，不依赖相机与串口）。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mcs_date.__main__ import _iter_frames, build_parser, main


def test_no_arguments_prints_help(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "usage" in output.lower()


def test_version_flag_exits_zero() -> None:
    try:
        main(["--version"])
    except SystemExit as exc:  # argparse 以 SystemExit(0) 结束
        assert exc.code == 0
    else:  # pragma: no cover - argparse 实现变化时的兜底
        raise AssertionError("--version 应触发 SystemExit")


def test_pump_subcommand_uses_simulator(capsys) -> None:
    assert main(["pump", "--q1", "80", "--q2", "20"]) == 0
    output = capsys.readouterr().out
    assert "readback" in output
    assert "80" in output


def test_vision_subcommand_with_synthetic_frames(tmp_path, capsys) -> None:
    assert main(["vision", "--synthetic", "--max-frames", "3", "--out", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert '"frames": 3' in output
    assert (tmp_path / "vision_preview.png").exists()


def test_scan_subcommand_without_animation(tmp_path, capsys) -> None:
    assert main(
        [
            "scan",
            "--node-seconds",
            "2",
            "--no-animate",
            "--out",
            str(tmp_path),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert '"nodes"' in output
    assert (tmp_path / "scan_result.json").exists()
    assert (tmp_path / "nodes.csv").exists()


def test_scan_defaults_and_budget_history_flags(tmp_path, capsys) -> None:
    defaults = build_parser().parse_args(["scan"])
    assert (defaults.center_q1, defaults.center_q2) == (90.0, 20.0)
    assert defaults.target_regime == "jetting"
    assert (defaults.max_nodes, defaults.max_hours, defaults.search_rounds) == (140, 4.0, 1)
    history = tmp_path / "missing_history"
    assert main([
        "scan", "--max-nodes", "2", "--max-hours", "0.1", "--search-rounds", "0",
        "--history-root", str(history), "--node-seconds", "2", "--no-animate",
        "--quiet", "--out", str(tmp_path),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "simulation"
    assert payload["target_regime"] == "jetting"
    assert payload["window_identified"] is False
    assert "droplet_window" not in payload
    saved = json.loads((tmp_path / "scan_result.json").read_text(encoding="utf-8"))
    assert saved["config"]["target_regime"] == "jetting"
    assert saved["config"]["max_nodes"] == 2
    assert saved["config"]["max_duration_s"] == 360
    assert saved["config"]["window_search_rounds"] == 0
    assert saved["metadata"]["source"] == "simulation"
    calibration = saved["metadata"]["calibration"]
    assert calibration["window_identified"] is False
    assert calibration["ratio_low_confidence"] == "prior"
    assert calibration["warnings"]
    assert "observation_ledger" in calibration
    assert "interval_constraints" in calibration
    assert not history.exists()  # A missing read-only history root is never created.


@pytest.mark.parametrize("regime, mode", [("jetting", "jetting"), ("dripping", "ratio_window")])
def test_scan_target_regime_sets_simulation_mode_and_center(
    monkeypatch, tmp_path, capsys, regime, mode,
) -> None:
    import mcs_date.experiment.chip as chip

    model_configs = []
    simulator_type = chip.ChipSimulator

    def simulator(*, config):
        model_configs.append(config)
        return simulator_type(config=config)

    monkeypatch.setattr(chip, "ChipSimulator", simulator)
    assert main([
        "scan", "--target-regime", regime, "--center-q1", "100", "--center-q2", "25",
        "--max-nodes", "1", "--node-seconds", "2", "--no-animate", "--quiet",
        "--history-root", str(tmp_path / "missing"), "--out", str(tmp_path),
    ]) == 0
    assert len(model_configs) == 1
    assert model_configs[0].operating_mode == mode
    assert (model_configs[0].jetting_center_q1, model_configs[0].jetting_center_q2) == (100, 25)
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "simulation"
    assert payload["target_regime"] == regime
    saved = json.loads((tmp_path / "scan_result.json").read_text(encoding="utf-8"))
    assert saved["config"]["target_regime"] == regime


@pytest.mark.parametrize("options", [
    ["--max-nodes", "0"], ["--max-hours", "0"], ["--max-hours", "nan"],
    ["--search-rounds", "-1"], ["--node-seconds", "0"], ["--center-q1", "nan"],
])
def test_scan_rejects_invalid_limits_before_saving(tmp_path, capsys, options) -> None:
    assert main(["scan", *options, "--no-animate", "--quiet", "--out", str(tmp_path)]) == 1
    assert not (tmp_path / "scan_result.json").exists()
    output = capsys.readouterr()
    assert not output.out
    assert "Traceback" not in output.err


def test_learn_accepts_explicit_reviewed_observations(tmp_path, capsys) -> None:
    path = tmp_path / "manual.json"
    path.write_text(json.dumps([
        [40, 20, "no_droplet_low", "high", "review:low"],
        [60, 20, "droplet", "high", "review:inside"],
        [90, 20, "no_droplet_high", "high", "review:high"],
    ]), encoding="utf-8")
    assert main(["learn", "--history-root", str(tmp_path / "missing"),
                 "--observations", str(path), "--out", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["window_identified"] is True
    assert payload["evidence"]["valid_label_points"] == 3
    saved = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert all(x["review_status"] == "reviewed" for x in saved["observation_ledger"])


@pytest.mark.parametrize("text", ['{}', '[[]]', '[[60,20,"droplet","high",""]]', 'not JSON'])
def test_learn_rejects_invalid_observations(tmp_path, capsys, text) -> None:
    path = tmp_path / "manual.json"
    path.write_text(text, encoding="utf-8")
    assert main(["learn", "--observations", str(path), "--out", str(tmp_path)]) == 1
    assert "Traceback" not in capsys.readouterr().err
    assert not (tmp_path / "calibration.json").exists()


@pytest.mark.parametrize("source", [[], ["--synthetic", "--video", "fake.mp4"],
                                    ["--synthetic", "--camera", "0"]])
def test_vision_requires_exactly_one_source(source) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["vision", *source])
    assert exc.value.code == 2


@pytest.mark.parametrize("options", [
    ["--max-frames", "0"], ["--max-frames", "-1"],
    ["--wall-top", "0.2"], ["--wall-bottom", "0.8"],
    ["--wall-top", "-0.1", "--wall-bottom", "0.8"],
    ["--wall-top", "0.5", "--wall-bottom", "0.5"],
    ["--wall-top", "0.8", "--wall-bottom", "0.2"],
    ["--wall-top", "0.2", "--wall-bottom", "1.1"],
    ["--wall-top", "nan", "--wall-bottom", "0.8"],
])
def test_vision_validates_before_opening_capture(monkeypatch, capsys, options) -> None:
    def forbidden(_source):
        raise AssertionError("capture must not be opened for invalid arguments")
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    assert main(["vision", "--camera", "0", *options]) == 1
    assert "Traceback" not in capsys.readouterr().err


class FakeCapture:
    def __init__(self, pts=(0.0, 0.0, 50.0), fps=20.0, *, opened=True, read_error=False):
        self.pts = pts
        self.fps = fps
        self.opened = opened
        self.read_error = read_error
        self.reads = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.read_error:
            raise RuntimeError("capture read failed")
        if self.reads >= len(self.pts):
            return False, None
        self.reads += 1
        return True, np.zeros((50, 100, 3), dtype=np.uint8)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.pts[self.reads - 1]
        raise AssertionError(prop)

    def release(self):
        self.released = True


def test_video_generator_is_lazy_monotonic_and_releases(monkeypatch) -> None:
    capture = FakeCapture(pts=(0.0, 0.0, -1.0, 10.0, 500.0))
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    args = build_parser().parse_args(["vision", "--video", "fake.mp4", "--max-frames", "5"])
    frames = _iter_frames(args)
    assert capture.reads == 0
    first = next(frames)
    assert capture.reads == 1
    samples = [first, *frames]
    timestamps = [x[1] for x in samples]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    assert samples[0][2] == "video_media_pts"
    assert samples[1][2] == "video_index_fps_fallback"
    assert samples[-1][1] == 0.5
    assert capture.released


@pytest.mark.parametrize("case", ["unopened", "read_error", "no_fps"])
def test_video_generator_releases_on_capture_errors(monkeypatch, case) -> None:
    capture = FakeCapture(opened=case != "unopened", read_error=case == "read_error",
                          fps=0.0 if case == "no_fps" else 20.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    args = build_parser().parse_args(["vision", "--video", "fake.mp4"])
    with pytest.raises((ValueError, RuntimeError)):
        list(_iter_frames(args))
    assert capture.released


def test_camera_uses_monotonic_acquisition_timestamps(monkeypatch) -> None:
    import mcs_date.__main__ as cli

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    clock = iter([123.0, 123.1, 123.2])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    args = build_parser().parse_args(["vision", "--camera", "0", "--max-frames", "3"])
    samples = list(_iter_frames(args))
    assert [x[1] for x in samples] == [123.0, 123.1, 123.2]
    assert {x[2] for x in samples} == {"camera_monotonic"}
    assert capture.released


def test_vision_processes_immediately_and_draws_on_analysis_frame(monkeypatch, tmp_path, capsys) -> None:
    import mcs_date.vision as vision

    capture = FakeCapture()
    timestamps = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    class Pipeline:
        def __init__(self, _config):
            pass

        def process_frame(self, frame, *, timestamp, current_wall_lines):
            assert capture.reads == len(timestamps) + 1  # No prebuffering.
            timestamps.append(timestamp)
            return SimpleNamespace(
                detections=SimpleNamespace(centers=[(5, 5)], radii=[2]),
                analysis_frame=np.full((30, 40, 3), 90, dtype=np.uint8),
                annotated_frame=frame,
            )

    monkeypatch.setattr(vision, "VisionPipeline", Pipeline)
    assert main(["vision", "--video", "fake.mp4", "--out", str(tmp_path)]) == 0
    assert capture.released
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    preview = cv2.imread(str(tmp_path / "vision_preview.png"))
    assert preview.shape == (30, 40, 3)
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_detections"] == 3
    assert payload["topology"] == "unknown"
    assert "rate" not in payload
    assert "not_unique_droplet_generation" in payload["detection_count_basis"]
    assert "video_index_fps_fallback" in payload["timestamp_basis"]


def test_vision_releases_capture_when_pipeline_raises(monkeypatch, tmp_path, capsys) -> None:
    import mcs_date.vision as vision

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    class Pipeline:
        def __init__(self, _config):
            pass

        def process_frame(self, *_args, **_kwargs):
            raise RuntimeError("pipeline failed")

    monkeypatch.setattr(vision, "VisionPipeline", Pipeline)
    assert main(["vision", "--video", "fake.mp4", "--out", str(tmp_path)]) == 1
    assert capture.reads == 1
    assert capture.released
    output = capsys.readouterr()
    assert "pipeline failed" in output.err
    assert not output.out


@pytest.mark.parametrize("options", [
    ["--q1", "nan"], ["--q1", "inf"], ["--q2", "0"],
    ["--q1", "30", "--q2", "20"],
])
def test_pump_validates_before_constructing_serial_backend(monkeypatch, capsys, options) -> None:
    import mcs_date.pump as pump

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must not construct hardware backend")
    monkeypatch.setattr(pump, "SerialPumpBackend", forbidden)
    assert main(["pump", "--port", "FAKE", *options]) == 1
    output = capsys.readouterr()
    assert "must not construct" not in output.err
    assert not output.out


@pytest.mark.parametrize("failure", [
    "connect", "apply", "read", "close", "interrupt", "both", "interrupt_and_close",
])
def test_pump_cleanup_and_errors_never_print_success(monkeypatch, capsys, failure) -> None:
    import mcs_date.pump as pump

    events = []

    class Controller:
        def __init__(self, _backend):
            pass

        def connect(self):
            events.append("connect")
            if failure in {"connect", "both"}:
                raise RuntimeError("primary connect failed")

        def apply(self, *_args):
            events.append("apply")
            if failure == "apply":
                raise RuntimeError("primary apply failed")
            if failure in {"interrupt", "interrupt_and_close"}:
                raise KeyboardInterrupt
            return 80, 20

        def read(self):
            events.append("read")
            if failure == "read":
                raise RuntimeError("primary read failed")
            return 80, 20

        def close(self):
            events.append("close")
            if failure in {"close", "both", "interrupt_and_close"}:
                raise RuntimeError("cleanup failed")

    monkeypatch.setattr(pump, "PumpController", Controller)
    assert main(["pump"]) == (130 if failure in {"interrupt", "interrupt_and_close"} else 1)
    assert events[-1] == "close"
    output = capsys.readouterr()
    assert not output.out
    assert "Traceback" not in output.err
    if failure == "both":
        assert "primary connect failed" in output.err
        assert "cleanup failed" in output.err
    if failure == "interrupt_and_close":
        assert "操作已取消" in output.err
        assert "cleanup failed" in output.err
