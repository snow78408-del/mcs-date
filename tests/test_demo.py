"""第一部分（视觉）与第二部分（泵）的演示级测试。"""

from __future__ import annotations

import numpy as np
import pytest

from mcs_date.demo import demo_pump, demo_vision, render_channel_frame, wall_lines
from mcs_date.pump import (
    PumpController,
    PumpSafetyError,
    SimulatedPumpBackend,
    channel_params_for_flow,
    flow_from_channel_params,
)
from mcs_date.vision import VisionPipeline, default_config


def test_synthetic_frame_shape_and_walls() -> None:
    frame = render_channel_frame(0)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    lines = wall_lines()
    assert len(lines) == 2
    assert all({"x1", "y1", "x2", "y2"} <= set(line) for line in lines)


def test_vision_pipeline_detects_droplets() -> None:
    pipeline = VisionPipeline(default_config())
    frames = [render_channel_frame(index) for index in range(3)]
    results = [
        pipeline.process_frame(frame, timestamp=index / 10.0, current_wall_lines=wall_lines())
        for index, frame in enumerate(frames)
    ]
    assert all(result.channel_region.status == "localized" for result in results)
    assert all(len(result.detections.centers) >= 1 for result in results)
    radius = float(results[0].detections.radii[0])
    assert radius == pytest.approx(22.0, abs=4.0)


def test_demo_vision_writes_montage(tmp_path) -> None:
    stats = demo_vision(tmp_path, frames=4, logger=lambda *_: None)
    assert stats["frames"] == 4
    assert stats["detected_frames"] >= 1
    assert stats["mean_radius_px"] is not None
    assert (tmp_path / "demo_vision.png").exists()


def test_simulated_pump_apply_read_stop() -> None:
    controller = PumpController(SimulatedPumpBackend())
    controller.connect()
    try:
        applied = controller.apply(80.0, 20.0)
        assert applied == pytest.approx((80.0, 20.0))
        assert controller.read() == pytest.approx((80.0, 20.0))
        controller.stop()
        assert controller.read() == pytest.approx((0.0, 0.0))
    finally:
        controller.close()


def test_simulated_pump_rejects_unsafe_ratio() -> None:
    controller = PumpController(SimulatedPumpBackend())
    controller.connect()
    try:
        with pytest.raises(PumpSafetyError):
            controller.apply(30.0, 20.0)
        # 失败不应改变当前流量
        assert controller.read() == pytest.approx((0.0, 0.0))
    finally:
        controller.close()


def test_channel_params_roundtrip() -> None:
    params = channel_params_for_flow(1, 80.0)
    assert params.channel == 1
    assert flow_from_channel_params(params) == pytest.approx(80.0, rel=1e-6)
    params2 = channel_params_for_flow(2, 20.0)
    assert flow_from_channel_params(params2) == pytest.approx(20.0, rel=1e-6)


def test_demo_pump_reports_safety_guard() -> None:
    stats = demo_pump(logger=lambda *_: None)
    assert stats["applied"] == pytest.approx([80.0, 20.0])
    assert stats["safety_guard"] is True
    assert stats["stopped_readback"] == pytest.approx([0.0, 0.0])
