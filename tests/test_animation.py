"""二相图动画（GIF 编码 / 渲染）测试。"""

from __future__ import annotations

import numpy as np

from mcs_date.experiment import (
    DiagramConfig,
    ScanConfig,
    animate_scan,
    run_offline_scan,
)
from mcs_date.experiment.animate import BG_BGR, GifWriter, PANEL_BGR, _palette

from _gif_tools import read_gif


def test_palette_has_valid_size() -> None:
    palette = _palette()
    assert 2 <= len(palette) <= 256
    assert palette[0] == (247, 247, 247)  # BGR 浅灰背景 -> RGB
    assert len({tuple(color) for color in palette}) == len(palette)


def test_quantize_maps_flat_colors_without_overflow() -> None:
    # 回归：255^2 曾以 int16 溢出，导致背景被映射成别的颜色
    palette = _palette()
    writer = GifWriter.__new__(GifWriter)
    writer._palette = list(palette) + [(0, 0, 0)] * (256 - len(palette))
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[:, :] = BG_BGR[::-1]
    indices = writer._quantize(image)
    assert indices.shape == (6, 8)
    assert set(np.unique(indices).tolist()) == {0}

    image[0:3, :] = PANEL_BGR[::-1]
    indices = writer._quantize(image.copy())
    assert len(set(np.unique(indices).tolist())) == 2


def test_gif_roundtrip_bytes(tmp_path) -> None:
    frames = [
        np.full((24, 32, 3), 0, dtype=np.uint8),
        np.full((24, 32, 3), 255, dtype=np.uint8),
    ]
    path = tmp_path / "roundtrip.gif"
    writer = GifWriter(path, 32, 24, _palette(), delay_cs=10)
    for frame in frames:
        writer.add_frame(frame)
    writer.close()

    width, height, decoded = read_gif(path)
    assert (width, height) == (32, 24)
    assert len(decoded) == 2
    palette = _palette()
    black_index = palette.index((0, 0, 0))
    white_index = palette.index((255, 255, 255))
    assert black_index != white_index
    assert decoded[0] == bytes([black_index]) * (32 * 24)
    assert decoded[1] == bytes([white_index]) * (32 * 24)


def test_animate_scan_writes_valid_gif_and_png(tmp_path) -> None:
    scan = ScanConfig(
        center_q1=80.0,
        center_q2=20.0,
        node_duration_s=2.0,
        startup_s=0.5,
        bisection_rounds=1,
        max_expand_steps=2,
        max_stations=2,
        max_nodes=16,
    )
    result = run_offline_scan(scan)
    diagram = DiagramConfig(
        width=320,
        height=240,
        margin_left=52,
        margin_right=84,
        margin_top=32,
        margin_bottom=38,
        grid_cols=40,
        grid_rows=30,
        fps=5,
        max_frames=6,
        hold_seconds=0.2,
    )

    artifacts = animate_scan(result, tmp_path, config=diagram)
    assert artifacts.gif_path.exists()
    assert artifacts.png_path.exists()
    assert artifacts.frame_count > 0

    width, height, decoded = read_gif(artifacts.gif_path)
    assert (width, height) == (320, 240)
    assert len(decoded) >= 2
    # 最终帧不应是纯色（要有面板、网格、区域色块）
    assert len(set(decoded[-1])) > 3
