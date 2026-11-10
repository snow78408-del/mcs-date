"""二相图动画：把扫描过程中逐步累积的节点和边界渲染成“缓慢形成”的流型图。

不依赖 matplotlib：图形用 OpenCV 绘制，GIF 由内置纯 Python 编码器写出，
因此离线环境也能直接生成交付物（``phase_diagram.gif``）。若系统装有
Pillow，则用微软雅黑渲染中文标签；否则自动退化为英文标签。
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .models import BoundaryBracket, NodeRecord, ScanResult
from .phase import FlowRegime
from .strategy import station_boundaries


# ---------------------------------------------------------------------------
# 配色
# ---------------------------------------------------------------------------

REGIME_COLORS_BGR: dict[FlowRegime, tuple[int, int, int]] = {
    FlowRegime.DRIPPING: (176, 206, 233),      # 浅蓝
    FlowRegime.JETTING: (232, 201, 160),       # 浅橙
    FlowRegime.THREADING: (171, 219, 174),     # 浅绿
    FlowRegime.TUBING: (214, 181, 224),        # 浅紫
    FlowRegime.DISPLACEMENT: (196, 196, 236),  # 浅粉紫
    FlowRegime.UNKNOWN: (228, 228, 228),
}
NODE_COLORS_BGR: dict[FlowRegime, tuple[int, int, int]] = {
    FlowRegime.DRIPPING: (150, 90, 40),
    FlowRegime.JETTING: (40, 120, 210),
    FlowRegime.THREADING: (40, 140, 70),
    FlowRegime.TUBING: (110, 60, 160),
    FlowRegime.DISPLACEMENT: (80, 60, 190),
    FlowRegime.UNKNOWN: (110, 110, 110),
}
LOWER_BOUNDARY_BGR = (190, 110, 40)   # 液滴下边界（r=ratio_high，转射流）
UPPER_BOUNDARY_BGR = (40, 90, 190)    # 液滴上边界（r=ratio_low，转栓塞）
BG_BGR = (247, 247, 247)
PANEL_BGR = (255, 255, 255)
TEXT_BGR = (45, 45, 45)
GRID_BGR = (214, 214, 214)
# 已测节点多远之外的区域不参与填充（log10 单位）；让动画开头保留“未测”底色，
# 而不是把整个象限涂成第一个节点的流型。
MAX_FILL_DIST_LOG10 = 0.55
AXIS_BGR = (90, 90, 90)
HIGHLIGHT_BGR = (20, 20, 20)

LABEL_ZH: dict[FlowRegime, str] = {
    FlowRegime.DRIPPING: "滴落",
    FlowRegime.JETTING: "射流",
    FlowRegime.THREADING: "束流",
    FlowRegime.TUBING: "栓塞",
    FlowRegime.DISPLACEMENT: "浸润",
    FlowRegime.UNKNOWN: "未知",
}
LABEL_EN: dict[FlowRegime, str] = {
    FlowRegime.DRIPPING: "dripping",
    FlowRegime.JETTING: "jetting",
    FlowRegime.THREADING: "threading",
    FlowRegime.TUBING: "tubing",
    FlowRegime.DISPLACEMENT: "wetting",
    FlowRegime.UNKNOWN: "unknown",
}


def _has_pillow() -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except Exception:
        return False
    return True


_HAS_PILLOW = _has_pillow()
_CJK_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_font_cache: dict[int, object] = {}


def _load_font(size: int):
    if not _HAS_PILLOW:
        return None
    if size in _font_cache:
        return _font_cache[size]
    from PIL import ImageFont

    for path in _CJK_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


# ---------------------------------------------------------------------------
# 坐标轴
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogAxis:
    vmin: float
    vmax: float
    px0: float
    px1: float

    def __post_init__(self) -> None:
        if not (self.vmin > 0 and self.vmax > self.vmin):
            raise ValueError("LogAxis 需要 0 < vmin < vmax")

    @property
    def lo(self) -> float:
        return math.log10(self.vmin)

    @property
    def hi(self) -> float:
        return math.log10(self.vmax)

    def to_px(self, value: float) -> float:
        value = min(max(value, self.vmin), self.vmax)
        frac = (math.log10(value) - self.lo) / (self.hi - self.lo)
        return self.px0 + frac * (self.px1 - self.px0)

    def ticks(self) -> list[float]:
        ticks: list[float] = []
        for decade in range(int(math.floor(self.lo)), int(math.ceil(self.hi)) + 1):
            base = 10.0 ** decade
            for mult in (1.0, 2.0, 5.0):
                value = mult * base
                if self.vmin <= value <= self.vmax:
                    ticks.append(value)
        if len(ticks) < 3:
            ticks = [self.vmin, math.sqrt(self.vmin * self.vmax), self.vmax]
        return sorted(set(ticks))


def _fmt_tick(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:g}"


# ---------------------------------------------------------------------------
# 绘图画布
# ---------------------------------------------------------------------------


@dataclass
class DiagramConfig:
    width: int = 1000
    height: int = 660
    margin_left: int = 96
    margin_right: int = 220
    margin_top: int = 64
    margin_bottom: int = 74
    grid_cols: int = 150
    grid_rows: int = 110
    fps: float = 8.0
    max_frames: int = 90
    hold_seconds: float = 2.5

    @property
    def plot_width(self) -> int:
        return self.width - self.margin_left - self.margin_right

    @property
    def plot_height(self) -> int:
        return self.height - self.margin_top - self.margin_bottom


class DiagramRenderer:
    def __init__(self, result: ScanResult, config: DiagramConfig | None = None) -> None:
        self.result = result
        self.config = config or DiagramConfig()
        self.q1_axis, self.q2_axis = self._build_axes()
        self._text_ops: list[tuple[str, int, int, int, tuple[int, int, int], bool]] = []

    def _build_axes(self) -> tuple[LogAxis, LogAxis]:
        cfg = self.config
        nodes = self.result.nodes
        q1_values = [n.q1 for n in nodes] or [self.result.config.center_q1]
        q2_values = [n.q2 for n in nodes] or [self.result.config.center_q2]
        q1_lo = max(1e-3, min(q1_values) / 1.35)
        q1_hi = max(q1_values) * 1.35
        q2_lo = max(1e-3, min(q2_values) / 1.5)
        q2_hi = max(q2_values) * 1.5
        x_axis = LogAxis(q1_lo, q1_hi, cfg.margin_left, cfg.margin_left + cfg.plot_width)
        y_axis = LogAxis(q2_lo, q2_hi, cfg.margin_top + cfg.plot_height, cfg.margin_top)
        return x_axis, y_axis

    # -- 文本 ---------------------------------------------------------------
    def _text(self, text: str, x: int, y: int, size: int = 16,
              color: tuple[int, int, int] = TEXT_BGR, bold: bool = False) -> None:
        self._text_ops.append((text, x, y, size, color, bold))

    def _flush_text(self, canvas: np.ndarray) -> np.ndarray:
        cfg = self.config
        if _HAS_PILLOW:
            from PIL import Image, ImageDraw

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            for text, x, y, size, color, bold in self._text_ops:
                font = _load_font(size + (1 if bold else 0))
                draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
            out = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        else:
            out = canvas
            for text, x, y, size, color, _bold in self._text_ops:
                ascii_text = text if _is_ascii(text) else text.encode("ascii", "replace").decode()
                scale = max(0.32, size / 34.0)
                cv2.putText(out, ascii_text, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 1, cv2.LINE_AA)
        self._text_ops.clear()
        return out

    # -- 区域填充 -----------------------------------------------------------
    def _regime_grid(self, nodes: Sequence[NodeRecord]) -> np.ndarray:
        cfg = self.config
        cols, rows = cfg.grid_cols, cfg.grid_rows
        grid = np.full((rows, cols, 3), 255, dtype=np.uint8)
        if not nodes:
            return grid
        log_nodes = np.array(
            [[math.log10(n.q1), math.log10(n.q2)] for n in nodes], dtype=np.float64
        )
        xs = np.linspace(self.q1_axis.lo, self.q1_axis.hi, cols)
        ys = np.linspace(self.q2_axis.hi, self.q2_axis.lo, rows)  # 图像行向下
        gx, gy = np.meshgrid(xs, ys)
        points = np.stack([gx.ravel(), gy.ravel()], axis=1)
        best = np.full(points.shape[0], np.inf)
        labels = np.zeros(points.shape[0], dtype=np.int16)
        for idx, node in enumerate(nodes):
            dx = points[:, 0] - log_nodes[idx, 0]
            dy = points[:, 1] - log_nodes[idx, 1]
            dist = dx * dx + dy * dy
            better = dist < best
            best[better] = dist[better]
            labels[better] = idx
        color_lut = np.array(
            [REGIME_COLORS_BGR[node.regime] for node in nodes], dtype=np.uint8
        )
        flat = np.full(
            (points.shape[0], 3),
            REGIME_COLORS_BGR[FlowRegime.UNKNOWN],
            dtype=np.uint8,
        )
        near = np.sqrt(best) <= MAX_FILL_DIST_LOG10
        flat[near] = color_lut[labels[near]]
        return flat.reshape(rows, cols, 3)

    # -- 边界 ---------------------------------------------------------------
    def _boundary_points(
        self, visible_bracket_ids: set[int]
    ) -> dict[str, list[tuple[float, float]]]:
        table: dict[str, dict[float, BoundaryBracket]] = {"droplet_lower": {}, "droplet_upper": {}}
        for bracket in self.result.brackets:
            if id(bracket) not in visible_bracket_ids:
                continue
            if bracket.kind not in table:
                continue
            entry = table[bracket.kind]
            current = entry.get(bracket.station_q1)
            if current is None or bracket.relative_width < current.relative_width:
                entry[bracket.station_q1] = bracket
        points: dict[str, list[tuple[float, float]]] = {}
        for kind, entry in table.items():
            points[kind] = sorted(
                ((q1, b.boundary_q2) for q1, b in entry.items()), key=lambda item: item[0]
            )
        return points

    # -- 主渲染 -------------------------------------------------------------
    def render(
        self,
        nodes: Sequence[NodeRecord],
        bracket_ids: set[int],
        *,
        final: bool = False,
        progress_text: str = "",
    ) -> np.ndarray:
        cfg = self.config
        canvas = np.full((cfg.height, cfg.width, 3), BG_BGR, dtype=np.uint8)
        # 绘图区面板
        x0, x1 = cfg.margin_left, cfg.margin_left + cfg.plot_width
        y0, y1 = cfg.margin_top, cfg.margin_top + cfg.plot_height
        cv2.rectangle(canvas, (x0, y0), (x1, y1), PANEL_BGR, -1)

        grid = self._regime_grid(nodes)
        grid = cv2.resize(grid, (cfg.plot_width, cfg.plot_height), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y1, x0:x1] = grid

        # 网格与刻度
        for tick in self.q1_axis.ticks():
            px = int(round(self.q1_axis.to_px(tick)))
            cv2.line(canvas, (px, y0), (px, y1), GRID_BGR, 1, cv2.LINE_8)
            self._text(_fmt_tick(tick), px - 10, y1 + 8, 14, TEXT_BGR)
        for tick in self.q2_axis.ticks():
            py = int(round(self.q2_axis.to_px(tick)))
            cv2.line(canvas, (x0, py), (x1, py), GRID_BGR, 1, cv2.LINE_8)
            label = _fmt_tick(tick)
            self._text(label, x0 - 12 - 7 * len(label), py - 8, 14, TEXT_BGR)

        cv2.rectangle(canvas, (x0, y0), (x1, y1), AXIS_BGR, 1, cv2.LINE_8)

        # 边界折线
        boundaries = self._boundary_points(bracket_ids)
        self._draw_boundary(canvas, boundaries.get("droplet_lower", []), LOWER_BOUNDARY_BGR)
        self._draw_boundary(canvas, boundaries.get("droplet_upper", []), UPPER_BOUNDARY_BGR)

        # 节点
        for idx, node in enumerate(nodes):
            px = int(round(self.q1_axis.to_px(node.q1)))
            py = int(round(self.q2_axis.to_px(node.q2)))
            color = NODE_COLORS_BGR[node.regime]
            r = 5 if idx < len(nodes) - 1 else 7
            cv2.circle(canvas, (px, py), r, color, -1, cv2.LINE_8)
            cv2.circle(canvas, (px, py), r, (255, 255, 255), 1, cv2.LINE_8)
            if idx == len(nodes) - 1 and not final:
                cv2.circle(canvas, (px, py), r + 6, HIGHLIGHT_BGR, 2, cv2.LINE_8)

        self._draw_legend(canvas, nodes)

        # 标题与状态
        title = "Two-phase flow map of the square microchannel" if not _HAS_PILLOW else "方形微通道二相图 · 蜘蛛式扫描"
        self._text(title, x0, 22, 22, TEXT_BGR, bold=True)
        sub = (
            f"Q1 = continuous oil (side inlets)   Q2 = dispersed water (center inlet)   "
            f"nodes={len(nodes)}/{len(self.result.nodes)}"
        )
        self._text(sub if not _HAS_PILLOW else
                   f"Q1 油相（两侧）  Q2 水相（中心）  节点 {len(nodes)}/{len(self.result.nodes)}",
                   x0, 46, 14, AXIS_BGR)
        if progress_text:
            self._text(progress_text, x1 - 8 - 10 * len(progress_text), 22, 16, AXIS_BGR)
        if final:
            note = "final map" if not _HAS_PILLOW else "最终二相图"
            self._text(note, x1 - 8 - 14 * len(note), 22, 16, (30, 120, 60), bold=True)

        # 轴标题
        self._text("Q1  [µL/min]" if _HAS_PILLOW else "Q1  [uL/min]",
                   (x0 + x1) // 2 - 40, cfg.height - 28, 16, TEXT_BGR)
        ylabel = "Q2  [µL/min]" if _HAS_PILLOW else "Q2  [uL/min]"
        self._text(ylabel, 10, y0 + cfg.plot_height // 2, 16, TEXT_BGR)

        return self._flush_text(canvas)

    def _draw_boundary(self, canvas: np.ndarray, points: list[tuple[float, float]],
                       color: tuple[int, int, int]) -> None:
        if len(points) < 2:
            for q1, q2 in points:
                px, py = int(self.q1_axis.to_px(q1)), int(self.q2_axis.to_px(q2))
                cv2.drawMarker(canvas, (px, py), color, cv2.MARKER_CROSS, 12, 2, cv2.LINE_8)
            return
        pts = np.array(
            [
                [int(round(self.q1_axis.to_px(q1))), int(round(self.q2_axis.to_px(q2)))]
                for q1, q2 in points
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_8)
        for px, py in pts:
            cv2.drawMarker(canvas, (int(px), int(py)), color, cv2.MARKER_CROSS, 10, 2, cv2.LINE_8)

    def _draw_legend(self, canvas: np.ndarray, nodes: Sequence[NodeRecord]) -> None:
        cfg = self.config
        x = cfg.margin_left + cfg.plot_width + 18
        y = cfg.margin_top + 6
        self._text("流型 / regimes" if _HAS_PILLOW else "regimes", x, y, 15, TEXT_BGR, bold=True)
        y += 26
        present = {node.regime for node in nodes} or {FlowRegime.UNKNOWN}
        for regime in (
            FlowRegime.THREADING,
            FlowRegime.JETTING,
            FlowRegime.DRIPPING,
            FlowRegime.TUBING,
            FlowRegime.DISPLACEMENT,
            FlowRegime.UNKNOWN,
        ):
            if regime not in present and regime is not FlowRegime.UNKNOWN:
                continue
            color = REGIME_COLORS_BGR[regime]
            cv2.rectangle(canvas, (x, y), (x + 18, y + 18), color, -1, cv2.LINE_8)
            cv2.rectangle(canvas, (x, y), (x + 18, y + 18), AXIS_BGR, 1, cv2.LINE_8)
            label = (
                f"{LABEL_ZH[regime]}  {LABEL_EN[regime]}" if _HAS_PILLOW else LABEL_EN[regime]
            )
            self._text(label, x + 26, y + 1, 14, TEXT_BGR)
            count = sum(1 for node in nodes if node.regime is regime)
            self._text(str(count), x + 170, y + 1, 13, AXIS_BGR)
            y += 26

        y += 12
        self._text("边界 / boundaries" if _HAS_PILLOW else "boundaries", x, y, 15, TEXT_BGR, bold=True)
        y += 26
        for color, label in (
            (LOWER_BOUNDARY_BGR, "液滴下边界" if _HAS_PILLOW else "droplet lower"),
            (UPPER_BOUNDARY_BGR, "液滴上边界" if _HAS_PILLOW else "droplet upper"),
        ):
            cv2.line(canvas, (x, y + 9), (x + 24, y + 9), color, 3, cv2.LINE_8)
            self._text(label, x + 32, y, 14, TEXT_BGR)
            y += 28

        y += 8
        hours = self.result.total_duration_s / 3600.0
        lines = (
            [f"每节点 {self.result.config.node_duration_s:g}s · 计划 {hours:.1f}h"]
            if _HAS_PILLOW
            else [f"node {self.result.config.node_duration_s:g}s",
                  f"planned {hours:.1f} h"]
        )
        for line in lines:
            self._text(line, x, y, 13, AXIS_BGR)
            y += 20


# ---------------------------------------------------------------------------
# GIF 编码（纯 Python，无外部依赖）
# ---------------------------------------------------------------------------


class GifWriter:
    """最小 GIF89a 写入器：固定调色板 + LZW，循环播放。"""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        palette: Sequence[tuple[int, int, int]],
        *,
        delay_cs: int = 12,
        loop: int = 0,
    ) -> None:
        if not 1 < len(palette) <= 256:
            raise ValueError("调色板颜色数必须在 2..256")
        self.path = Path(path)
        self.width = width
        self.height = height
        self.delay_cs = max(1, delay_cs)
        self.loop = loop
        self._palette = list(palette) + [(0, 0, 0)] * (256 - len(palette))
        self._frames = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("wb")
        self._write_header()

    def _write_header(self) -> None:
        self._handle.write(b"GIF89a")
        self._handle.write(struct.pack("<HH", self.width, self.height))
        self._handle.write(bytes([0xF7, 0, 0]))  # 全局调色板 256 色
        for r, g, b in self._palette:
            self._handle.write(bytes([r, g, b]))
        # Netscape 循环扩展
        self._handle.write(b"\x21\xff\x0bNETSCAPE2.0\x03\x01")
        self._handle.write(struct.pack("<H", self.loop))
        self._handle.write(b"\x00")

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"帧尺寸 {frame_bgr.shape[:2]} 与 GIF 尺寸 {(self.height, self.width)} 不一致"
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        indices = self._quantize(rgb)
        # 图形控制扩展
        self._handle.write(b"\x21\xf9\x04\x04")
        self._handle.write(struct.pack("<H", self.delay_cs))
        self._handle.write(b"\x00\x00")
        # 图像描述符
        self._handle.write(b"\x2c")
        self._handle.write(struct.pack("<HHHH", 0, 0, self.width, self.height))
        self._handle.write(b"\x00")
        self._handle.write(bytes([8]))  # LZW 最小码长
        for block in _lzw_encode(indices.tobytes(), 8):
            self._handle.write(bytes([len(block)]))
            self._handle.write(block)
        self._handle.write(b"\x00")
        self._frames += 1

    def _quantize(self, rgb: np.ndarray) -> np.ndarray:
        # 用 int32：255^2=65025 会溢出 int16，导致距离出现负值、argmin 选错颜色
        palette = np.array(self._palette, dtype=np.int32)
        flat = rgb.reshape(-1, 3)
        # 先按唯一颜色量化，再映射回像素，避免逐帧做 65 万 × 256 的暴力距离
        unique, inverse = np.unique(flat, axis=0, return_inverse=True)
        diff = unique[:, None, :].astype(np.int32) - palette[None, :, :]
        dist = (diff * diff).sum(axis=2)
        indices = np.argmin(dist, axis=1).astype(np.uint8)
        return indices[inverse].reshape(rgb.shape[:2])

    def close(self) -> None:
        self._handle.write(b"\x3b")
        self._handle.close()

    def __enter__(self) -> "GifWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _lzw_encode(data: bytes, min_code_size: int) -> Iterable[bytes]:
    """GIF LZW 编码，返回按 255 字节切块的字节串序列。

    码宽必须按“发射当时”的宽度逐码打包：字典每新增一项就可能加宽码宽，
    若先收集全部码再统一打包，早期码会用最终码宽写出，解码器立即失步。
    """
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    next_code = end_code + 1
    code_size = min_code_size + 1
    dictionary: dict[tuple[int, ...], int] = {(i,): i for i in range(clear_code)}

    blocks: list[bytes] = []
    block = bytearray()
    bits = 0
    nbits = 0

    def emit(code: int) -> None:
        nonlocal bits, nbits
        bits |= code << nbits
        nbits += code_size
        while nbits >= 8:
            block.append(bits & 0xFF)
            bits >>= 8
            nbits -= 8
            if len(block) == 255:
                blocks.append(bytes(block))
                block.clear()

    emit(clear_code)
    buffer: tuple[int, ...] = ()
    for value in data:
        candidate = buffer + (value,)
        if candidate in dictionary:
            buffer = candidate
            continue
        emit(dictionary[buffer])
        if next_code < 4096:
            dictionary[candidate] = next_code
            next_code += 1
            if next_code > (1 << code_size) and code_size < 12:
                code_size += 1
        else:
            emit(clear_code)
            dictionary = {(i,): i for i in range(clear_code)}
            next_code = end_code + 1
            code_size = min_code_size + 1
        buffer = (value,)
    if buffer:
        emit(dictionary[buffer])
    emit(end_code)

    if nbits:
        block.append(bits & 0xFF)
    if block:
        blocks.append(bytes(block))
    return blocks


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


@dataclass
class AnimationArtifacts:
    gif_path: Path
    png_path: Path
    mp4_path: Path | None = None
    frame_count: int = 0


def _palette() -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    for color in (
        BG_BGR, PANEL_BGR, TEXT_BGR, GRID_BGR, AXIS_BGR, HIGHLIGHT_BGR,
        LOWER_BOUNDARY_BGR, UPPER_BOUNDARY_BGR,
    ):
        colors.append((int(color[2]), int(color[1]), int(color[0])))
    for table in (REGIME_COLORS_BGR, NODE_COLORS_BGR):
        for color in table.values():
            colors.append((int(color[2]), int(color[1]), int(color[0])))
    # 去重并补充常用灰阶
    unique: list[tuple[int, int, int]] = []
    for color in colors + [(255, 255, 255), (0, 0, 0), (120, 120, 120), (200, 200, 200)]:
        if color not in unique:
            unique.append(color)
    return unique


def animate_scan(
    result: ScanResult,
    out_dir: str | Path = "out",
    *,
    gif_name: str = "phase_diagram.gif",
    png_name: str = "phase_diagram_final.png",
    mp4_name: str | None = None,
    config: DiagramConfig | None = None,
) -> AnimationArtifacts:
    """渲染扫描过程动画（GIF）+ 最终静态图（PNG），可选 MP4。"""
    cfg = config or DiagramConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = DiagramRenderer(result, cfg)

    # 帧序列：每个节点一帧（超出上限时均匀抽样），最后追加定格帧
    nodes = result.nodes
    if not nodes:
        raise ValueError("扫描结果没有节点，无法生成动画")
    visible_order = list(range(len(nodes)))
    if len(visible_order) > cfg.max_frames:
        stride = len(visible_order) / float(cfg.max_frames)
        visible_order = [visible_order[min(len(nodes) - 1, int(i * stride))] for i in range(cfg.max_frames)]
        visible_order = sorted(set(visible_order))

    palette = _palette()
    delay_cs = max(1, int(round(100.0 / cfg.fps)))
    gif_path = out_dir / gif_name
    png_path = out_dir / png_name
    mp4_path = out_dir / mp4_name if mp4_name else None

    video = None
    if mp4_path is not None:
        video = cv2.VideoWriter(
            str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, cfg.fps),
            (cfg.width, cfg.height),
        )
        if not video.isOpened():
            video.release()
            video = None
            mp4_path = None

    frames = 0
    last_frame: np.ndarray | None = None
    with GifWriter(gif_path, cfg.width, cfg.height, palette, delay_cs=delay_cs) as gif:
        for order_index in visible_order:
            prefix = nodes[: order_index + 1]
            visible_ids = {
                id(bracket)
                for bracket in result.brackets
                if max(bracket.inside.index, bracket.outside.index) <= prefix[-1].index
            }
            plt_text = f"{prefix[-1].index}/{len(nodes)}"
            frame = renderer.render(prefix, visible_ids, progress_text=plt_text)
            gif.add_frame(frame)
            if video is not None:
                video.write(frame)
            frames += 1
            last_frame = frame

        # 最终帧：全部节点 + 最终边界 + 定格
        if last_frame is None:
            raise RuntimeError("未渲染出任何帧")
        final_frame = renderer.render(nodes, {id(b) for b in result.brackets}, final=True)
        cv2.imwrite(str(png_path), final_frame)
        hold_frames = max(1, int(round(cfg.hold_seconds * cfg.fps)))
        for _ in range(hold_frames):
            gif.add_frame(final_frame)
            if video is not None:
                video.write(final_frame)
            frames += 1

    if video is not None:
        video.release()

    return AnimationArtifacts(
        gif_path=gif_path,
        png_path=png_path,
        mp4_path=mp4_path,
        frame_count=frames,
    )
