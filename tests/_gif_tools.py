"""测试辅助：极简 GIF 解析与 LZW 解码，用来验证自写 GifWriter 的输出。"""

from __future__ import annotations

import struct
from pathlib import Path


def lzw_decode(payload: bytes, min_code_size: int) -> bytes:
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    table: list[bytes] = [bytes([i]) for i in range(clear_code)] + [b"", b""]
    out = bytearray()
    data = int.from_bytes(payload, "little")
    total_bits = len(payload) * 8
    position = 0
    previous: bytes | None = None

    while position + code_size <= total_bits:
        code = (data >> position) & ((1 << code_size) - 1)
        position += code_size
        if code == clear_code:
            table = table[: clear_code + 2]
            code_size = min_code_size + 1
            previous = None
            continue
        if code == end_code:
            break
        if code < len(table) and table[code]:
            entry = table[code]
        elif code == len(table) and previous is not None:
            entry = previous + previous[:1]
        else:
            raise ValueError(f"bad LZW code {code}, table size {len(table)}, code size {code_size}")
        out.extend(entry)
        if previous is not None:
            table.append(previous + entry[:1])
            if len(table) == (1 << code_size) and code_size < 12:
                code_size += 1
        previous = entry
    return bytes(out)


def read_gif(path: str | Path) -> tuple[int, int, list[bytes]]:
    """返回 (宽, 高, 每帧解码后的像素索引字节串)。"""
    data = Path(path).read_bytes()
    assert data[:6] in (b"GIF89a", b"GIF87a"), "不是 GIF87a/89a"
    position = 6
    width, height, packed = struct.unpack("<HHB", data[position : position + 5])
    position += 5
    position += 2  # 背景色索引 + 像素宽高比
    if packed & 0x80:
        position += 3 * (2 ** ((packed & 0x07) + 1))

    frames: list[bytes] = []
    while position < len(data):
        marker = data[position]
        position += 1
        if marker == 0x3B:  # trailer
            break
        if marker == 0x21:  # 扩展块
            position += 1  # 扩展标签
            while True:
                size = data[position]
                position += 1
                if size == 0:
                    break
                position += size
            continue
        if marker == 0x2C:  # 图像描述符
            _, _, frame_w, frame_h, flags = struct.unpack(
                "<HHHHB", data[position : position + 9]
            )
            position += 9
            if flags & 0x80:
                position += 3 * (2 ** ((flags & 0x07) + 1))
            min_code_size = data[position]
            position += 1
            payload = bytearray()
            while True:
                size = data[position]
                position += 1
                if size == 0:
                    break
                payload.extend(data[position : position + size])
                position += size
            indices = lzw_decode(bytes(payload), min_code_size)
            assert len(indices) == frame_w * frame_h, (
                f"帧像素数 {len(indices)} != {frame_w} x {frame_h}"
            )
            frames.append(indices)
            continue
        raise AssertionError(f"未知 GIF 块标记 0x{marker:02x}")
    return width, height, frames
