#!/usr/bin/env python3
"""Writes a small synthetic frame capture (KYTY_FRAME_CAPTURE format, version 2) for smoke tests
of kyty_emulator --replay and the tools in this folder, without a game.

The capture maps one direct memory range, registers one 1280x720 display buffer filled with a
solid colour, flips it --flips times and keeps the colour as its reference image, so a replay
should report 100% identical pixels. With --shader it also lists one shader (code only), for
--replay-shaders; replay such a capture only with --replay-shaders.

Needs the zstd command line tool or the zstandard Python module.

  python3 tools/replay/make_test_capture.py out/capture
  kyty_emulator --replay out/capture --replay-repeat 3
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

CALL_MAGIC = 0x4C41434B
MEMORY, SUBMIT, RELEASE_REWIND, SUSPEND_POINT, VIDEO_OUT, FRAME_END, START, END, SHADER = range(1, 10)
VO_OPEN, VO_REGISTER_BUFFERS2, VO_SUBMIT_FLIP = 1, 3, 8

CHUNK = 1 << 20
BASE = 0x1000000000          # guest virtual address of the range
PHYSICAL = 0x10000000        # its direct memory offset
SIZE = 16 << 20
WIDTH, HEIGHT = 1280, 720
BGRA = bytes([0x20, 0x80, 0xC0, 0xFF])  # guest bytes of B8G8R8A8_SRGB; reads back as RGBA C0 80 20 FF


def zstd(raw: bytes) -> bytes:
    try:
        import zstandard  # pip install zstandard, where no zstd command line tool is around
        return zstandard.ZstdCompressor(level=1).compress(raw)
    except ImportError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "raw"
        src.write_bytes(raw)
        return subprocess.run(["zstd", "-q", "-1", "-c", str(src)], check=True,
                              stdout=subprocess.PIPE).stdout


class Calls:
    def __init__(self):
        self.out = bytearray()
        self.ordinal = 0

    def add(self, kind, args=(), blob=b"", time_ns=0):
        self.ordinal += 1
        packed = zstd(blob) if blob else b""
        self.out += struct.pack("<IIQQIIQQ", CALL_MAGIC, kind, self.ordinal, time_ns, len(args), 0,
                                len(blob), len(packed))
        self.out += struct.pack(f"<{len(args)}q", *args)
        self.out += packed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--flips", type=int, default=4)
    parser.add_argument("--shader", action="store_true", help="list one shader (replay with --replay-shaders only)")
    args = parser.parse_args()
    out: Path = args.output
    if (out / "header.json").exists():
        sys.exit(f"{out} already holds a capture")
    out.mkdir(parents=True, exist_ok=True)

    # ranges.bin: count, CaptureRange[] (start, size, offset, protection, memory_type, type, reserved),
    # count, CapturePhysical[] (start, size, memory_type, kind).
    ranges = struct.pack("<I", 1) + struct.pack("<QQQiiII", BASE, SIZE, PHYSICAL, 0x33, 3, 0, 0)
    ranges += struct.pack("<I", 1) + struct.pack("<QQiI", PHYSICAL, SIZE, 3, 0)
    (out / "ranges.bin").write_bytes(ranges)

    # The display buffer fills the first 8 MiB; a solid colour is the same in any tiling.
    display = BGRA * (8 * CHUNK // 4)
    code = bytes(range(256)) * 4  # 1 KiB of "shader code" at BASE + 8 MiB
    snapshot = bytearray()
    for index in range(SIZE // CHUNK):
        address = BASE + index * CHUNK
        if index < 8:
            raw = display[index * CHUNK:(index + 1) * CHUNK]
        elif index == 8 and args.shader:
            raw = code + bytes(CHUNK - len(code))
        else:
            snapshot += struct.pack("<QII", address, CHUNK, 0)
            continue
        packed = zstd(raw)
        snapshot += struct.pack("<QII", address, CHUNK, len(packed)) + packed
    (out / "snapshot.bin").write_bytes(snapshot)

    calls = Calls()
    handle = 1
    calls.add(VIDEO_OUT, (VO_OPEN, 0, 0, handle))
    buffers = struct.pack("<QQQQ", BASE, 0, 0, 0)
    attribute = struct.pack("<IIIIIIQQQII3Q", 0, 0, 0, WIDTH, HEIGHT, 0, 0, 0x8000000000000000, 0, 0, 0,
                            0, 0, 0)
    assert len(attribute) == 80
    calls.add(VIDEO_OUT, (VO_REGISTER_BUFFERS2, handle, 0, 0, 1, 0), buffers + attribute)
    if args.shader:
        # address, type, user data, input semantics, count, size, scratch
        calls.add(SHADER, (BASE + 8 * CHUNK, 7, 0, 0, 0, len(code), 0))
    calls.add(START, (1,))
    for flip in range(args.flips):
        calls.add(VIDEO_OUT, (VO_SUBMIT_FLIP, handle, 0, 1, flip), time_ns=flip * 16_666_667)
        calls.add(FRAME_END, (flip + 2,))
    calls.add(END)
    (out / "calls.bin").write_bytes(bytes(calls.out))

    rgba = bytes([BGRA[2], BGRA[1], BGRA[0], 0xFF]) * (WIDTH * HEIGHT)
    (out / "reference.rgba").write_bytes(rgba)
    header = {"magic": "kyty-frame-capture", "version": 2, "start_flip": 1, "frames": args.flips,
              "calls": calls.ordinal, "ranges": 1, "guest_bytes": SIZE, "snapshot_bytes": len(snapshot),
              "call_bytes": len(calls.out), "reference_width": WIDTH, "reference_height": HEIGHT,
              "synthetic": True}
    (out / "header.json").write_text(json.dumps(header, separators=(",", ":")) + "\n")
    print(f"synthetic capture written to {out}")


if __name__ == "__main__":
    main()
