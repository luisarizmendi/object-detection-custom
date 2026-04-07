"""
shm_frame.py — Zero-copy shared-memory frame protocol
======================================================

Layout in /dev/shm/<SHM_NAME>:

  [0:4]    uint32  sequence number   (incremented each frame)
  [4:8]    uint32  width
  [8:12]   uint32  height
  [12:16]  uint32  channels          (3 = BGR)
  [16:24]  float64 capture timestamp (time.monotonic())
  [24:28]  uint32  jpeg_size         (0 = raw BGR stored instead)
  [28:]    bytes   pixel data

Readers spin on sequence number change (or use POSIX semaphore if
FRAME_SEMAPHORE=1 is set). The slot is single-producer, multi-consumer.
"""

import mmap
import os
import struct
import time
from pathlib import Path

HEADER_FMT  = "=IIIId I"   # seq, w, h, ch, ts, jpeg_size
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 28 bytes

SHM_DIR     = "/dev/shm"
DEFAULT_NAME = os.environ.get("SHM_FRAME_NAME", "camera_frame")

# Maximum frame buffer: 4096×2160×3 ≈ 26 MB (4K); JPEG always smaller
MAX_FRAME_BYTES = int(os.environ.get("SHM_MAX_FRAME_BYTES", str(4096 * 2160 * 3)))
SHM_SIZE        = HEADER_SIZE + MAX_FRAME_BYTES


def _path(name: str) -> str:
    return os.path.join(SHM_DIR, name)


class ShmFrameWriter:
    """Single-producer writer.  Call write_frame() from the capture loop."""

    def __init__(self, name: str = DEFAULT_NAME):
        self._path = _path(name)
        self._seq  = 0

        # Open (or create) the backing file and map it
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            # Extend to SHM_SIZE if needed
            cur_size = os.fstat(fd).st_size
            if cur_size < SHM_SIZE:
                os.ftruncate(fd, SHM_SIZE)
            self._mm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED,
                                 mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)

    def write_frame(self, frame_bytes: bytes, width: int, height: int,
                    channels: int = 3, jpeg_size: int = 0) -> int:
        """Write a frame and return the new sequence number."""
        ts   = time.monotonic()
        self._seq += 1
        hdr = struct.pack(HEADER_FMT,
                          self._seq, width, height, channels, ts, jpeg_size)
        self._mm.seek(0)
        self._mm.write(hdr)
        self._mm.write(frame_bytes[:MAX_FRAME_BYTES])
        self._mm.flush()
        return self._seq

    def close(self):
        self._mm.close()
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass


class ShmFrameReader:
    """Multi-consumer reader.  Call read_frame() in a poll loop."""

    def __init__(self, name: str = DEFAULT_NAME, timeout: float = 5.0):
        self._path    = _path(name)
        self._timeout = timeout
        self._mm      = None
        self._last_seq = 0
        self._open()

    def _open(self):
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if Path(self._path).exists():
                try:
                    fd = os.open(self._path, os.O_RDONLY)
                    try:
                        size = os.fstat(fd).st_size
                        if size >= SHM_SIZE:
                            self._mm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED,
                                                 mmap.PROT_READ)
                            return
                    finally:
                        os.close(fd)
                except OSError:
                    pass
            time.sleep(0.05)
        raise TimeoutError(f"SHM frame not available after {self._timeout}s: {self._path}")

    def read_frame(self, wait_new: bool = True, poll_interval: float = 0.001):
        """
        Returns (seq, width, height, channels, timestamp, data_bytes, jpeg_size).
        If wait_new=True, blocks until a frame newer than the last seen arrives.
        """
        while True:
            self._mm.seek(0)
            raw = self._mm.read(HEADER_SIZE)
            seq, w, h, ch, ts, jpeg_size = struct.unpack(HEADER_FMT, raw)
            if not wait_new or seq != self._last_seq:
                n_bytes = jpeg_size if jpeg_size else (w * h * ch)
                data = self._mm.read(n_bytes)
                self._last_seq = seq
                return seq, w, h, ch, ts, data, jpeg_size
            time.sleep(poll_interval)

    def close(self):
        if self._mm:
            self._mm.close()
