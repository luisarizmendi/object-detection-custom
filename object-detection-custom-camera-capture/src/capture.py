#!/usr/bin/env python3
"""
object-detection-custom-camera-capture — USB camera (or video file fallback) → shared memory
=====================================================================

Logic ported from the original camera-gateway-rtsp/stream.py:
  1. Enumerate /dev/video* devices
  2. For each device, use v4l2-ctl --list-formats-ext to discover
     supported modes (format × resolution × fps)
  3. Select best mode: closest fps to CAM_TARGET_FPS, largest frame,
     MJPG > YUYV > others
  4. Probe with a single ffmpeg -vframes 1 test (with EPROTO retry)
  5. Capture loop via OpenCV CAP_V4L2 with detected parameters
  6. If no camera works → loop video files from VID_DIR

Frames are written to POSIX shared memory (shm_frame.py) so that
inference containers can read them with zero copies.

Environment variables
---------------------
CAMERA_DEVICE        Force a specific /dev/videoN (default: auto-detect)
CAM_TARGET_FPS       Desired FPS for mode selection (default: 30)
CAM_FRAMERATE        Override FPS to pass to v4l2 (default: auto from v4l2-ctl)
CAM_RESOLUTION       Override resolution WxH, e.g. 1280x720 (default: auto)
CAM_FORMAT           Force pixel format: MJPG | YUYV (default: auto)
DEVICE_PROBE_TIMEOUT Seconds per ffmpeg probe attempt (default: 5)
VID_DIR              Directory with fallback video files (default: /videos)
SHM_FRAME_NAME       Shared memory slot name (default: camera_frame)
VIEWER_JPEG_QUALITY  JPEG quality stored in SHM: 0=raw BGR, 1-95=JPEG
                     (default: 0 — raw BGR; inference containers decode natively)
LOG_LEVEL            DEBUG / INFO / WARNING (default: INFO)
VERBOSE_STATS        Log periodic fps/resolution/payload stats (default: 0 — off)
                     Set to "1" to enable. Useful for debugging; noisy in production.
CAMERA_RETRY_INTERVAL  Seconds between camera re-probe attempts when the device is
                     present but failing, before falling back to video (default: 15)
"""

import glob
import logging
import os
import subprocess
import sys
import time
from fractions import Fraction

import cv2
import numpy as np

from shm_frame import ShmFrameWriter

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_DEVICE        = os.environ.get("CAMERA_DEVICE", "")
CAM_TARGET_FPS       = int(os.environ.get("CAM_TARGET_FPS", "30"))
CAM_FRAMERATE        = os.environ.get("CAM_FRAMERATE", "")     # "" = use detected
CAM_RESOLUTION       = os.environ.get("CAM_RESOLUTION", "")   # "" = use detected
CAM_FORMAT           = os.environ.get("CAM_FORMAT", "")        # "" = auto
DEVICE_PROBE_TIMEOUT = int(os.environ.get("DEVICE_PROBE_TIMEOUT", "5"))
VID_DIR              = os.environ.get("VID_DIR", "/videos")
JPEG_QUALITY         = int(os.environ.get("VIEWER_JPEG_QUALITY", "0"))
LOG_LEVEL            = os.environ.get("LOG_LEVEL", "INFO").upper()
# Set to "1" to log periodic fps/payload stats; off by default to keep logs clean
VERBOSE_STATS        = os.environ.get("VERBOSE_STATS", "0").strip() in ("1", "true", "yes")
# How long to wait before re-probing a camera that failed, or re-trying after video fallback (s)
CAMERA_RETRY_INTERVAL = int(os.environ.get("CAMERA_RETRY_INTERVAL", "15"))
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Formats that ffmpeg v4l2 demuxer understands
_FMT_MAP = {
    "yuyv": "yuyv422", "yuyv422": "yuyv422",
    "mjpg":  "mjpeg",  "mjpeg":   "mjpeg",
    "h264":  "h264",
    "nv12":  "nv12",
    "rgb3":  "rgb24",  "rgb24":   "rgb24",
}

# Map ffmpeg format name → OpenCV fourcc string
_FFMT_TO_FOURCC = {
    "mjpeg":   "MJPG",
    "yuyv422": "YUYV",
    "h264":    "H264",
    "nv12":    "NV12",
    "rgb24":   "RGB3",
}


# ── v4l2-ctl helpers (ported 1:1 from original stream.py) ────────────────────

def _parse_fraction(s: str) -> float:
    s = s.strip()
    try:
        return float(Fraction(s)) if "/" in s else float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _pixel_count(size: str) -> int:
    try:
        w, h = size.lower().split("x")
        return int(w) * int(h)
    except Exception:
        return 0


def enumerate_camera_modes(device: str) -> list:
    """Return [{fmt, size, fps}, …] via v4l2-ctl --list-formats-ext."""
    modes = []
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", device, "--list-formats-ext"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            log.warning("v4l2-ctl returned error for %s", device)
            return modes
    except (subprocess.TimeoutExpired, FileNotFoundError):
        log.warning("v4l2-ctl not available or timed out for %s", device)
        return modes

    current_fmt = current_size = ""
    for line in result.stdout.splitlines():
        s = line.strip()
        if s.startswith("[") and "'" in s:
            try:
                raw = s.split("'")[1].strip().lower()
                current_fmt  = _FMT_MAP.get(raw, "")
                current_size = ""
            except IndexError:
                current_fmt = ""
            continue
        if not current_fmt:
            continue
        if s.startswith("Size:") and "Discrete" in s:
            try:
                current_size = s.split("Discrete")[1].strip()
            except IndexError:
                current_size = ""
            continue
        if not current_size:
            continue
        if s.startswith("Interval:") and "fps" in s:
            try:
                fps_str = s.split("(")[1].split("fps")[0].strip()
                fps_val = _parse_fraction(fps_str)
                if fps_val > 0:
                    modes.append({"fmt": current_fmt,
                                  "size": current_size,
                                  "fps":  fps_val})
            except (IndexError, ValueError):
                pass

    log.info("v4l2-ctl: %d usable mode(s) for %s", len(modes), device)
    if modes:
        # Print a readable summary table grouped by pixel format
        by_fmt: dict = {}
        for m in modes:
            by_fmt.setdefault(m["fmt"], []).append(m)

        log.info("  +-- Supported video modes on %s --+", device)
        for fmt, fmt_modes in sorted(by_fmt.items()):
            by_size: dict = {}
            for m in fmt_modes:
                by_size.setdefault(m["size"], []).append(m["fps"])
            for size, fps_list in sorted(by_size.items(),
                                         key=lambda kv: -_pixel_count(kv[0])):
                fps_str = "  ".join(f"{f:.0f}" for f in sorted(fps_list, reverse=True))
                log.info("  |  %-8s  %-12s  fps: %s", fmt, size, fps_str)
        log.info("  +--------------------------------------+")
    return modes


def select_best_mode(modes: list, target_fps: float = 30.0) -> "dict | None":
    """
    Priority:
      1. fps closest to target_fps
      2. largest frame (pixel count)
      3. mjpeg > yuyv422 > others
    """
    if not modes:
        return None
    _FMT_RANK = {"mjpeg": 0, "yuyv422": 1}

    def score(m):
        return (abs(m["fps"] - target_fps),
                -_pixel_count(m["size"]),
                _FMT_RANK.get(m["fmt"], 99))

    ranked = sorted(modes, key=score)
    best = ranked[0]

    log.info(
        "  => Selected mode: fmt=%-8s  size=%-12s  fps=%.1f"
        "  (target=%.0f fps, delta=%.1f fps)",
        best["fmt"], best["size"], best["fps"],
        target_fps, abs(best["fps"] - target_fps),
    )
    if len(ranked) > 1:
        runner_up = ranked[1]
        log.info(
            "     Runner-up:     fmt=%-8s  size=%-12s  fps=%.1f",
            runner_up["fmt"], runner_up["size"], runner_up["fps"],
        )
    return best


def probe_device_ffmpeg(device: str, chosen: "dict | None") -> "dict | None":
    """
    Probe device with ffmpeg -vframes 1.
    Handles EPROTO (device still resetting) with up to 4 retries.
    Returns the working params dict, or None if all attempts fail.
    """
    if chosen:
        probe_candidates = [chosen, None]   # None = ffmpeg auto-detect fallback
        extra = max(4, 3 * (1 + (1 // max(int(chosen["fps"]), 1))))
    else:
        probe_candidates = [
            {"fmt": "mjpeg",   "size": "", "fps": 0.0},
            {"fmt": "yuyv422", "size": "", "fps": 0.0},
            {"fmt": "",        "size": "", "fps": 0.0},
        ]
        extra = 4

    per_attempt = DEVICE_PROBE_TIMEOUT + extra
    PROBE_MAX_RETRIES = 4
    PROBE_RETRY_DELAY = 3

    log.info("Probing %s (per-attempt timeout %d s)", device, per_attempt)

    for attempt in range(1, PROBE_MAX_RETRIES + 1):
        eproto_count = 0
        for candidate in probe_candidates:
            if candidate is None:
                fmt, size, fps = "", "", ""
            else:
                fmt   = candidate.get("fmt",  "")
                size  = candidate.get("size", "")
                fps_f = candidate.get("fps",  0.0)
                fps   = str(int(round(fps_f))) if fps_f > 0 else ""

            cmd = ["ffmpeg", "-loglevel", "error", "-f", "v4l2"]
            if fmt:  cmd += ["-input_format", fmt]
            if fps:  cmd += ["-framerate", fps]
            if size: cmd += ["-video_size", size]
            cmd += ["-i", device, "-vframes", "1", "-f", "null", "-"]

            try:
                r = subprocess.run(cmd, timeout=per_attempt,
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    final_fmt  = fmt  or (chosen["fmt"]  if chosen else "")
                    final_size = size or (chosen["size"] if chosen else "")
                    final_fps  = fps  or (str(int(round(chosen["fps"])))
                                          if chosen and chosen["fps"] > 0 else "")
                    log.info("Device %s working (fmt=%s size=%s fps=%s)",
                             device, final_fmt or "auto",
                             final_size or "?", final_fps or "?")
                    return {"fmt": final_fmt, "size": final_size, "fps": final_fps}

                is_eproto = any("Protocol error" in l or "EPROTO" in l
                                for l in (r.stderr or "").splitlines())
                if is_eproto:
                    eproto_count += 1
                    log.debug("EPROTO on %s fmt=%s", device, fmt or "auto")
                else:
                    for line in (r.stderr or "").strip().splitlines():
                        log.warning("ffmpeg [%s fmt=%s]: %s",
                                    device, fmt or "auto", line)

            except subprocess.TimeoutExpired:
                log.warning("Probe timed out for %s fmt=%s (%d s)",
                            device, fmt or "auto", per_attempt)

        if eproto_count == len(probe_candidates):
            if attempt < PROBE_MAX_RETRIES:
                log.info("EPROTO on all formats (attempt %d/%d) — retrying in %d s",
                         attempt, PROBE_MAX_RETRIES, PROBE_RETRY_DELAY)
                time.sleep(PROBE_RETRY_DELAY)
            else:
                log.warning("Giving up on %s after %d EPROTO attempts",
                            device, PROBE_MAX_RETRIES)
        else:
            break

    log.info("No working format found for %s", device)
    return None


def find_working_camera() -> "tuple[str, dict] | tuple[None, None]":
    """Try each /dev/video* and return the first (device, params) that works."""
    devices = [CAMERA_DEVICE] if CAMERA_DEVICE else sorted(glob.glob("/dev/video*"))
    if not devices:
        log.warning("No /dev/video* devices found")
        return None, None

    for dev in devices:
        if not os.access(dev, os.R_OK):
            log.warning("%s not readable — add --device %s and --group-add video",
                        dev, dev)
            continue
        modes  = enumerate_camera_modes(dev)
        chosen = select_best_mode(modes, target_fps=float(CAM_TARGET_FPS))
        params = probe_device_ffmpeg(dev, chosen)
        if params is not None:
            log.info("Found working camera: %s", dev)
            return dev, params

    return None, None


# ── OpenCV capture loop ───────────────────────────────────────────────────────

def open_capture(device: str, params: dict) -> "cv2.VideoCapture | None":
    """Open the camera via OpenCV CAP_V4L2 using the probed parameters."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        log.warning("cv2 could not open %s with CAP_V4L2", device)
        return None

    # Pixel format
    fmt_str = CAM_FORMAT or _FFMT_TO_FOURCC.get(params.get("fmt", ""), "MJPG")
    fourcc  = cv2.VideoWriter_fourcc(*fmt_str[:4])
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # Resolution
    res = CAM_RESOLUTION or params.get("size", "")
    if res:
        try:
            w_s, h_s = res.lower().split("x")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(w_s))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h_s))
        except ValueError:
            log.warning("Cannot parse resolution '%s'", res)

    # FPS
    fps_str = CAM_FRAMERATE or params.get("fps", "")
    if fps_str:
        try:
            cap.set(cv2.CAP_PROP_FPS, float(fps_str))
        except ValueError:
            pass

    # Minimal internal buffer → always freshest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, _ = cap.read()
    if not ok:
        log.warning("First frame read failed on %s", device)
        cap.release()
        return None

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)
    log.info("Opened %s — %dx%d @ %.1f fps  fourcc=%s",
             device, aw, ah, af, fmt_str)
    return cap


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def stream_camera(writer: ShmFrameWriter, device: str, params: dict) -> None:
    """Camera capture loop. Returns when the camera disappears."""
    cap = open_capture(device, params)
    if cap is None:
        return

    # Determine the pacing interval from the actual FPS OpenCV negotiated.
    # CAP_PROP_FPS reflects what the driver committed to; if it returns 0
    # (some UVC drivers), fall back to CAM_TARGET_FPS.
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or float(CAM_TARGET_FPS)
    target_interval = 1.0 / actual_fps
    log.info("Capture pacing: %.1f fps target (interval %.3f s)", actual_fps, target_interval)

    frame_count  = 0
    t_report     = time.monotonic()

    while True:
        t0 = time.monotonic()

        ok, frame = cap.read()
        if not ok:
            log.warning("Frame read failed on %s", device)
            cap.release()
            return

        h, w = frame.shape[:2]

        if JPEG_QUALITY > 0:
            payload   = encode_jpeg(frame, JPEG_QUALITY)
            jpeg_size = len(payload)
        else:
            payload   = frame.tobytes()
            jpeg_size = 0

        writer.write_frame(payload, w, h, 3, jpeg_size)

        frame_count += 1
        now = time.monotonic()

        # Pace to the negotiated frame rate.  The driver already blocks inside
        # cap.read() for the exposure/sync time on most UVC cameras, so this
        # sleep is often 0.  For cameras that drain the kernel buffer faster
        # than the sensor produces frames (BUFFERSIZE=1 can still slip), the
        # sleep prevents spinning at 2× rate on cached frames.
        elapsed = now - t0
        slack   = target_interval - elapsed
        if slack > 0.001:
            time.sleep(slack)

        if now - t_report >= 5.0:
            achieved = frame_count / (now - t_report)
            if VERBOSE_STATS:
                log.info("%s  %.1f fps  %dx%d  payload=%d B  %s",
                         device, achieved, w, h, len(payload),
                         "JPEG" if jpeg_size else "BGR")
            if achieved < actual_fps * 0.8:
                log.warning(
                    "%s: achieved %.1f fps is well below target %.1f fps. "
                    "The BGR raw payload (%d B) may be too large for SHM throughput. "
                    "Consider VIEWER_JPEG_QUALITY=80 or a lower CAM_RESOLUTION.",
                    device, achieved, actual_fps, len(payload),
                )
            frame_count = 0
            t_report    = now


# ── Video file fallback (ported from original stream.py) ─────────────────────

def list_video_files() -> list:
    exts = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.ts", "*.flv", "*.webm")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(VID_DIR, ext)))
    return sorted(files)


def stream_videos_timed(writer: ShmFrameWriter, max_seconds: float = 0) -> None:
    """Loop video files from VID_DIR for up to max_seconds, then return.

    If max_seconds is 0 (or negative), run indefinitely (original behaviour).
    Exits the process if VID_DIR contains no video files at all.
    """
    files = list_video_files()
    if not files:
        log.error(
            "No video files found in '%s' and no working camera. "
            "Mount a directory with video files: -v /your/videos:%s:ro",
            VID_DIR, VID_DIR,
        )
        sys.exit(1)

    log.info("Video fallback: %d file(s) in %s", len(files), VID_DIR)

    deadline = (time.monotonic() + max_seconds) if max_seconds > 0 else None

    while True:
        if deadline and time.monotonic() >= deadline:
            return  # time to re-probe the camera

        files = list_video_files()
        if not files:
            log.warning("No video files in %s — retrying in 10 s", VID_DIR)
            time.sleep(10)
            continue

        for vf in files:
            if deadline and time.monotonic() >= deadline:
                return

            if VERBOSE_STATS:
                log.info("Playing: %s", vf)
            cap = cv2.VideoCapture(vf)
            if not cap.isOpened():
                log.warning("Cannot open %s", vf)
                continue

            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            interval   = 1.0 / native_fps

            while True:
                if deadline and time.monotonic() >= deadline:
                    cap.release()
                    return

                t0        = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    break   # EOF → next file

                h, w = frame.shape[:2]

                if JPEG_QUALITY > 0:
                    payload   = encode_jpeg(frame, JPEG_QUALITY)
                    jpeg_size = len(payload)
                else:
                    payload   = frame.tobytes()
                    jpeg_size = 0

                writer.write_frame(payload, w, h, 3, jpeg_size)

                elapsed = time.monotonic() - t0
                sleep   = interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)

            cap.release()
            time.sleep(0.5)

        if VERBOSE_STATS:
            log.info("Playlist finished — restarting from beginning")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("object-detection-custom-camera-capture starting (target_fps=%d)", CAM_TARGET_FPS)
    writer = ShmFrameWriter()
    log.info("SHM writer ready: %s", writer._path)

    consecutive_failures = 0

    while True:
        device, params = find_working_camera()
        if device:
            consecutive_failures = 0
            log.info("Using camera: %s", device)
            stream_camera(writer, device, params)
            # stream_camera() returned — camera was lost mid-stream
            log.warning("Camera lost on %s — re-probing in 2 s", device)
            time.sleep(2)
        else:
            consecutive_failures += 1
            # If a specific device was requested (CAMERA_DEVICE set) or we have
            # previously seen the device work, keep retrying so that a camera
            # reconnect or a brief driver hiccup is recovered automatically.
            # After the first failure we run the video fallback for
            # CAMERA_RETRY_INTERVAL seconds, then probe again.
            if consecutive_failures == 1:
                log.warning(
                    "No working camera found — starting video file fallback. "
                    "Will re-probe for a camera every %d s.",
                    CAMERA_RETRY_INTERVAL,
                )
            else:
                log.info(
                    "Camera still unavailable (attempt %d) — continuing video fallback.",
                    consecutive_failures,
                )

            # Run video fallback for CAMERA_RETRY_INTERVAL seconds, then try
            # the camera again.  stream_videos_timed() returns after the
            # interval (or exits if no video files are found at all).
            stream_videos_timed(writer, CAMERA_RETRY_INTERVAL)


if __name__ == "__main__":
    main()
