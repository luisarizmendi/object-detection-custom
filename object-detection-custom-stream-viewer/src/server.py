#!/usr/bin/env python3
"""
stream-viewer backend — SHM frames + ZMQ detections → browser via WebSocket
============================================================================

Single WebSocket connection per client.  The server pushes two message types:

  Binary message:   raw JPEG bytes  (the browser draws this on <canvas>)
  Text  message:    JSON string with detection overlay data

This avoids any RTSP/WebRTC complexity and gives sub-100 ms end-to-end
latency on a local device.

Environment variables
---------------------
SHM_FRAME_NAME      Shared memory name  (default: camera_frame)
ZMQ_SUB_HOST        ZeroMQ PUB host     (default: localhost)
ZMQ_SUB_PORT        ZeroMQ PUB port     (default: 5555)
ZMQ_TOPIC           ZeroMQ topic prefix (default: detections)
HTTP_PORT           HTTP/WS server port  (default: 8080)
FRAME_JPEG_QUALITY  JPEG re-encode quality 1-95, 0=passthrough if already JPEG
                    (default: 75)
MAX_STREAM_FPS      Max FPS pushed to each client (default: 35)
LOG_LEVEL           DEBUG / INFO / WARNING (default: INFO)
"""

import asyncio
import json
import logging
import os
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
import zmq.asyncio
from aiohttp import web
import aiohttp

# ── Config ────────────────────────────────────────────────────────────────────
SHM_NAME       = os.environ.get("SHM_FRAME_NAME",   "camera_frame")
ZMQ_HOST       = os.environ.get("ZMQ_SUB_HOST",     "localhost")
ZMQ_PORT       = int(os.environ.get("ZMQ_SUB_PORT", "5555"))
ZMQ_TOPIC      = os.environ.get("ZMQ_TOPIC",         "detections")
HTTP_PORT      = int(os.environ.get("HTTP_PORT",    "8080"))
JPEG_QUALITY   = int(os.environ.get("FRAME_JPEG_QUALITY", "75"))
MAX_FPS        = float(os.environ.get("MAX_STREAM_FPS", "35"))
LOG_LEVEL      = os.environ.get("LOG_LEVEL", "INFO").upper()
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

FRAME_INTERVAL = 1.0 / MAX_FPS


# ── Shared state ──────────────────────────────────────────────────────────────
# Threads write; asyncio tasks read via asyncio.Queue per client

class SharedState:
    def __init__(self):
        self._lock        = threading.Lock()
        self._frame_jpeg  = b""          # latest JPEG bytes
        self._frame_seq   = 0
        self._detections  = "{}"         # latest detection JSON string
        self._det_seq     = 0
        self._frame_evt   = asyncio.Event()   # set by frame thread, cleared by readers
        self._det_evt     = asyncio.Event()

        self._loop: asyncio.AbstractEventLoop = None

    def set_loop(self, loop):
        self._loop = loop

    def update_frame(self, jpeg_bytes: bytes, seq: int):
        with self._lock:
            self._frame_jpeg = jpeg_bytes
            self._frame_seq  = seq
        if self._loop:
            self._loop.call_soon_threadsafe(self._frame_evt.set)

    def update_detections(self, json_str: str, seq: int):
        with self._lock:
            self._detections = json_str
            self._det_seq    = seq
        if self._loop:
            self._loop.call_soon_threadsafe(self._det_evt.set)

    def get_frame(self):
        with self._lock:
            return self._frame_jpeg, self._frame_seq

    def get_detections(self):
        with self._lock:
            return self._detections, self._det_seq


STATE = SharedState()


# ── Frame reader thread (SHM) ─────────────────────────────────────────────────

def frame_reader_thread():
    from shm_frame import ShmFrameReader

    log.info("Frame reader: waiting for SHM '%s'", SHM_NAME)
    while True:
        try:
            reader = ShmFrameReader(name=SHM_NAME, timeout=60.0)
            break
        except TimeoutError:
            log.warning("SHM not available yet — retrying")

    log.info("Frame reader: SHM ready")
    last_seq    = 0
    t_next_push = 0.0

    while True:
        try:
            seq, w, h, ch, ts, data, jpeg_size = reader.read_frame(wait_new=True)

            now = time.monotonic()
            if now < t_next_push:
                continue   # skip frame to cap push rate
            t_next_push = now + FRAME_INTERVAL

            if jpeg_size and JPEG_QUALITY == 0:
                # pass-through: SHM already stored JPEG
                jpeg = data
            else:
                if jpeg_size:
                    arr   = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                else:
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, ch)
                if frame is None:
                    continue
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY or 75])
                if not ok:
                    continue
                jpeg = buf.tobytes()

            STATE.update_frame(jpeg, seq)

        except Exception as e:
            log.error("Frame reader error: %s", e)
            time.sleep(1)


# ── Detection reader thread (ZeroMQ) ─────────────────────────────────────────

def detection_reader_thread():
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, ZMQ_TOPIC)
    sub.setsockopt(zmq.RCVTIMEO, 500)   # ms
    addr = f"tcp://{ZMQ_HOST}:{ZMQ_PORT}"
    sub.connect(addr)
    log.info("Detection reader: connected to %s", addr)

    while True:
        try:
            msg = sub.recv_string()
            # Strip topic prefix
            _, json_str = msg.split(" ", 1)
            payload = json.loads(json_str)
            STATE.update_detections(json_str, payload.get("seq", 0))
        except zmq.Again:
            pass   # timeout — no message
        except Exception as e:
            log.error("Detection reader error: %s", e)
            time.sleep(0.1)


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    log.info("WS client connected: %s", request.remote)

    last_frame_seq = -1
    last_det_seq   = -1

    try:
        while not ws.closed:
            # Wait for either a new frame or new detections
            await asyncio.sleep(0)   # yield

            frame_jpeg, frame_seq = STATE.get_frame()
            det_json,   det_seq   = STATE.get_detections()

            sent_something = False

            if frame_seq != last_frame_seq and frame_jpeg:
                await ws.send_bytes(frame_jpeg)
                last_frame_seq = frame_seq
                sent_something = True

            if det_seq != last_det_seq and det_json:
                await ws.send_str(det_json)
                last_det_seq = det_seq
                sent_something = True

            if not sent_something:
                # Nothing new — wait briefly for events
                try:
                    done, _ = await asyncio.wait(
                        [
                            asyncio.create_task(STATE._frame_evt.wait()),
                            asyncio.create_task(STATE._det_evt.wait()),
                        ],
                        timeout=0.1,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    STATE._frame_evt.clear()
                    STATE._det_evt.clear()
                except Exception:
                    await asyncio.sleep(0.02)

    except ConnectionResetError:
        pass
    finally:
        log.info("WS client disconnected: %s", request.remote)

    return ws


# ── HTTP static file handler ──────────────────────────────────────────────────

async def index_handler(request):
    static_dir = Path(__file__).parent / "static"
    index      = static_dir / "index.html"
    if index.exists():
        return web.FileResponse(index)
    return web.Response(text="stream-viewer: index.html not found", status=404)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async():
    STATE.set_loop(asyncio.get_event_loop())

    # Start background threads
    for fn in (frame_reader_thread, detection_reader_thread):
        t = threading.Thread(target=fn, daemon=True)
        t.start()

    app = web.Application()
    app.router.add_get("/ws",    ws_handler)
    app.router.add_get("/",      index_handler)
    app.router.add_static("/static",
                          Path(__file__).parent / "static",
                          show_index=False)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
    await site.start()
    log.info("stream-viewer listening on http://0.0.0.0:%d", HTTP_PORT)

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main_async())
