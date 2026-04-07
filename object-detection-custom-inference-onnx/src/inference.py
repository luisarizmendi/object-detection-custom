#!/usr/bin/env python3
"""
object-detection-custom-inference-onnx — Shared-memory frame → ONNX Runtime → ZeroMQ detections
=========================================================================

Reads raw frames from shared memory written by object-detection-custom-camera-capture, runs YOLO
inference via ONNX Runtime (CPU or CUDA EP), and publishes detection results
as JSON over ZeroMQ PUB.

Supported YOLO output formats (auto-detected at startup)
---------------------------------------------------------
  classic   [1, 4+nc, anchors]  — YOLOv8, YOLO11 (no built-in NMS)
  end2end   [1, N, 6]           — YOLO12, YOLO26+ exported with nms=True
                                   columns: x1, y1, x2, y2, confidence, class_id

JSON schema per frame
---------------------
{
  "seq":        1234,
  "ts_capture": 12345.678,
  "ts_infer":   12345.690,
  "latency_ms": 12.3,
  "frame_w":    640,
  "frame_h":    480,
  "model":      "yolo26n.onnx",
  "detections": [
    {
      "class_id":   0,
      "class_name": "person",
      "confidence": 0.87,
      "bbox": {
        "x1": 100, "y1": 50,
        "x2": 300, "y2": 400,
        "cx": 200, "cy": 225,
        "w":  200, "h":  350
      }
    }
  ]
}

Environment variables
---------------------
SHM_FRAME_NAME      Shared memory name (default: camera_frame)
ZMQ_PUB_PORT        ZeroMQ PUB port    (default: 5555)
ZMQ_TOPIC           ZeroMQ topic prefix (default: "detections")
INFERENCE_MODEL     ONNX model file    (default: /opt/models/yolo11n.onnx)
MODELS_DIR          Model search dir   (default: /opt/models)
INFERENCE_WIDTH     Input width        (default: 640)
INFERENCE_HEIGHT    Input height       (default: 640)
CONFIDENCE_THRESH   Min confidence     (default: 0.4)
NMS_THRESH          NMS IoU threshold  (default: 0.45)  [classic format only]
TARGET_FPS          Max inference FPS  (default: 15)
EXECUTION_PROVIDER  onnxruntime EP: CPUExecutionProvider |
                    CUDAExecutionProvider |
                    TensorrtExecutionProvider |
                    auto  (default: auto)
CLASS_NAMES         Comma-separated names (optional, overrides model metadata)
CLASS_NAMES_FILE    Path to names file, one per line (optional)
LOG_LEVEL           DEBUG / INFO / WARNING (default: INFO)
"""

import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import zmq

from shm_frame import ShmFrameReader

# ── Config ────────────────────────────────────────────────────────────────────
SHM_NAME    = os.environ.get("SHM_FRAME_NAME",    "camera_frame")
ZMQ_PORT    = int(os.environ.get("ZMQ_PUB_PORT",  "5555"))
ZMQ_TOPIC   = os.environ.get("ZMQ_TOPIC",          "detections")
MODELS_DIR  = os.environ.get("MODELS_DIR",          "/opt/models")
MODEL_NAME  = os.environ.get("INFERENCE_MODEL",    "yolo11n.onnx")
INF_W       = int(os.environ.get("INFERENCE_WIDTH",  "640"))
INF_H       = int(os.environ.get("INFERENCE_HEIGHT", "640"))
CONF_THRESH = float(os.environ.get("CONFIDENCE_THRESH", "0.4"))
NMS_THRESH  = float(os.environ.get("NMS_THRESH",        "0.45"))
TARGET_FPS  = float(os.environ.get("TARGET_FPS",        "15"))
EP          = os.environ.get("EXECUTION_PROVIDER",  "auto")
LOG_LEVEL   = os.environ.get("LOG_LEVEL", "INFO").upper()
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

INTERVAL = 1.0 / max(TARGET_FPS, 0.1)


# ── ONNX helpers ──────────────────────────────────────────────────────────────

def resolve_model_path() -> str:
    stem = os.path.splitext(MODEL_NAME)[0]
    for ext in (".onnx",):
        p = os.path.join(MODELS_DIR, stem + ext)
        if os.path.exists(p):
            return p
    if os.path.exists(MODEL_NAME):
        return MODEL_NAME
    raise FileNotFoundError(
        f"Model not found: tried {MODELS_DIR}/{stem}.onnx and {MODEL_NAME}"
    )


def resolve_ep(model_path: str) -> list:
    import onnxruntime as ort
    available = ort.get_available_providers()
    log.info("Available ONNX Runtime providers: %s", available)

    if "CUDAExecutionProvider" not in available:
        log.warning(
            "CUDAExecutionProvider is NOT available. "
            "Ensure you installed 'onnxruntime-gpu' (not 'onnxruntime') and "
            "the container has GPU access (--gpus all / devices: nvidia.com/gpu=all). "
            "Falling back to CPU."
        )

    if EP != "auto":
        if EP not in available:
            log.warning("Requested EP '%s' not available — falling back to CPU", EP)
            return ["CPUExecutionProvider"]
        return [EP, "CPUExecutionProvider"]

    # Auto: prefer CUDA, fall back to CPU
    for ep in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if ep in available:
            log.info("Auto-selected EP: %s", ep)
            return [ep, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_class_names(model_path: str) -> dict:
    env_names = os.environ.get("CLASS_NAMES", "").strip()
    env_file  = os.environ.get("CLASS_NAMES_FILE", "").strip()

    if env_names:
        names = [n.strip() for n in env_names.split(",") if n.strip()]
        log.info("Class names from CLASS_NAMES (%d)", len(names))
        return {i: n for i, n in enumerate(names)}

    if env_file and os.path.exists(env_file):
        with open(env_file) as f:
            names = [l.strip() for l in f if l.strip()]
        log.info("Class names from file (%d)", len(names))
        return {i: n for i, n in enumerate(names)}

    # Try ONNX model metadata
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        meta = sess.get_modelmeta().custom_metadata_map
        if "names" in meta:
            import ast
            names_raw = ast.literal_eval(meta["names"])
            log.info("Class names from model metadata (%d)", len(names_raw))
            return names_raw if isinstance(names_raw, dict) else {i: n for i, n in enumerate(names_raw)}
    except Exception as e:
        log.debug("Could not read class names from model metadata: %s", e)

    log.warning("No class names found — detections will use numeric IDs")
    return {}


# ── Output format detection ───────────────────────────────────────────────────

def detect_output_format(sess) -> str:
    """
    Auto-detect YOLO output format from the session output tensor shape.

    Returns:
      "end2end"  — output [1, N, 6]: x1,y1,x2,y2,conf,class_id  (NMS baked in)
                   Used by YOLO12, YOLO26, and any model exported with nms=True
      "classic"  — output [1, 4+nc, anchors] or [1, anchors, 4+nc]
                   Used by YOLOv8, YOLO11
    """
    out_shape = sess.get_outputs()[0].shape  # may contain strings for dynamic dims

    def to_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    if len(out_shape) == 3:
        d2 = to_int(out_shape[2])
        if d2 == 6:
            log.info(
                "Detected output format: end2end  shape=[1,%s,6]  "
                "(NMS baked in — x1,y1,x2,y2,conf,class_id)",
                out_shape[1],
            )
            return "end2end"

    log.info(
        "Detected output format: classic  shape=%s  "
        "(YOLOv8/v11 style — NMS applied in post-processing)",
        list(out_shape),
    )
    return "classic"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(frame_bgr: np.ndarray, inf_w: int, inf_h: int):
    """Returns (blob, scale, pad_x, pad_y) using letterbox."""
    h, w = frame_bgr.shape[:2]
    scale = min(inf_w / w, inf_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_x = (inf_w - nw) // 2
    pad_y = (inf_h - nh) // 2
    canvas = np.full((inf_h, inf_w, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, /255
    blob = blob.transpose(2, 0, 1)[np.newaxis]             # HWC→NCHW
    return blob, scale, pad_x, pad_y


# ── Post-processing ───────────────────────────────────────────────────────────

def _unletterbox(x1, y1, x2, y2, scale, pad_x, pad_y, orig_w, orig_h):
    """Remove letterbox padding and scale back to original frame coordinates."""
    x1 = max(0.0, min((x1 - pad_x) / scale, orig_w))
    y1 = max(0.0, min((y1 - pad_y) / scale, orig_h))
    x2 = max(0.0, min((x2 - pad_x) / scale, orig_w))
    y2 = max(0.0, min((y2 - pad_y) / scale, orig_h))
    return x1, y1, x2, y2


def postprocess_end2end(
    output: np.ndarray,
    scale: float, pad_x: int, pad_y: int,
    orig_w: int, orig_h: int,
    conf_thresh: float,
    class_names: dict,
) -> list:
    """
    Parse end2end YOLO output tensor shaped [1, N, 6].

    Columns: x1, y1, x2, y2, confidence, class_id
    Boxes are already in letterboxed pixel coordinates.
    NMS has already been applied by the model — skip it here.
    """
    rows = output[0]   # [N, 6]
    if rows.ndim != 2 or rows.shape[1] != 6:
        log.warning("Unexpected end2end output shape: %s", output.shape)
        return []

    detections = []
    for row in rows:
        x1, y1, x2, y2, conf, cls_raw = row
        conf = float(conf)
        if conf < conf_thresh:
            continue

        cls_id = int(round(float(cls_raw)))
        bx1, by1, bx2, by2 = _unletterbox(
            float(x1), float(y1), float(x2), float(y2),
            scale, pad_x, pad_y, orig_w, orig_h,
        )

        detections.append({
            "class_id":   cls_id,
            "class_name": class_names.get(cls_id, str(cls_id)),
            "confidence": round(conf, 4),
            "bbox": {
                "x1": round(bx1, 1), "y1": round(by1, 1),
                "x2": round(bx2, 1), "y2": round(by2, 1),
                "cx": round((bx1 + bx2) / 2, 1),
                "cy": round((by1 + by2) / 2, 1),
                "w":  round(bx2 - bx1, 1),
                "h":  round(by2 - by1, 1),
            },
        })
    return detections


def postprocess_classic(
    output: np.ndarray,
    scale: float, pad_x: int, pad_y: int,
    orig_w: int, orig_h: int,
    conf_thresh: float, nms_thresh: float,
    class_names: dict,
) -> list:
    """
    Parse classic YOLOv8/v11 output tensor [1, 4+nc, anchors].
    Applies NMS internally.
    """
    preds = output[0]  # [4+nc, anchors]

    if preds.ndim == 1:
        return []

    # Always work in [anchors, 4+nc] layout
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    num_anchors, row_size = preds.shape
    if row_size < 5:
        log.warning("Unexpected classic output row size: %d", row_size)
        return []

    boxes_xywh = preds[:, :4]
    scores     = preds[:, 4:]

    class_ids   = np.argmax(scores, axis=1)
    confidences = scores[np.arange(num_anchors), class_ids]

    mask        = confidences >= conf_thresh
    boxes_xywh  = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # cx,cy,w,h → x1,y1,x2,y2
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    boxes_nms = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    indices   = cv2.dnn.NMSBoxes(
        boxes_nms.tolist(), confidences.tolist(), conf_thresh, nms_thresh
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten()

    detections = []
    for i in indices:
        bx1, by1, bx2, by2 = _unletterbox(
            float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
            scale, pad_x, pad_y, orig_w, orig_h,
        )
        cls_id = int(class_ids[i])
        detections.append({
            "class_id":   cls_id,
            "class_name": class_names.get(cls_id, str(cls_id)),
            "confidence": round(float(confidences[i]), 4),
            "bbox": {
                "x1": round(bx1, 1), "y1": round(by1, 1),
                "x2": round(bx2, 1), "y2": round(by2, 1),
                "cx": round((bx1 + bx2) / 2, 1),
                "cy": round((by1 + by2) / 2, 1),
                "w":  round(bx2 - bx1, 1),
                "h":  round(by2 - by1, 1),
            },
        })
    return detections


def postprocess(
    output: np.ndarray,
    output_format: str,
    scale: float, pad_x: int, pad_y: int,
    orig_w: int, orig_h: int,
    conf_thresh: float, nms_thresh: float,
    class_names: dict,
) -> list:
    """Dispatch to the correct post-processor based on detected output format."""
    if output_format == "end2end":
        return postprocess_end2end(
            output, scale, pad_x, pad_y, orig_w, orig_h,
            conf_thresh, class_names,
        )
    return postprocess_classic(
        output, scale, pad_x, pad_y, orig_w, orig_h,
        conf_thresh, nms_thresh, class_names,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import onnxruntime as ort

    log.info("object-detection-custom-inference-onnx starting")
    log.info("onnxruntime version: %s", ort.__version__)

    model_path    = resolve_model_path()
    providers     = resolve_ep(model_path)
    class_names   = load_class_names(model_path)

    log.info("Loading model: %s", model_path)
    sess          = ort.InferenceSession(model_path, providers=providers)
    input_name    = sess.get_inputs()[0].name
    output_name   = sess.get_outputs()[0].name
    log.info("Model loaded. Input: %s  Output: %s", input_name, output_name)
    log.info("Providers in use: %s", sess.get_providers())

    # Auto-detect output tensor format once at startup
    output_format = detect_output_format(sess)

    # ZeroMQ PUB socket
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{ZMQ_PORT}")
    log.info("ZeroMQ PUB bound on tcp://*:%d  topic='%s'", ZMQ_PORT, ZMQ_TOPIC)

    # SHM reader
    log.info("Waiting for SHM frame slot: %s", SHM_NAME)
    reader = ShmFrameReader(name=SHM_NAME, timeout=30.0)
    log.info("SHM reader ready")

    model_basename = os.path.basename(model_path)
    frame_count = det_count = 0
    t_report    = time.monotonic()

    while True:
        t0 = time.monotonic()

        seq, w, h, ch, ts_capture, data, jpeg_size = reader.read_frame(wait_new=True)

        # Decode frame
        if jpeg_size:
            arr   = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                log.warning("JPEG decode failed for seq=%d", seq)
                continue
        else:
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, ch)

        # Preprocess
        blob, scale, pad_x, pad_y = preprocess(frame, INF_W, INF_H)

        # Inference
        outputs  = sess.run([output_name], {input_name: blob})
        ts_infer = time.monotonic()

        # Postprocess — format-aware
        detections = postprocess(
            outputs[0], output_format,
            scale, pad_x, pad_y, w, h,
            CONF_THRESH, NMS_THRESH, class_names,
        )

        latency_ms = (ts_infer - ts_capture) * 1000.0

        payload = {
            "seq":        seq,
            "ts_capture": round(ts_capture, 6),
            "ts_infer":   round(ts_infer,   6),
            "latency_ms": round(latency_ms, 2),
            "frame_w":    w,
            "frame_h":    h,
            "model":      model_basename,
            "detections": detections,
        }

        msg = f"{ZMQ_TOPIC} {json.dumps(payload)}"
        pub.send_string(msg)

        frame_count += 1
        det_count   += len(detections)

        # Throttle to TARGET_FPS
        elapsed = time.monotonic() - t0
        sleep   = INTERVAL - elapsed
        if sleep > 0:
            time.sleep(sleep)

        # Periodic stats
        now = time.monotonic()
        if now - t_report >= 5.0:
            dt  = now - t_report
            fps = frame_count / dt
            log.info(
                "fps=%.1f  dets/frame=%.1f  latency≈%.1f ms  ep=%s  fmt=%s",
                fps, det_count / max(frame_count, 1),
                latency_ms, providers[0], output_format,
            )
            frame_count = det_count = 0
            t_report    = now


if __name__ == "__main__":
    main()