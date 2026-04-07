#!/usr/bin/env python3
"""
inference-tensorrt — Shared memory → TensorRT → ZeroMQ detections
==================================================================

Alternative to inference-onnx for NVIDIA Jetson or desktop GPU.
Uses the TensorRT Python bindings (TRT 10.x API) to run a TRT engine, or
builds one from an ONNX model on first run.

Publishes the same JSON schema as inference-onnx so the viewer and
downstream consumers are fully interchangeable.

GPU memory management uses the official `cuda-python` package (replaces the
legacy pycuda, which is incompatible with NumPy 2.x and TensorRT 10.x).

Supported YOLO output formats (auto-detected at startup)
---------------------------------------------------------
  classic   [1, 4+nc, anchors]  — YOLOv8, YOLO11 (no built-in NMS)
  end2end   [1, N, 6]           — YOLO12, YOLO26+ exported with nms=True
                                   columns: x1, y1, x2, y2, confidence, class_id

Environment variables
---------------------
SHM_FRAME_NAME      Shared memory name          (default: camera_frame)
ZMQ_PUB_PORT        ZeroMQ PUB port             (default: 5555)
ZMQ_TOPIC           ZeroMQ topic prefix         (default: detections)
INFERENCE_MODEL     .engine or .onnx file       (default: /opt/models/yolo11n.onnx)
MODELS_DIR          Model search dir            (default: /opt/models)
INFERENCE_WIDTH     Input width                 (default: 640)
INFERENCE_HEIGHT    Input height                (default: 640)
CONFIDENCE_THRESH   Min confidence              (default: 0.4)
NMS_THRESH          NMS IoU threshold           (default: 0.45)  [classic only]
TARGET_FPS          Max inference FPS           (default: 15)
TRT_FP16            Use FP16 precision          (default: 1)
CLASS_NAMES         Comma-separated names (optional, overrides ONNX metadata)
CLASS_NAMES_FILE    Path to names file (optional)
LOG_LEVEL           DEBUG / INFO / WARNING      (default: INFO)
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
SHM_NAME    = os.environ.get("SHM_FRAME_NAME",   "camera_frame")
ZMQ_PORT    = int(os.environ.get("ZMQ_PUB_PORT", "5555"))
ZMQ_TOPIC   = os.environ.get("ZMQ_TOPIC",         "detections")
MODELS_DIR  = os.environ.get("MODELS_DIR",         "/opt/models")
MODEL_NAME  = os.environ.get("INFERENCE_MODEL",   "yolo11n.onnx")
INF_W       = int(os.environ.get("INFERENCE_WIDTH",  "640"))
INF_H       = int(os.environ.get("INFERENCE_HEIGHT", "640"))
CONF_THRESH = float(os.environ.get("CONFIDENCE_THRESH", "0.4"))
NMS_THRESH  = float(os.environ.get("NMS_THRESH",        "0.45"))
TARGET_FPS  = float(os.environ.get("TARGET_FPS",        "15"))
TRT_FP16    = os.environ.get("TRT_FP16", "1") == "1"
LOG_LEVEL   = os.environ.get("LOG_LEVEL", "INFO").upper()
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

INTERVAL = 1.0 / max(TARGET_FPS, 0.1)


# ── Class names ───────────────────────────────────────────────────────────────

def load_class_names(onnx_path: str = "") -> dict:
    env = os.environ.get("CLASS_NAMES", "").strip()
    if env:
        names = [n.strip() for n in env.split(",") if n.strip()]
        log.info("Class names from CLASS_NAMES (%d)", len(names))
        return {i: n for i, n in enumerate(names)}

    fn = os.environ.get("CLASS_NAMES_FILE", "").strip()
    if fn and os.path.exists(fn):
        with open(fn) as f:
            names = [l.strip() for l in f if l.strip()]
        log.info("Class names from file (%d)", len(names))
        return {i: n for i, n in enumerate(names)}

    # Try ONNX metadata (only available before engine build)
    if onnx_path and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            meta = sess.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import ast
                names_raw = ast.literal_eval(meta["names"])
                log.info("Class names from ONNX metadata (%d)", len(names_raw))
                return names_raw if isinstance(names_raw, dict) else {i: n for i, n in enumerate(names_raw)}
        except Exception as e:
            log.debug("Could not read class names from ONNX metadata: %s", e)

    log.warning("No class names found — detections will use numeric IDs")
    return {}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(frame_bgr: np.ndarray, inf_w: int, inf_h: int):
    """Letterbox → float32 NCHW blob."""
    h, w  = frame_bgr.shape[:2]
    scale = min(inf_w / w, inf_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (inf_w - nw) // 2
    pad_y = (inf_h - nh) // 2
    canvas = np.full((inf_h, inf_w, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, pad_x, pad_y


# ── Post-processing ───────────────────────────────────────────────────────────

def _unletterbox(x1, y1, x2, y2, scale, pad_x, pad_y, orig_w, orig_h):
    x1 = max(0.0, min((x1 - pad_x) / scale, orig_w))
    y1 = max(0.0, min((y1 - pad_y) / scale, orig_h))
    x2 = max(0.0, min((x2 - pad_x) / scale, orig_w))
    y2 = max(0.0, min((y2 - pad_y) / scale, orig_h))
    return x1, y1, x2, y2


def postprocess_end2end(output, scale, pad_x, pad_y, orig_w, orig_h,
                         conf_thresh, class_names) -> list:
    """
    End2end YOLO output [1, N, 6]: x1,y1,x2,y2,conf,class_id.
    NMS is already baked in — skip it here.
    """
    rows = output[0]  # [N, 6]
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
            scale, pad_x, pad_y, orig_w, orig_h)
        detections.append({
            "class_id":   cls_id,
            "class_name": class_names.get(cls_id, str(cls_id)),
            "confidence": round(conf, 4),
            "bbox": {
                "x1": round(bx1, 1), "y1": round(by1, 1),
                "x2": round(bx2, 1), "y2": round(by2, 1),
                "cx": round((bx1+bx2)/2, 1), "cy": round((by1+by2)/2, 1),
                "w":  round(bx2-bx1, 1),     "h":  round(by2-by1, 1),
            },
        })
    return detections


def postprocess_classic(output, scale, pad_x, pad_y, orig_w, orig_h,
                         conf_thresh, nms_thresh, class_names) -> list:
    """Classic YOLO [1, 4+nc, anchors] — applies NMS internally."""
    preds = output[0]
    if preds.ndim == 1:
        return []
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes_xywh  = preds[:, :4]
    scores      = preds[:, 4:]
    class_ids   = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(preds)), class_ids]

    mask        = confidences >= conf_thresh
    boxes_xywh  = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]
    if len(boxes_xywh) == 0:
        return []

    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    indices = cv2.dnn.NMSBoxes(
        np.stack([x1,y1,x2,y2], 1).tolist(),
        confidences.tolist(), conf_thresh, nms_thresh)
    if len(indices) == 0:
        return []

    detections = []
    for i in indices.flatten():
        bx1, by1, bx2, by2 = _unletterbox(
            float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
            scale, pad_x, pad_y, orig_w, orig_h)
        cls = int(class_ids[i])
        detections.append({
            "class_id":   cls,
            "class_name": class_names.get(cls, str(cls)),
            "confidence": round(float(confidences[i]), 4),
            "bbox": {
                "x1": round(bx1,1), "y1": round(by1,1),
                "x2": round(bx2,1), "y2": round(by2,1),
                "cx": round((bx1+bx2)/2,1), "cy": round((by1+by2)/2,1),
                "w":  round(bx2-bx1,1),     "h":  round(by2-by1,1),
            },
        })
    return detections


def postprocess(output, output_format, scale, pad_x, pad_y, orig_w, orig_h,
                conf_thresh, nms_thresh, class_names) -> list:
    if output_format == "end2end":
        return postprocess_end2end(
            output, scale, pad_x, pad_y, orig_w, orig_h,
            conf_thresh, class_names)
    return postprocess_classic(
        output, scale, pad_x, pad_y, orig_w, orig_h,
        conf_thresh, nms_thresh, class_names)


# ── TensorRT engine (TRT 10.x + cuda-python) ─────────────────────────────────

class TRTEngine:
    """
    TensorRT 10.x inference engine.

    Key API changes vs TRT 8/9:
      - num_bindings / get_binding_* / execute_async_v2  →  REMOVED
      - Use: num_io_tensors / get_tensor_* / set_tensor_address / execute_async_v3

    GPU memory via cuda-python (nvidia-maintained pycuda replacement,
    compatible with NumPy 2.x).
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt
        from cuda import cudart

        self._trt    = trt
        self._cudart = cudart

        # ── Load serialised engine ──────────────────────────────────────────
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()

        # ── Allocate device buffers and register tensor addresses ───────────
        self._inputs  = []
        self._outputs = []

        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

            # Allocate contiguous host buffer (page-locked would be faster
            # but requires extra setup — regular numpy is fine for <30 FPS)
            host_buf = np.empty(shape, dtype=dtype)

            # Allocate device memory
            err, dev_ptr = cudart.cudaMalloc(nbytes)
            self._check(err, "cudaMalloc")

            # Tell TRT where each tensor lives on the device
            self._context.set_tensor_address(name, dev_ptr)

            info = {
                "name":   name,
                "shape":  shape,
                "dtype":  dtype,
                "host":   host_buf,
                "dev":    dev_ptr,
                "nbytes": nbytes,
            }
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append(info)
            else:
                self._outputs.append(info)

        # ── Create a persistent CUDA stream ─────────────────────────────────
        err, self._stream = cudart.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")

        # ── Detect output format ─────────────────────────────────────────────
        self.output_format = self._detect_output_format()

        log.info(
            "TRT engine loaded: %s  inputs=%d  outputs=%d  fmt=%s",
            engine_path, len(self._inputs), len(self._outputs), self.output_format,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _check(self, err, op: str):
        from cuda import cudart
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error in {op}: {err}")

    def _detect_output_format(self) -> str:
        """
        Auto-detect YOLO output format from engine output shape.
          end2end  → [1, N, 6]  (NMS baked in: x1,y1,x2,y2,conf,cls)
          classic  → [1, 4+nc, anchors] or [1, anchors, 4+nc]
        """
        if not self._outputs:
            return "classic"
        shape = self._outputs[0]["shape"]
        if len(shape) == 3 and int(shape[2]) == 6:
            log.info(
                "Detected output format: end2end  shape=[1,%s,6]  "
                "(NMS baked in — x1,y1,x2,y2,conf,class_id)", shape[1])
            return "end2end"
        log.info(
            "Detected output format: classic  shape=%s  "
            "(YOLOv8/v11 style — NMS applied in post-processing)", list(shape))
        return "classic"

    # ── Inference ────────────────────────────────────────────────────────────

    def infer(self, blob: np.ndarray) -> list:
        """
        Copy blob → GPU, run inference, copy results back → CPU.
        Returns a list of numpy arrays (one per output tensor).
        """
        cr = self._cudart

        # Copy input blob to device
        inp = self._inputs[0]
        np.copyto(inp["host"], blob.reshape(inp["shape"]))
        cr.cudaMemcpyAsync(
            inp["dev"],
            inp["host"].ctypes.data,
            inp["nbytes"],
            cr.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self._stream,
        )

        # Execute
        self._context.execute_async_v3(self._stream)

        # Copy all outputs back to host
        for out in self._outputs:
            cr.cudaMemcpyAsync(
                out["host"].ctypes.data,
                out["dev"],
                out["nbytes"],
                cr.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            )

        # Synchronise before reading host buffers
        cr.cudaStreamSynchronize(self._stream)

        return [out["host"].copy() for out in self._outputs]

    def __del__(self):
        try:
            cr = self._cudart
            cr.cudaStreamDestroy(self._stream)
            for buf in self._inputs + self._outputs:
                cr.cudaFree(buf["dev"])
        except Exception:
            pass


# ── Engine builder ────────────────────────────────────────────────────────────

def build_engine_from_onnx(onnx_path: str) -> str:
    """Build (or return cached) TRT engine from an ONNX file."""
    import tensorrt as trt

    engine_path = os.path.splitext(onnx_path)[0] + ".engine"
    if os.path.exists(engine_path):
        log.info("Using cached TRT engine: %s", engine_path)
        return engine_path

    log.info("Building TRT engine from %s (this may take several minutes) …", onnx_path)
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("TRT ONNX parser error: %s", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if TRT_FP16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        log.info("TRT FP16 enabled")

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TRT engine build failed — see warnings above")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    log.info("TRT engine saved to %s", engine_path)
    return engine_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import tensorrt as trt
    log.info("inference-tensorrt starting")
    log.info("TensorRT version: %s", trt.__version__)

    # ── Resolve model paths ──────────────────────────────────────────────────
    stem        = os.path.splitext(MODEL_NAME)[0]
    engine_path = os.path.join(MODELS_DIR, stem + ".engine")
    onnx_path   = os.path.join(MODELS_DIR, stem + ".onnx")

    # Fallback: bare path passed directly in INFERENCE_MODEL
    if not os.path.exists(engine_path) and not os.path.exists(onnx_path):
        if os.path.exists(MODEL_NAME):
            onnx_path = MODEL_NAME
            engine_path = os.path.splitext(MODEL_NAME)[0] + ".engine"

    # Load class names from ONNX before building the engine
    # (metadata is still accessible from the ONNX file even after engine build)
    class_names = load_class_names(onnx_path)

    if not os.path.exists(engine_path):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"Neither engine nor ONNX model found.\n"
                f"  engine: {engine_path}\n"
                f"  onnx:   {onnx_path}"
            )
        engine_path = build_engine_from_onnx(onnx_path)

    engine = TRTEngine(engine_path)

    # ── ZeroMQ PUB ───────────────────────────────────────────────────────────
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{ZMQ_PORT}")
    log.info("ZeroMQ PUB bound on tcp://*:%d  topic='%s'", ZMQ_PORT, ZMQ_TOPIC)

    # ── SHM reader ───────────────────────────────────────────────────────────
    log.info("Waiting for SHM: %s", SHM_NAME)
    reader = ShmFrameReader(name=SHM_NAME, timeout=30.0)
    log.info("SHM reader ready")

    model_basename = os.path.basename(engine_path)
    frame_count = det_count = 0
    t_report    = time.monotonic()

    while True:
        t0 = time.monotonic()

        seq, w, h, ch, ts_capture, data, jpeg_size = reader.read_frame(wait_new=True)

        if jpeg_size:
            arr   = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                log.warning("JPEG decode failed for seq=%d", seq)
                continue
        else:
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, ch)

        blob, scale, pad_x, pad_y = preprocess(frame, INF_W, INF_H)
        outputs  = engine.infer(blob)
        ts_infer = time.monotonic()

        detections = postprocess(
            outputs[0], engine.output_format,
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
        pub.send_string(f"{ZMQ_TOPIC} {json.dumps(payload)}")

        frame_count += 1
        det_count   += len(detections)

        elapsed = time.monotonic() - t0
        sleep   = INTERVAL - elapsed
        if sleep > 0:
            time.sleep(sleep)

        now = time.monotonic()
        if now - t_report >= 5.0:
            dt = now - t_report
            log.info(
                "fps=%.1f  dets/frame=%.1f  latency≈%.1f ms  [TRT %s]",
                frame_count / dt, det_count / max(frame_count, 1),
                latency_ms, engine.output_format,
            )
            frame_count = det_count = 0
            t_report    = now


if __name__ == "__main__":
    main()