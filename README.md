# object-detection-custom

Real-time object detection from a USB camera or video file, designed for edge
devices (x86_64, Jetson Nano/Orin) with minimum latency and minimum dependencies.

## Architecture

```
USB Camera (/dev/videoN)  ─or─  Video file fallback (/videos/)
    │  v4l2 direct (OpenCV CAP_V4L2)
    ▼
object-detection-custom-camera-capture
    │  POSIX shared memory /dev/shm/camera_frame
    │  (header: seq + timestamp + dims; payload: raw BGR or JPEG)
    ▼
[ pick ONE inference service — see "Choosing an inference backend" below ]

object-detection-custom-inference-onnx            object-detection-custom-inference-tensorrt        object-detection-custom-inference-tensorrt-jetson
ONNX Runtime              TensorRT engine           TensorRT engine
CPU / CUDA EP auto        x86_64 / generic arm64    Jetson (l4t, aarch64 only)
    │                         │                         │
    └─────────────────────────┴─────────────────────────┘
                              │  ZeroMQ PUB tcp://5555
                              ▼
                        object-detection-custom-stream-viewer
                              │  WebSocket :8080
                           Browser
```

> **Run only one inference service at a time.**
> All three publish on the same ZeroMQ port (5555). Running two simultaneously
> causes port conflicts and undefined detection output.


## Choosing an inference backend

| Situation | Use |
|---|---|
| Any machine, no GPU, or just getting started | `object-detection-custom-inference-onnx` (CPU fallback automatic) |
| Machine with NVIDIA GPU (x86_64 or generic arm64) | `object-detection-custom-inference-onnx` with CUDA EP, or `object-detection-custom-inference-tensorrt` for max throughput |
| NVIDIA Jetson (JetPack 6.x, aarch64) | `object-detection-custom-inference-tensorrt-jetson` — uses the Tegra-optimised L4T base |
| Prototyping / model switching | `object-detection-custom-inference-onnx` — no engine build step, hot-swap `.onnx` files |
| Production edge, latency critical, fixed model | `object-detection-custom-inference-tensorrt` or `object-detection-custom-inference-tensorrt-jetson` |

TensorRT gives the lowest latency on NVIDIA hardware but requires a one-time
engine build (~5 min on Jetson, ~2 min on desktop GPU) per model. ONNX is
more portable and starts instantly.


## Services

### object-detection-custom-camera-capture

Enumerates all `/dev/video*` devices and queries the full list of supported
formats, resolutions, and frame rates using `v4l2-ctl`. The mode table is
printed at startup so you can confirm exactly what was selected.

**Video file fallback:** if no usable camera is detected (or a previously
working device fails), object-detection-custom-camera-capture automatically falls back to looping
video files from `VID_DIR` (`/videos` by default). It then re-probes for a
live camera every `CAMERA_RETRY_INTERVAL` seconds (default 30 s) so that
plugging in a camera mid-run is detected automatically. No restart needed.

The shared memory slot layout:

```
[0:4]   uint32  sequence number
[4:8]   uint32  width
[8:12]  uint32  height
[12:16] uint32  channels  (3 = BGR)
[16:24] float64 capture timestamp (monotonic)
[24:28] uint32  jpeg_size (0 = raw BGR)
[28:]   bytes   pixel data
```

Default SHM buffer: **4K** (`4096×2160×3 ≈ 26 MB`). Override with
`SHM_MAX_FRAME_BYTES` if needed (must be consistent across all services).

### object-detection-custom-inference-onnx
- ONNX Runtime with automatic EP selection: TensorRT EP → CUDA EP → CPU
- No engine build step — swap `.onnx` files freely
- GPU-first, CPU fallback with no configuration change

### object-detection-custom-inference-tensorrt  _(desktop/server GPU — x86_64 or generic arm64)_
- Base image: `nvcr.io/nvidia/tensorrt:24.05-py3`
- Builds a `.engine` on first run (~2–5 min, cached in `/opt/models`)
- FP16 enabled by default
- **Do not use on Jetson** — use `object-detection-custom-inference-tensorrt-jetson` instead

### object-detection-custom-inference-tensorrt-jetson  _(Jetson, JetPack 6.x, aarch64 only)_
- Base image: `nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime` (Tegra-optimised)
- Engine build ~5 min on Jetson; subsequent starts instant
- **Only for Jetson Orin/Nano** — use `object-detection-custom-inference-tensorrt` on everything else

### object-detection-custom-stream-viewer
- Reads SHM frames, re-encodes to JPEG, pushes over WebSocket
- Subscribes to ZeroMQ PUB for detection overlays
- aiohttp async server — handles many clients without extra threads


## Building

`build.sh` in each service directory builds and optionally pushes that
service's image. `build-all.sh` at the project root orchestrates all of them.

**Default behaviour: build host arch only, no push (local images).**

```bash
# Build all services locally
./build-all.sh


# Build both amd64 and arm64, then push
./build-all.sh --cross

# Skip both TensorRT images
./build-all.sh --skip-tensorrt

# Build Jetson TensorRT only (skip desktop/server)
./build-all.sh --jetson-only

# Desktop/server TensorRT only (skip Jetson)
./build-all.sh --no-jetson

# Custom registry and stable manifest tag
./build-all.sh --registry quay.io/myorg --prod-tag v1.2
```

### How multi-arch manifests work

Each `build.sh` always tags the built image as `:amd64` or `:arm64`.
When `--push` is given, it also creates `:latest` and `:prod` (or
`--prod-tag <name>`) multi-arch manifests. If only one arch was built,
the script pulls the other arch tag from the registry so the manifest
stays multi-arch. Use `--force-manifest-reset` to start the manifest
from scratch instead.



## Quick start

### ONNX (GPU-first, CPU fallback)

```bash
# 1. Copy your ONNX model
cp my_model.onnx models/

# 2. Build images or pull them from registry

# 3. Check which video device to pass (see Diagnostics)
#    Then edit _run_/compose/compose.yml → object-detection-custom-camera-capture → devices:

# 4. Start
cd _run_/compose
podman compose --profile onnx up -d

# 5. Open browser
firefox http://localhost:8080
```

### Run TensorRT — desktop/server GPU (x86_64 or generic arm64)

```bash
cd _run_/compose
podman compose --profile tensorrt up -d

podman logs -f object-detection-custom-inference-tensorrt  # first run builds engine (~2–5 min)
firefox http://localhost:8080
```

### Run TensorRT — Jetson (JetPack 6.x, aarch64)

```bash
cd _run_/compose
podman compose --profile tensorrt-jetson up -d

podman logs -f object-detection-custom-inference-tensorrt-jetson   # first run ~5 min on Jetson
firefox http://localhost:8080
```

### Video file fallback (no camera)

The `../../videos:/videos:ro` volume is already active in compose.yml.
Drop your video files in the `videos/` directory and start normally.
object-detection-custom-camera-capture will use video automatically if no camera is found, and
will keep checking for a live camera every 30 s (configurable via
`CAMERA_RETRY_INTERVAL`).

```bash
cp my_video.mp4 videos/
podman compose --profile onnx up -d
```

A demo clip is included at `videos/demo.mp4`.

### Podman Quadlets (systemd)

> NOTE: Run from a non-root user

```bash
mkdir -p ~/.config/containers/systemd
cp _run_/quadlets/object-detection-custom-camera-capture.container ~/.config/containers/systemd/
cp _run_/quadlets/object-detection-custom-stream-viewer.container  ~/.config/containers/systemd/

# Pick ONE inference quadlet:
cp _run_/quadlets/object-detection-custom-inference-tensorrt-jetson.container   ~/.config/containers/systemd/
# or:
#cp _run_/quadlets/object-detection-custom-inference-onnx.container              ~/.config/containers/systemd/
# or:
#cp _run_/quadlets/object-detection-custom-inference-tensorrt.container          ~/.config/containers/systemd/


systemctl --user daemon-reload
systemctl --user start object-detection-custom-camera-capture.service
systemctl --user start object-detection-custom-inference-tensorrt-jetson.service
# or:
#systemctl --user start object-detection-custom-inference-onnx.service
# or:
#systemctl --user start object-detection-custom-inference-tensorrt.service
systemctl --user start object-detection-custom-stream-viewer.service
```


## Diagnostics

### Identify the right /dev/video device

USB cameras expose multiple `/dev/video*` nodes. Find the capture node:

```bash
# Human-readable list of devices
v4l2-ctl --list-devices

# Find which nodes support Video Capture (not metadata):
for dev in /dev/video*; do
  echo "=== $dev ==="
  v4l2-ctl --device "$dev" --all 2>/dev/null | grep -E "Card type|Bus info|Capture"
done

# List supported modes for a specific device:
v4l2-ctl --device /dev/video0 --list-formats-ext
```

Once you have the right node (e.g. `/dev/video2`), set it in:

- **Compose:** `_run_/compose/compose.yml` → `object-detection-custom-camera-capture` → `devices:`
- **Quadlet:** `_run_/quadlets/object-detection-custom-camera-capture.container` → `AddDevice=`

Or force it at runtime without editing files:

```bash
CAMERA_DEVICE=/dev/video2 podman compose --profile onnx up -d
```

### Check and configure the model path

If you want to use your own model, you need to make it available inside the
inference container. Check the compose file or quadlet to confirm the volume
mapping is correct before starting:

- **Compose** (`_run_/compose/compose.yml`): look for the `volumes:` section
  under the inference service. By default `../../models:/opt/models` is
  mounted. Put your `.onnx` file in `models/` and set `INFERENCE_MODEL` to
  its filename.
- **Quadlet**: check the `Volume=` line in the relevant `.container` file.
  Add or update it to point at your local model directory.

To verify the model is visible inside a running container:

```bash
podman exec object-detection-custom-inference-onnx              ls -lh /opt/models/
podman exec object-detection-custom-inference-tensorrt          ls -lh /opt/models/
podman exec object-detection-custom-inference-tensorrt-jetson   ls -lh /opt/models/
```

### Enable verbose capture stats

By default, the periodic fps/resolution/payload log lines are suppressed to
keep logs clean. Enable them when debugging capture issues:

```bash
# In compose.yml, under object-detection-custom-camera-capture → environment:
VERBOSE_STATS: "1"

# Or at runtime:
VERBOSE_STATS=1 podman compose --profile onnx up -d
```

With `VERBOSE_STATS=1` you'll see lines like:
```
[INFO] /dev/video5  30.0 fps  1280x720  payload=2764800 B  BGR
```


## Environment variables

### object-detection-custom-camera-capture
| Variable | Default | Description |
|---|---|---|
| `CAMERA_DEVICE` | auto | `/dev/videoN`; auto-detects if unset |
| `CAM_TARGET_FPS` | 30 | Desired FPS — used for mode selection |
| `CAM_FRAMERATE` | auto | Override FPS passed to v4l2 |
| `CAM_RESOLUTION` | auto | Override resolution e.g. `1280x720` |
| `CAM_FORMAT` | auto | Force `MJPG` or `YUYV` |
| `DEVICE_PROBE_TIMEOUT` | 5 | Seconds per ffmpeg probe attempt |
| `VID_DIR` | /videos | Directory with fallback video files |
| `CAMERA_RETRY_INTERVAL` | 30 | Seconds between camera re-probe attempts while in video fallback |
| `SHM_FRAME_NAME` | camera_frame | SHM slot name |
| `SHM_MAX_FRAME_BYTES` | 26542080 | Max frame buffer size (default: 4K raw) |
| `VIEWER_JPEG_QUALITY` | 0 | JPEG quality in SHM (0 = raw BGR) |
| `VERBOSE_STATS` | 0 | Set to `1` to log periodic fps/payload stats |
| `LOG_LEVEL` | INFO | DEBUG / INFO / WARNING |

### object-detection-custom-inference-onnx / object-detection-custom-inference-tensorrt / object-detection-custom-inference-tensorrt-jetson
| Variable | Default | Description |
|---|---|---|
| `SHM_FRAME_NAME` | camera_frame | SHM slot name |
| `SHM_MAX_FRAME_BYTES` | 26542080 | Must match object-detection-custom-camera-capture |
| `ZMQ_PUB_PORT` | 5555 | ZeroMQ PUB port |
| `INFERENCE_MODEL` | model.onnx | Model filename in `MODELS_DIR` |
| `MODELS_DIR` | /opt/models | Model directory inside container |
| `INFERENCE_WIDTH/HEIGHT` | 640 | Input resolution |
| `CONFIDENCE_THRESH` | 0.4 | Minimum confidence |
| `NMS_THRESH` | 0.45 | NMS IoU threshold |
| `TARGET_FPS` | 30 | Max inference rate |
| `EXECUTION_PROVIDER` | auto | ONNX EP (auto = TRT→CUDA→CPU) |
| `TRT_FP16` | 1 | TensorRT FP16 mode |
| `CLASS_NAMES` | — | Comma-separated class names |
| `CLASS_NAMES_FILE` | — | Path to names file |

### object-detection-custom-stream-viewer
| Variable | Default | Description |
|---|---|---|
| `SHM_FRAME_NAME` | camera_frame | SHM slot name |
| `SHM_MAX_FRAME_BYTES` | 26542080 | Must match object-detection-custom-camera-capture |
| `ZMQ_SUB_HOST` | localhost | ZeroMQ PUB host |
| `ZMQ_SUB_PORT` | 5555 | ZeroMQ PUB port |
| `HTTP_PORT` | 8080 | HTTP/WebSocket port |
| `FRAME_JPEG_QUALITY` | 75 | Browser JPEG quality |
| `MAX_STREAM_FPS` | 35 | Max frames pushed to browser |


## ZeroMQ detection schema

```json
{
  "seq": 1234,
  "ts_capture": 12345.678,
  "ts_infer":   12345.710,
  "latency_ms": 32.1,
  "frame_w": 640,
  "frame_h": 480,
  "model": "model.onnx",
  "detections": [
    {
      "class_id":   0,
      "class_name": "person",
      "confidence": 0.87,
      "bbox": { "x1": 100, "y1": 50, "x2": 300, "y2": 400,
                "cx": 200, "cy": 225, "w": 200, "h": 350 }
    }
  ]
}
```

Subscribe from any service:

```python
import zmq, json
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5555")
sub.setsockopt_string(zmq.SUBSCRIBE, "detections")
while True:
    topic, payload = sub.recv_string().split(" ", 1)
    for det in json.loads(payload)["detections"]:
        print(det["class_name"], det["confidence"])
```
