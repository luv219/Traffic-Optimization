# Traffic Optimization

An end-to-end traffic management system that combines **YOLO vehicle detection**, **zone-based counting**, **adaptive signal control algorithms**, and a **real-time web dashboard**. The project simulates a four-way intersection while processing live video to track vehicles in defined lane zones.

## Features

- **Vehicle detection & tracking** — YOLOv11 with persistent track IDs across frames
- **Zone-based counting** — Polygon regions defined in `counting/zones.yaml` for per-lane active and cumulative vehicle counts
- **Traffic signal controllers**
  - **Fixed-Time** — Cycles green/yellow phases on a fixed schedule
  - **Actuated** — Extends or switches phases based on queue presence
  - **Adaptive** — Max-pressure inspired controller that favors the axis with higher queue pressure
- **Intersection simulator** — Stochastic vehicle arrivals, departures, throughput, and delay metrics
- **Web dashboard** — Live video feed, real-time charts, controller comparison, and simulation controls

## Project Structure

```
Traffic-Optimization/
├── annotation/          # COCO-format CSV annotations for dataset prep
├── controller/
│   ├── algos.py         # Signal control algorithms
│   └── simulator.py     # Four-way intersection simulator
├── counting/
│   ├── counter.py       # YOLO tracking + zone counting
│   └── zones.yaml       # Lane polygon definitions
├── data/
│   ├── images/          # Train/val images
│   ├── labels/          # YOLO-format labels
│   └── videos/          # Sample traffic video
├── detectors/
│   ├── prepare_data.py  # Download COCO images and build dataset
│   ├── train.py         # Fine-tune YOLO on traffic data
│   └── infer.py         # Run inference on images/videos
├── scripts/
│   └── prepare_dataset.sh
├── tests/
│   └── test_algos.py    # Unit tests for controllers and simulator
├── ui/
│   ├── dashboard.py     # FastAPI server
│   └── templates/
│       └── index.html   # Dashboard UI
└── data.yaml            # YOLO dataset configuration
```

## Requirements

- Python 3.9+
- Dependencies:

```bash
pip install ultralytics opencv-python numpy pyyaml fastapi uvicorn jinja2 pydantic requests
```

A GPU is optional but recommended for YOLO training and faster inference.

## Quick Start

### 1. Prepare the dataset

Downloads COCO traffic images, converts annotations to YOLO format, generates `data.yaml`, and fetches a sample traffic video:

```bash
python detectors/prepare_data.py
```

Or on Unix:

```bash
bash scripts/prepare_dataset.sh
```

By default this downloads 15 training and 5 validation images plus `data/videos/traffic.mp4`.

### 2. Run inference

```bash
python detectors/infer.py --source data/images/train/000000140006.jpg
```

Options: `--model`, `--source`, `--conf`, `--save-dir`. Results are saved under `runs/detect/infer/`.

### 3. Train a custom model (optional)

```bash
python detectors/train.py --epochs 50 --batch 8 --device 0
```

Training outputs are written to `runs/train/traffic_yolo/`. Place the best weights at `models/yolo11n.pt` for the dashboard and counter to use them.

### 4. Start the dashboard

From the project root:

```bash
python -m ui.dashboard
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

The dashboard provides:

- Live MJPEG video feed with YOLO detections and zone overlays (`/video_feed`)
- Real-time zone counts (`/live_counts`)
- Server-sent events stream for simulator metrics (`/simulation_stream`)
- Controller switching, arrival-rate tuning, pause/resume/reset
- Side-by-side controller comparison over 200 simulation steps (`/compare`)

### 5. Run tests

```bash
python -m unittest discover -s tests -v
```

## Configuration

### Counting zones (`counting/zones.yaml`)

Define polygon regions in image coordinates. Each zone has a name, BGR color, and vertex list:

```yaml
zones:
  - name: "left_lane"
    color: [255, 0, 0]
    polygon:
      - [50, 180]
      - [370, 180]
      - [250, 420]
      - [0, 420]
```

Vehicles are counted when the bottom-center of their bounding box falls inside a zone.

### Dataset (`data.yaml`)

Points YOLO to the `data/` directory with train/val splits and COCO class names. Regenerated automatically by `prepare_data.py`.

## How It Works

```
Video feed ──► YOLO tracking ──► Zone counter ──► Dashboard
                                    │
Intersection simulator ◄── Controllers (Fixed / Actuated / Adaptive)
        │
        └──► Throughput, delay, queue metrics ──► Dashboard charts
```

1. **Detection** — `VehicleCounter` runs YOLO with `persist=True` to maintain track IDs and filters COCO vehicle classes (bicycle, car, motorcycle, bus, truck).
2. **Counting** — Each tracked vehicle is assigned to a zone via point-in-polygon test on its bottom-center point.
3. **Simulation** — `TrafficSimulator` models a four-phase signal (NS green → NS yellow → EW green → EW yellow) with stochastic arrivals and configurable departure rates.
4. **Control** — Controllers receive queue lengths, current phase, and phase timer each second and return `KEEP` or `SWITCH`.
5. **Dashboard** — FastAPI ties the video pipeline and simulator together with SSE for live updates.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard UI |
| `GET` | `/video_feed` | MJPEG stream with detections |
| `GET` | `/live_counts` | Current and cumulative zone counts |
| `GET` | `/simulation_stream` | SSE stream of simulator state |
| `POST` | `/config` | Update controller, arrival rates, sim speed |
| `POST` | `/control/pause` | Pause simulation |
| `POST` | `/control/resume` | Resume simulation |
| `POST` | `/control/reset` | Reset simulator state |
| `GET` | `/compare` | Benchmark all three controllers |

## License

See repository license for details.
