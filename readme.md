# Profiler

Python package for recording power, CPU, GPU, and memory metrics during testing of ML models, SLAM systems, etc.

## Install

```bash
pip install -e .         # GPU memory, power, CPU usage, etc. (nvidia-ml-py, pyJoules, psutil, matplotlib)
pip install -e .[test]   # Add pytest for running tests
```

## Usage

```python
from profiler import Profiler, ProfilerSetupError

# Optional: verify before a long run - fails immediately if packages/permissions wrong
Profiler.verify_setup()

# Create profiler: output directory, sampling frequency (Hz), title
# A new subdirectory is created per run: {dir}/{YYYY}_{MM}_{DD}_{HHMM}_{title_with_underscores}/
p = Profiler("runs", frequency_hz=2.0, title="SLAM inference test")

# Start recording (runs in background thread)
p.start()

# ... run your ML model, SLAM pipeline, etc. ...

# Stop and write outputs; optionally pass num_frames for energy-per-frame stats
p.stop(num_frames=100)

# Outputs (in e.g. runs/2026_02_19_1430_SLAM_inference_test/):
#   data.csv      - Time-series data (time, cpu %, cpu power, gpu power, etc.)
#   plot.png      - Matplotlib plot of all metrics
#   metadata.json - Run time, title, energy per frame (min/max/avg) if num_frames given
```

## Testing

Tests live in `tests/` at the project root (separate from the installable package):

```bash
source activate.sh   # Activate venv first
pip install -e .[test]
pytest
```

## Permissions (CPU power and GPU metrics)

- **CPU power** (pyJoules/RAPL): Requires read access to `/sys/class/powercap/`. On many systems this needs root or:
  ```bash
  sudo chmod -R a+r /sys/class/powercap/intel-rapl
  ```
- **GPU metrics** (nvidia-ml-py): Requires NVIDIA drivers and typically user in `video` group. If metrics are empty, check `nvidia-smi` works.

**If metrics are missing**, run the diagnostic to see exactly what fails:
```bash
python -m profiler.diagnose
```

Check `metadata.json` after a run: `metrics_available` shows which optional metrics were successfully recorded.

## Activate venv

Venv is stored in a separate folder to prevent conflicts. Run `source activate.sh` or `. activate.sh` to activate.
