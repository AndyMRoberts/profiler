# Profiler

Python package for recording power, CPU, GPU, and memory metrics during testing of ML models, SLAM systems, etc.


# Pros
- simple to run, once installed all data is saved to its own "timestamp_title" folder to ensure data organisation
- this location is returned so the user can add any other logs to the same folder, taking care or logging infrastructure
- ability to record reference data which is automatically subtracted from later logs. But raw data is left untouched to maintain all original data. But the metadata and plots provide the reference adjusted values, as would be required in publications. 
- Recording a reference dataset during a constant idle state of the system allows standard deviation to be logged as well and applied to the run graphs.
- Detailed, publication-worthy graphs with infill +/- 1 standard deviation for quick run data inspection. 



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

# optional: record a reference set of data first:
# these are automatically recorded to the {output_dir}/reference folder. So subsequent recordings will
# overwrite each other. Move the reference directory to another folder and provide filepath later on if desired.
print(f'Recording reference to {output_dir}/reference')
ref_dir  = p.record_reference(duration_seconds=10)

# option to store own data in reference directory
with open(f'{ref_dir}/extra_data.txt', 'w') as f:
    f.write('This data was recorded whilst the system was in a rest state.')

# Start recording (runs in background thread) 
# if a reference has been recorded
run_dir = p.start(
    use_reference=True,
    ref_dir = '' # if blank or not provided the code will pick up the last previously recorded reference
    )

# ... run your ML model, SLAM pipeline, etc. ...

# Stop and write outputs; optionally pass num_frames for energy-per-frame stats
p.stop(num_frames=100)

# option to add data the run_dir
with open(f'{run_dir}/extra_data.txt', 'w') as f:
    f.write('Run completed successfully')

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
