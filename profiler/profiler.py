"""
Profiler for recording power, CPU, GPU, and memory metrics during ML/SLAM testing.
Outputs: CSV data, PNG plot, and metadata JSON.
"""

import csv
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Core dependencies (always available)
import psutil

# Required: CPU power (Linux, Intel RAPL) and GPU (NVIDIA)
_PYJOULES_IMPORT_ERROR = None  # Set if pyJoules import fails
_PYNVML_IMPORT_ERROR = None  # Set if pynvml import fails
try:
    from pyJoules.device.rapl_device import RaplPackageDomain
    from pyJoules.energy_meter import EnergyMeter
    from pyJoules.device import DeviceFactory
    PYOULES_AVAILABLE = True
except ImportError as e:
    PYOULES_AVAILABLE = False
    _PYJOULES_IMPORT_ERROR = e

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError as e:
    PYNVML_AVAILABLE = False
    _PYNVML_IMPORT_ERROR = e


class ProfilerSetupError(Exception):
    """Raised when required packages or hardware access (GPU, RAPL) are unavailable."""


@dataclass
class Sample:
    """Single sampling point of system metrics."""
    timestamp: float
    elapsed_s: float
    cpu_usage_percent: float
    cpu_power_w: Optional[float]
    gpu_power_w: Optional[float]
    gpu_usage_percent: Optional[float]
    gpu_memory_gb: Optional[float]
    gpu_memory_percent: Optional[float]
    ram_usage_percent: float
    ram_used_gb: float
    ram_available_gb: float


class Profiler:
    """
    Records system metrics (power, CPU, GPU, memory) at a configurable frequency.

    Use from other projects to profile ML model inference, SLAM pipelines, etc.

    Call Profiler.verify_setup() before long runs to fail fast if packages/permissions are wrong.
    """

    @staticmethod
    def verify_setup() -> None:
        """
        Verify that required packages and hardware access work. Raises ProfilerSetupError on failure.
        Call before long runs to fail fast:  Profiler.verify_setup()
        """
        p = Profiler.__new__(Profiler)
        p.output_directory = Path(".")
        p.frequency_hz = 1.0
        p.title = ""
        p._run_dir = None
        p._samples = []
        p._running = False
        p._thread = None
        p._start_time = None
        p._start_wall_time = None
        p._energy_meter = None
        p._energy_samples = []
        p._gpu_handle = None
        p._sample_error = None
        p._validate_and_init()
        # Cleanup what we inited
        if p._energy_meter is not None:
            p._energy_meter.stop()
        if PYNVML_AVAILABLE and p._gpu_handle is not None:
            pynvml.nvmlShutdown()

    def __init__(
        self,
        output_directory: str,
        frequency_hz: float = 1.0,
        title: str = "Profiling run",
        cpu_power_max_w: Optional[float] = None,
        gpu_power_max_w: Optional[float] = None,
    ):
        """
        Args:
            output_directory: Base directory for run outputs. A new subdir is created per run:
                {output_directory}/{YYYY}_{MM}_{DD}_{HHMM}_{title_with_underscores}/
                containing metadata.json, data.csv, plot.png
            frequency_hz: Sampling frequency in Hz (e.g. 1.0 = once per second)
            title: Title for the profiling run (spaces → underscores in run dir name), stored in metadata.
            cpu_power_max_w: Optional upper y-axis limit (W) for the CPU power subplot.
            gpu_power_max_w: Optional upper y-axis limit (W) for the GPU power subplot.
        """
        self.output_directory = Path(output_directory)
        self.frequency_hz = frequency_hz
        self.title = title
        self.cpu_power_max_w = cpu_power_max_w
        self.gpu_power_max_w = gpu_power_max_w
        self._gpu_memory_total_gb: Optional[float] = None
        self._run_dir: Optional[Path] = None  # Set in stop() when run directory is created
        self._samples: list[Sample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None  # perf_counter for elapsed time
        self._start_wall_time: Optional[float] = None  # time.time() for run directory timestamp
        self._energy_meter = None
        self._energy_samples: list[dict] = []
        self._gpu_handle = None
        self._sample_error: Optional[Exception] = None  # Set if sampling loop fails
        self._run_dir_override: Optional[Path] = None  # e.g. output_directory / "reference"
        self._use_reference = False
        self._ref_dir: Optional[Path] = None  # location of reference folder when use_reference=True
        self._output_written = False  # True after stop() has written outputs (prevents double stop)

    def _validate_and_init(self) -> None:
        """Validate required packages and init GPU/RAPL. Raises ProfilerSetupError on failure."""
        errors = []

        # Check nvidia-ml-py and GPU
        if not PYNVML_AVAILABLE:
            err = _PYNVML_IMPORT_ERROR or ImportError("nvidia-ml-py not installed")
            errors.append(f"GPU (nvidia-ml-py): {err}. Install: pip install nvidia-ml-py")
        else:
            try:
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                if count == 0:
                    pynvml.nvmlShutdown()
                    errors.append("GPU: No NVIDIA GPUs found. Ensure nvidia-smi works.")
                else:
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    # Verify we can read at least one metric
                    try:
                        pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    except Exception as e:
                        self._gpu_handle = None
                        pynvml.nvmlShutdown()
                        errors.append(f"GPU: Cannot read utilization: {e}")
                    # Read total GPU memory for plot y-axis upper limit
                    if self._gpu_handle is not None:
                        try:
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                            self._gpu_memory_total_gb = mem_info.total / (1024 ** 3)
                        except Exception:
                            pass  # Non-critical: plot will auto-scale the GPU memory axis
            except Exception as e:
                errors.append(f"GPU: {e}. Ensure nvidia-smi works and you're in 'video' group.")

        # Check pyJoules and RAPL
        if not PYOULES_AVAILABLE:
            err = _PYJOULES_IMPORT_ERROR or ImportError("pyJoules not installed")
            errors.append(f"CPU power (pyJoules): {err}. Install: pip install pyJoules")
        else:
            try:
                devices = DeviceFactory.create_devices([RaplPackageDomain(0)])
                self._energy_meter = EnergyMeter(devices)
                self._energy_meter.start()
            except Exception as e:
                errors.append(
                    f"CPU power (RAPL): {e}. "
                    "Try: sudo chmod -R a+r /sys/class/powercap/intel-rapl"
                )

        if errors:
            raise ProfilerSetupError(
                "Profiler setup failed:\n  - " + "\n  - ".join(errors)
            )

    def _shutdown_gpu(self):
        """Clean up GPU resources."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass  # Best-effort cleanup on shutdown
        self._gpu_handle = None
    
    def _read_gpu_metrics(self) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Returns (gpu_power_w, gpu_usage_%, gpu_memory_gb, gpu_memory_%). Raises on failure."""
        if self._gpu_handle is None:
            return None, None, None, None
        power_w = None
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
            power_w = power_mw / 1000.0
        except Exception:
            pass  # Some GPUs (e.g. GeForce) don't support power reporting
        util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
        gpu_usage = float(util.gpu)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
        gpu_mem_gb = mem.used / (1024 ** 3)
        gpu_mem_pct = (mem.used / mem.total * 100) if mem.total > 0 else None
        return power_w, gpu_usage, gpu_mem_gb, gpu_mem_pct

    def _compute_averages(self) -> dict:
        """Compute average of each metric across samples. None values excluded from mean."""
        if not self._samples:
            return {}
        n = len(self._samples)
        cpu_power_vals = [s.cpu_power_w for s in self._samples if s.cpu_power_w is not None]
        gpu_power_vals = [s.gpu_power_w for s in self._samples if s.gpu_power_w is not None]
        gpu_usage_vals = [s.gpu_usage_percent for s in self._samples if s.gpu_usage_percent is not None]
        gpu_mem_gb_vals = [s.gpu_memory_gb for s in self._samples if s.gpu_memory_gb is not None]
        gpu_mem_pct_vals = [s.gpu_memory_percent for s in self._samples if s.gpu_memory_percent is not None]
        return {
            "cpu_usage_percent": sum(s.cpu_usage_percent for s in self._samples) / n,
            "cpu_power_w": sum(cpu_power_vals) / len(cpu_power_vals) if cpu_power_vals else None,
            "gpu_power_w": sum(gpu_power_vals) / len(gpu_power_vals) if gpu_power_vals else None,
            "gpu_usage_percent": sum(gpu_usage_vals) / len(gpu_usage_vals) if gpu_usage_vals else None,
            "gpu_memory_gb": sum(gpu_mem_gb_vals) / len(gpu_mem_gb_vals) if gpu_mem_gb_vals else None,
            "gpu_memory_percent": sum(gpu_mem_pct_vals) / len(gpu_mem_pct_vals) if gpu_mem_pct_vals else None,
            "ram_usage_percent": sum(s.ram_usage_percent for s in self._samples) / n,
            "ram_used_gb": sum(s.ram_used_gb for s in self._samples) / n,
            "ram_available_gb": sum(s.ram_available_gb for s in self._samples) / n,
        }

    def _compute_stddevs(self, averages: dict) -> dict:
        """Sample standard deviation for each metric (same keys as averages). n-1 denominator."""
        if not self._samples or not averages:
            return {}
        n = len(self._samples)
        if n <= 1:
            return {k: 0.0 if averages.get(k) is not None else None for k in averages}

        def _std(vals: list, mean_val: Optional[float]) -> Optional[float]:
            if mean_val is None or not vals:
                return None
            if len(vals) <= 1:
                return 0.0
            variance = sum((x - mean_val) ** 2 for x in vals) / (len(vals) - 1)
            return (variance ** 0.5) if variance >= 0 else 0.0

        cpu_power_vals = [s.cpu_power_w for s in self._samples if s.cpu_power_w is not None]
        gpu_power_vals = [s.gpu_power_w for s in self._samples if s.gpu_power_w is not None]
        gpu_usage_vals = [s.gpu_usage_percent for s in self._samples if s.gpu_usage_percent is not None]
        gpu_mem_gb_vals = [s.gpu_memory_gb for s in self._samples if s.gpu_memory_gb is not None]
        gpu_mem_pct_vals = [s.gpu_memory_percent for s in self._samples if s.gpu_memory_percent is not None]

        return {
            "cpu_usage_percent": _std([s.cpu_usage_percent for s in self._samples], averages.get("cpu_usage_percent")),
            "cpu_power_w": _std(cpu_power_vals, averages.get("cpu_power_w")),
            "gpu_power_w": _std(gpu_power_vals, averages.get("gpu_power_w")),
            "gpu_usage_percent": _std(gpu_usage_vals, averages.get("gpu_usage_percent")),
            "gpu_memory_gb": _std(gpu_mem_gb_vals, averages.get("gpu_memory_gb")),
            "gpu_memory_percent": _std(gpu_mem_pct_vals, averages.get("gpu_memory_percent")),
            "ram_usage_percent": _std([s.ram_usage_percent for s in self._samples], averages.get("ram_usage_percent")),
            "ram_used_gb": _std([s.ram_used_gb for s in self._samples], averages.get("ram_used_gb")),
            "ram_available_gb": _std([s.ram_available_gb for s in self._samples], averages.get("ram_available_gb")),
        }

    def _sample_loop(self):
        """Background thread: collect metrics at the requested frequency."""
        interval = 1.0 / self.frequency_hz

        first_sample = True
        try:
            while self._running:
                if first_sample == True:
                    # initial cpu power reading will be dsicarded so is started one interval before other readings. 
                    if self._energy_meter is not None:
                        self._energy_meter.record(tag=f"t{-interval}")
                    first_sample = False
                    time.sleep(interval)
                    self._start_time = time.perf_counter()
                    continue
                # CPU power: record with pyJoules (RAPL) - energy between records backfilled after stop()
                # First record is skipped in backfill as it is normally erroneously high
                if self._energy_meter is not None:
                    self._energy_meter.record(tag=f"t{len(self._samples)}")

                loop_start = time.perf_counter()
                elapsed = loop_start - self._start_time

                # CPU usage
                cpu_pct = psutil.cpu_percent(interval=None)

                # RAM
                vm = psutil.virtual_memory()

                # GPU
                gpu_power, gpu_usage, gpu_mem_gb, gpu_mem_pct = self._read_gpu_metrics()

                sample = Sample(
                    timestamp=loop_start,
                    elapsed_s=elapsed,
                    cpu_usage_percent=cpu_pct,
                    cpu_power_w=None,  # Backfilled from pyJoules trace after stop()
                    gpu_power_w=gpu_power,
                    gpu_usage_percent=gpu_usage,
                    gpu_memory_gb=gpu_mem_gb,
                    gpu_memory_percent=gpu_mem_pct,
                    ram_usage_percent=vm.percent,
                    ram_used_gb=vm.used / (1024 ** 3),
                    ram_available_gb=vm.available / (1024 ** 3),
                )
                self._samples.append(sample)

                # Sleep to maintain frequency
                elapsed_in_loop = time.perf_counter() - loop_start
                sleep_time = max(0, interval - elapsed_in_loop)
                if sleep_time > 0 and self._running:
                    time.sleep(sleep_time)

        except Exception as e:
            self._sample_error = e
    
    def record_reference(self, duration_seconds: int) -> str:
        """
        Record a reference set of data for the given duration (in seconds).
        Saves to {output_directory}/reference/ (no timestamp). Same outputs as a normal run:
        data.csv, plot.png, metadata.json (including averages used for reference-adjusted runs).

        Args:
            duration_seconds: How long to record (integer seconds).

        Returns:
            Path to the run directory (reference folder) for storing your own data.
        """
        self._run_dir_override = self.output_directory / "reference"
        run_dir = self.start()
        try:
            time.sleep(int(duration_seconds))
        finally:
            self.stop(num_frames=None)
        self._run_dir_override = None
        return run_dir

    def start(
        self,
        use_reference: bool = False,
        ref_dir: Optional[str] = None,
    ) -> str:
        """
        Start recording metrics in a background thread. Raises ProfilerSetupError if setup fails.
        Creates the run directory immediately so you can store your own data there.

        Args:
            use_reference: If True, load reference metadata from ref_dir and subtract reference
                averages from this run's metadata and plot. CSV remains raw (non-referenced).
            ref_dir: Optional path to the reference folder. If use_reference is True and ref_dir
                is not provided, uses {output_directory}/reference.

        Returns:
            Path to the run directory for storing your own data.
        """
        if self._running:
            return str(self._run_dir) if self._run_dir else ""
        self._samples.clear()
        self._sample_error = None
        self._output_written = False  # allow this run to be stopped and written (clears previous run’s dir)
        self._use_reference = use_reference
        self._ref_dir = Path(ref_dir) if ref_dir else (self.output_directory / "reference" if use_reference else None)
        self._start_wall_time = time.time()
        # Create run directory now so user can use it immediately
        if self._run_dir_override is not None:
            self._run_dir = self._run_dir_override
        else:
            dt = datetime.fromtimestamp(self._start_wall_time)
            sanitized_title = re.sub(r"[^\w\-]", "_", self.title).strip("_") or "run"
            run_dir_name = f"{dt:%Y_%m_%d_%H%M}_{sanitized_title}"
            self._run_dir = self.output_directory / run_dir_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()
        self._running = True

        # Validate packages and init GPU + RAPL; raises ProfilerSetupError on failure
        self._validate_and_init()

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return str(self._run_dir)

    def stop(self, num_frames: Optional[int] = None):
        """
        Stop recording and write all output files.
        
        Args:
            num_frames: If provided, compute energy per frame (min, max, avg) from
                       total energy and store in metadata. Useful for ML/SLAM frame-based workloads.
        """
        if self._output_written:
            return  # Already stopped and written
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Cleanup before potentially raising
        if self._energy_meter is not None:
            try:
                self._energy_meter.stop()
            except Exception as e:
                raise RuntimeError(f"Failed to stop energy meter: {e}") from e
        self._shutdown_gpu()

        # Re-raise any error that occurred during sampling
        if self._sample_error is not None:
            err = self._sample_error
            self._sample_error = None
            raise err

        # _run_dir was already created in start()
        # Backfill CPU power from pyJoules trace (energy/duration per interval)
        # RAPL reports energy in microjoules (µJ); convert to J then power = J/s = W
        if self._energy_meter is not None and self._samples:
            trace = self._energy_meter.get_trace()

            if trace and trace._samples:
                for i, sample in enumerate(trace._samples):
                    if i == 0:
                        # Skip first trace interval (erroneously high)
                        continue
                    sample_idx = i - 1  # trace[1] -> samples[0], trace[2] -> samples[1], ...
                    if sample_idx < len(self._samples) and sample.duration > 0 and sample.energy:
                        total_uj = sum(sample.energy.values())  # microjoules
                        total_j = total_uj / 1e6  # convert to joules
                        self._samples[sample_idx].cpu_power_w = total_j / sample.duration

        run_time = time.perf_counter() - self._start_time if self._start_time else 0
        
        # Compute energy per frame stats if requested (and sum_dt_sq for error propagation)
        energy_per_frame_min = energy_per_frame_max = energy_per_frame_avg = None
        sum_dt_sq: float = 0.0
        if num_frames is not None and num_frames > 0 and self._samples:
            total_energy_j = 0
            time_per_frame = run_time / num_frames
            for i in range(1, len(self._samples)):
                dt = self._samples[i].elapsed_s - self._samples[i - 1].elapsed_s
                if dt <= 0:
                    dt = 1.0 / self.frequency_hz
                sum_dt_sq += dt * dt
                cpu_w = self._samples[i].cpu_power_w or 0
                gpu_w = self._samples[i].gpu_power_w or 0
                total_energy_j += (cpu_w + gpu_w) * dt

            energy_per_frame_avg = total_energy_j / num_frames
            # Min/max: (min/max power across samples) * time per frame
            total_powers = [
                (s.cpu_power_w or 0) + (s.gpu_power_w or 0)
                for s in self._samples
            ]
            if total_powers and time_per_frame > 0:
                energy_per_frame_min = min(total_powers) * time_per_frame
                energy_per_frame_max = max(total_powers) * time_per_frame
        
        # Compute which optional metrics were actually recorded (for diagnostics)
        metrics_available = {
            "cpu_power": any(s.cpu_power_w is not None for s in self._samples),
            "gpu_power": any(s.gpu_power_w is not None for s in self._samples),
            "gpu_usage": any(s.gpu_usage_percent is not None for s in self._samples),
            "gpu_memory": any(s.gpu_memory_gb is not None for s in self._samples),
        }

        averages = self._compute_averages()
        stddevs = self._compute_stddevs(averages) if averages else {}
        ref_averages: Optional[dict] = None
        ref_stddevs: Optional[dict] = None
        ref_energy_per_frame_j: Optional[dict] = None
        if self._use_reference and self._ref_dir is not None:
            ref_meta_path = self._ref_dir / "metadata.json"
            if ref_meta_path.exists():
                with open(ref_meta_path) as f:
                    ref_meta = json.load(f)
                raw_av = ref_meta.get("averages") or {}
                ref_averages = {k: v for k, v in raw_av.items() if not k.endswith("_stddev")}
                ref_stddevs = {k[:-7]: v for k, v in raw_av.items() if k.endswith("_stddev") and v is not None}
                ref_energy_per_frame_j = ref_meta.get("energy_per_frame_j")

        # Write outputs (CSV is always raw; plot and metadata use reference adjustment when requested)
        self._write_csv()
        self._write_plot(
            ref_averages=ref_averages,
            ref_stddevs=ref_stddevs,
            title_suffix=" (reference adjusted)" if (self._use_reference and ref_averages) else None,
            averages=averages,
            stddevs=stddevs,
        )
        self._write_metadata(
            run_time_s=run_time,
            num_frames=num_frames,
            energy_per_frame_min=energy_per_frame_min,
            energy_per_frame_max=energy_per_frame_max,
            energy_per_frame_avg=energy_per_frame_avg,
            sum_dt_sq=sum_dt_sq,
            metrics_available=metrics_available,
            averages=averages,
            stddevs=stddevs,
            ref_averages=ref_averages,
            ref_stddevs=ref_stddevs,
            ref_energy_per_frame_j=ref_energy_per_frame_j,
        )
        self._output_written = True

    def _write_csv(self):
        """Write sampled metrics to CSV."""
        csv_path = self._run_dir / "data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s", "cpu_usage_percent", "cpu_power_w",
                "gpu_power_w", "gpu_usage_percent", "gpu_memory_gb", "gpu_memory_percent",
                "ram_usage_percent", "ram_used_gb", "ram_available_gb"
            ])
            for s in self._samples:
                writer.writerow([
                    f"{s.elapsed_s:.2f}",
                    f"{s.cpu_usage_percent:.2f}",
                    f"{s.cpu_power_w:.2f}" if s.cpu_power_w is not None else "",
                    f"{s.gpu_power_w:.2f}" if s.gpu_power_w is not None else "",
                    f"{s.gpu_usage_percent:.2f}" if s.gpu_usage_percent is not None else "",
                    f"{s.gpu_memory_gb:.2f}" if s.gpu_memory_gb is not None else "",
                    f"{s.gpu_memory_percent:.2f}" if s.gpu_memory_percent is not None else "",
                    f"{s.ram_usage_percent:.2f}",
                    f"{s.ram_used_gb:.2f}",
                    f"{s.ram_available_gb:.2f}",
                ])
    
    def _write_plot(
        self,
        ref_averages: Optional[dict] = None,
        ref_stddevs: Optional[dict] = None,
        title_suffix: Optional[str] = None,
        averages: Optional[dict] = None,
        stddevs: Optional[dict] = None,
    ):
        """Generate matplotlib PNG plot of all metrics. If ref_averages given, plot (data - ref). Fill = ±1 stdev."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._samples:
            return

        averages = averages or {}
        stddevs = stddevs or {}
        ref_stddevs = ref_stddevs or {}

        def _fill_stddev(key: str):
            """Stdev for plot fill: use reference stdev when reference data exists, else run stdev."""
            if ref_averages and ref_stddevs and key in ref_stddevs and ref_stddevs[key] is not None:
                return ref_stddevs[key] or 0.0
            return (stddevs.get(key) or 0.0)

        def _sub_ref(vals: list, key: str):
            if not ref_averages or key not in ref_averages or ref_averages[key] is None:
                return [v if v is not None else float("nan") for v in vals]
            ref = ref_averages[key]
            return [(max(0.0, v - ref)) if v is not None else float("nan") for v in vals]

        def _add_fill(ax, times, ys: list, stddev: float, color: str, lower_clamp: float = 0.0):
            """Fill between (value - stdev) and (value + stdev) at each point."""
            if stddev is None or stddev <= 0:
                return
            nan = float("nan")
            lo = []
            hi = []
            for y in ys:
                if y is None or (isinstance(y, float) and y != y):  # NaN check
                    lo.append(nan)
                    hi.append(nan)
                else:
                    lo.append(max(lower_clamp, y - stddev))
                    hi.append(y + stddev)
            ax.fill_between(times, lo, hi, color=color, alpha=0.3)

        times = [s.elapsed_s for s in self._samples]
        title = self.title + (title_suffix or "")

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14)

        # CPU usage
        ax = axes[0, 0]
        color = "blue"
        cpu_usage = [s.cpu_usage_percent for s in self._samples]
        ys_cpu = _sub_ref(cpu_usage, "cpu_usage_percent")
        std = _fill_stddev("cpu_usage_percent")
        _add_fill(ax, times, ys_cpu, std, color, 0.0)
        ax.plot(times, ys_cpu, "-", color=color, label="CPU %")
        ax.set_ylabel("CPU usage (%)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # CPU power
        ax = axes[0, 1]
        color = "green"
        ys_cpup = _sub_ref([s.cpu_power_w for s in self._samples], "cpu_power_w")
        std = _fill_stddev("cpu_power_w")
        _add_fill(ax, times, ys_cpup, std, color, 0.0)
        ax.plot(times, ys_cpup, "-", color=color, label="CPU power (W)")
        ax.set_ylabel("CPU power (W)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # GPU usage
        ax = axes[1, 0]
        color = "orange"
        ys_gpu = _sub_ref([s.gpu_usage_percent for s in self._samples], "gpu_usage_percent")
        std = _fill_stddev("gpu_usage_percent")
        _add_fill(ax, times, ys_gpu, std, color, 0.0)
        ax.plot(times, ys_gpu, "-", color=color, label="GPU %")
        ax.set_ylabel("GPU usage (%)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # GPU power
        ax = axes[1, 1]
        color = "red"
        ys_gpup = _sub_ref([s.gpu_power_w for s in self._samples], "gpu_power_w")
        std = _fill_stddev("gpu_power_w")
        _add_fill(ax, times, ys_gpup, std, color, 0.0)
        ax.plot(times, ys_gpup, "-", color=color, label="GPU power (W)")
        ax.set_ylabel("GPU power (W)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # GPU memory
        ax = axes[2, 1]
        color = "purple"
        ys_gmem = _sub_ref([s.gpu_memory_gb for s in self._samples], "gpu_memory_gb")
        std = _fill_stddev("gpu_memory_gb")
        _add_fill(ax, times, ys_gmem, std, color, 0.0)
        ax.plot(times, ys_gmem, "-", color=color, label="GPU memory (GB)")
        ax.set_ylabel("GPU memory (GB)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # RAM
        ax = axes[2, 0]
        color = "brown"
        ram_usage = [s.ram_usage_percent for s in self._samples]
        ys_ram = _sub_ref(ram_usage, "ram_usage_percent")
        std = _fill_stddev("ram_usage_percent")
        _add_fill(ax, times, ys_ram, std, color, 0.0)
        ax.plot(times, ys_ram, "-", color=color, label="RAM %")
        ax.set_ylabel("RAM usage (%)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        for ax in axes.flat:
            ax.set_xlabel("Time (s)" if ax.get_subplotspec().is_last_row() else "")

        plt.tight_layout()

        # Apply y-axis upper limits: % metrics capped at 100, power/memory from config/system
        _ylim_tops = {
            axes[0, 0]: 100.0,                          # CPU usage (%)
            axes[1, 0]: 100.0,                          # GPU usage (%)
            axes[2, 0]: 100.0,                          # RAM usage (%)
            axes[0, 1]: self.cpu_power_max_w,           # CPU power (W)
            axes[1, 1]: self.gpu_power_max_w,           # GPU power (W)
            axes[2, 1]: self._gpu_memory_total_gb,      # GPU memory (GB)
        }
        for ax in fig.axes:
            ax.set_ylim(bottom=0)
            low, high = ax.get_ylim()
            # Ensure zero is visible: if range is flat or tiny, set a small range so 0 is shown
            if high <= low or (high - low) < 0.01:
                ax.set_ylim(0, 1)
            top = _ylim_tops.get(ax)
            if top is not None:
                ax.set_ylim(0, top)
        png_path = self._run_dir / "plot.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
    
    def _write_metadata(
        self,
        run_time_s: float,
        num_frames: Optional[int],
        energy_per_frame_min: Optional[float],
        energy_per_frame_max: Optional[float],
        energy_per_frame_avg: Optional[float],
        sum_dt_sq: float = 0.0,
        metrics_available: dict = None,
        averages: Optional[dict] = None,
        stddevs: Optional[dict] = None,
        ref_averages: Optional[dict] = None,
        ref_stddevs: Optional[dict] = None,
        ref_energy_per_frame_j: Optional[dict] = None,
    ):
        """Write metadata JSON. All numeric values rounded to 2 decimal places."""
        def _round_val(v):
            return round(v, 2) if v is not None else None

        meta = {
            "timestamp": datetime.now().isoformat(),
            "title": self.title,
            "run_time_s": round(run_time_s, 2),
            "num_samples": len(self._samples),
            "frequency_hz": round(self.frequency_hz, 2),
            "run_directory": str(self._run_dir),
            "metrics_available": metrics_available or {},
        }
        if averages:
            # Each key followed by key_stddev: ref stdev when reference exists, else run stdev.
            meta["averages"] = {}
            stdev_source = (ref_stddevs or {}) if ref_averages else (stddevs or {})
            for k in averages:
                meta["averages"][k] = _round_val(averages[k])
                meta["averages"][k + "_stddev"] = _round_val(stdev_source.get(k))
        if num_frames is not None:
            meta["num_frames"] = num_frames
            if energy_per_frame_avg is not None:
                # Error propagation: E = P*t or sum(P_i*dt_i). Assume 0 error on num_frames.
                # sigma_P = sqrt(sigma_cpu^2 + sigma_gpu^2); sigma_total_energy = sigma_P * sqrt(sum(dt_i^2))
                # sigma_avg = sigma_total_energy / num_frames; sigma_min/max = sigma_P * time_per_frame
                time_per_frame = run_time_s / num_frames
                _stdev_src = (ref_stddevs or {}) if ref_averages else (stddevs or {})
                sigma_cpu = _stdev_src.get("cpu_power_w") or 0
                sigma_gpu = _stdev_src.get("gpu_power_w") or 0
                sigma_power = (sigma_cpu ** 2 + sigma_gpu ** 2) ** 0.5
                sigma_total = sigma_power * (sum_dt_sq ** 0.5) if sum_dt_sq > 0 else 0.0
                ef_min_std = sigma_power * time_per_frame if time_per_frame > 0 else None
                ef_max_std = sigma_power * time_per_frame if time_per_frame > 0 else None
                ef_avg_std = sigma_total / num_frames if num_frames > 0 else None

                meta["energy_per_frame_j"] = {
                    "min": round(energy_per_frame_min, 2),
                    "min_stddev": _round_val(ef_min_std),
                    "max": round(energy_per_frame_max, 2),
                    "max_stddev": _round_val(ef_max_std),
                    "avg": round(energy_per_frame_avg, 2),
                    "avg_stddev": _round_val(ef_avg_std),
                }
                if ref_energy_per_frame_j:
                    ref_ef = ref_energy_per_frame_j
                    def _adj_ef(raw, ref_val):
                        if raw is None or ref_val is None:
                            return None
                        return max(0.0, raw - ref_val)
                    meta["energy_per_frame_j_reference_adjusted"] = {
                        "min": _round_val(_adj_ef(energy_per_frame_min, ref_ef.get("min"))),
                        "max": _round_val(_adj_ef(energy_per_frame_max, ref_ef.get("max"))),
                        "avg": _round_val(_adj_ef(energy_per_frame_avg, ref_ef.get("avg"))),
                    }
                    # Propagated stddev for adjusted: sqrt(sigma_run^2 + sigma_ref^2)
                    ref_min_s = ref_ef.get("min_stddev") if isinstance(ref_ef.get("min_stddev"), (int, float)) else 0
                    ref_max_s = ref_ef.get("max_stddev") if isinstance(ref_ef.get("max_stddev"), (int, float)) else 0
                    ref_avg_s = ref_ef.get("avg_stddev") if isinstance(ref_ef.get("avg_stddev"), (int, float)) else 0
                    run_min_s = ef_min_std or 0
                    run_max_s = ef_max_std or 0
                    run_avg_s = ef_avg_std or 0
                    meta["energy_per_frame_j_reference_adjusted"]["min_stddev"] = _round_val((run_min_s ** 2 + ref_min_s ** 2) ** 0.5)
                    meta["energy_per_frame_j_reference_adjusted"]["max_stddev"] = _round_val((run_max_s ** 2 + ref_max_s ** 2) ** 0.5)
                    meta["energy_per_frame_j_reference_adjusted"]["avg_stddev"] = _round_val((run_avg_s ** 2 + ref_avg_s ** 2) ** 0.5)
        if ref_averages and averages:
            meta["averages_reference_adjusted"] = {}
            ref_std = ref_stddevs or {}
            run_std = stddevs or {}
            for k, v in averages.items():
                r = ref_averages.get(k)
                if r is not None and v is not None and k != "ram_available_gb":
                    adj_val = max(0.0, v - r)
                    meta["averages_reference_adjusted"][k] = _round_val(adj_val)
                    meta["averages_reference_adjusted"][k + "_stddev"] = _round_val(ref_std.get(k))
                else:
                    meta["averages_reference_adjusted"][k] = _round_val(v)
                    meta["averages_reference_adjusted"][k + "_stddev"] = _round_val(run_std.get(k))

        meta_path = self._run_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
