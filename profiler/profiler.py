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
    ):
        """
        Args:
            output_directory: Base directory for run outputs. A new subdir is created per run:
                {output_directory}/{YYYY}_{MM}_{DD}_{HHMM}_{title_with_underscores}/
                containing metadata.json, data.csv, plot.png
            frequency_hz: Sampling frequency in Hz (e.g. 1.0 = once per second)
            title: Title for the profiling run (spaces → underscores in run dir name), stored in metadata.
        """
        self.output_directory = Path(output_directory)
        self.frequency_hz = frequency_hz
        self.title = title
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
    
    def _sample_loop(self):
        """Background thread: collect metrics at the requested frequency."""
        interval = 1.0 / self.frequency_hz
        try:
            while self._running:
                loop_start = time.perf_counter()
                elapsed = loop_start - self._start_time

                # CPU power: record with pyJoules (RAPL) - energy between records backfilled after stop()
                if self._energy_meter is not None:
                    self._energy_meter.record(tag=f"t{len(self._samples)}")
                cpu_power = None  # Backfilled from pyJoules trace after stop()

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
                    cpu_power_w=cpu_power,
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
    
    def start(self):
        """Start recording metrics in a background thread. Raises ProfilerSetupError if setup fails."""
        if self._running:
            return
        self._samples.clear()
        self._sample_error = None
        self._start_time = time.perf_counter()
        self._start_wall_time = time.time()
        self._running = True

        # Validate packages and init GPU + RAPL; raises ProfilerSetupError on failure
        self._validate_and_init()

        self._thread = threading.Thread(target=self._sample_loop)
        self._thread.start()
    
    def stop(self, num_frames: Optional[int] = None):
        """
        Stop recording and write all output files.
        
        Args:
            num_frames: If provided, compute energy per frame (min, max, avg) from
                       total energy and store in metadata. Useful for ML/SLAM frame-based workloads.
        """
        if self._run_dir is not None:
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
        
        # Create run subdirectory: {output_dir}/{YYYY}_{MM}_{DD}_{HHMM}_{title_with_underscores}/
        dt = datetime.fromtimestamp(self._start_wall_time) if self._start_wall_time else datetime.now()
        sanitized_title = re.sub(r"[^\w\-]", "_", self.title).strip("_") or "run"
        run_dir_name = f"{dt:%Y_%m_%d_%H%M}_{sanitized_title}"
        self._run_dir = self.output_directory / run_dir_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        
        # Backfill CPU power from pyJoules trace (energy/duration per interval)
        # RAPL reports energy in microjoules (µJ); convert to J then power = J/s = W
        if self._energy_meter is not None and self._samples:
            trace = self._energy_meter.get_trace()
            if trace and trace._samples:
                for i, sample in enumerate(trace._samples):
                    if i < len(self._samples) and sample.duration > 0 and sample.energy:
                        total_uj = sum(sample.energy.values())  # microjoules
                        total_j = total_uj / 1e6  # convert to joules
                        self._samples[i].cpu_power_w = total_j / sample.duration
        
        run_time = time.perf_counter() - self._start_time if self._start_time else 0
        
        # Compute energy per frame stats if requested
        energy_per_frame_min = energy_per_frame_max = energy_per_frame_avg = None
        if num_frames is not None and num_frames > 0 and self._samples:
            total_energy_j = 0
            time_per_frame = run_time / num_frames
            for i in range(1, len(self._samples)):
                dt = self._samples[i].elapsed_s - self._samples[i - 1].elapsed_s
                if dt <= 0:
                    dt = 1.0 / self.frequency_hz
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

        # Write outputs
        self._write_csv()
        self._write_plot()
        self._write_metadata(
            run_time_s=run_time,
            num_frames=num_frames,
            energy_per_frame_min=energy_per_frame_min,
            energy_per_frame_max=energy_per_frame_max,
            energy_per_frame_avg=energy_per_frame_avg,
            metrics_available=metrics_available,
        )
    
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
    
    def _write_plot(self):
        """Generate matplotlib PNG plot of all metrics."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._samples:
            return
        
        times = [s.elapsed_s for s in self._samples]
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(self.title, fontsize=14)
        
        # CPU usage
        ax = axes[0, 0]
        ax.plot(times, [s.cpu_usage_percent for s in self._samples], "b-", label="CPU %")
        ax.set_ylabel("CPU usage (%)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # CPU power
        ax = axes[0, 1]
        cpu_power = [s.cpu_power_w if s.cpu_power_w is not None else float("nan") for s in self._samples]
        ax.plot(times, cpu_power, "g-", label="CPU power (W)")
        ax.set_ylabel("CPU power (W)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # GPU usage
        ax = axes[1, 0]
        gpu_usage = [s.gpu_usage_percent if s.gpu_usage_percent is not None else float("nan") for s in self._samples]
        ax.plot(times, gpu_usage, "orange", label="GPU %")
        ax.set_ylabel("GPU usage (%)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # GPU power
        ax = axes[1, 1]
        gpu_power = [s.gpu_power_w if s.gpu_power_w is not None else float("nan") for s in self._samples]
        ax.plot(times, gpu_power, "red", label="GPU power (W)")
        ax.set_ylabel("GPU power (W)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # GPU memory
        ax = axes[2, 0]
        gpu_mem = [s.gpu_memory_gb if s.gpu_memory_gb is not None else float("nan") for s in self._samples]
        ax.plot(times, gpu_mem, "purple", label="GPU memory (GB)")
        ax.set_ylabel("GPU memory (GB)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # RAM
        ax = axes[2, 1]
        ax.plot(times, [s.ram_usage_percent for s in self._samples], "brown", label="RAM %")
        ax.set_ylabel("RAM usage (%)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel("Time (s)" if ax.get_subplotspec().is_last_row() else "")
        
        plt.tight_layout()
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
        metrics_available: dict,
    ):
        """Write metadata JSON. All numeric values rounded to 2 decimal places."""
        meta = {
            "title": self.title,
            "run_time_s": round(run_time_s, 2),
            "num_samples": len(self._samples),
            "frequency_hz": round(self.frequency_hz, 2),
            "run_directory": str(self._run_dir),
            "metrics_available": metrics_available,
        }
        if num_frames is not None:
            meta["num_frames"] = num_frames
            if energy_per_frame_avg is not None:
                meta["energy_per_frame_j"] = {
                    "min": round(energy_per_frame_min, 2),
                    "max": round(energy_per_frame_max, 2),
                    "avg": round(energy_per_frame_avg, 2),
                }
        
        meta_path = self._run_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
