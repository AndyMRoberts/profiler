"""Unit tests for the profiler package."""

import csv
import json
import tempfile
import time
from pathlib import Path

import pytest

from profiler import Profiler, ProfilerSetupError, Sample


def _parse_csv(csv_path):
    """Parse profiler CSV and return (headers, data_rows)."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows


def _require_full_setup():
    """Skip if GPU/RAPL setup fails (e.g. in CI without hardware)."""
    try:
        Profiler.verify_setup()
    except ProfilerSetupError as e:
        pytest.skip(str(e))


class TestSample:
    """Tests for the Sample dataclass."""

    def test_sample_creation(self):
        """Sample can be created with required fields."""
        s = Sample(
            timestamp=0.0,
            elapsed_s=0.0,
            cpu_usage_percent=50.0,
            cpu_power_w=None,
            gpu_power_w=None,
            gpu_usage_percent=None,
            gpu_memory_gb=None,
            gpu_memory_percent=None,
            ram_usage_percent=60.0,
            ram_used_gb=8.0,
            ram_available_gb=5.0,
        )
        assert s.cpu_usage_percent == 50.0
        assert s.ram_used_gb == 8.0


class TestProfilerInit:
    """Tests for Profiler initialization."""

    def test_init_basic(self):
        """Profiler initializes with output directory, frequency, title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=2.0, title="Test run")
            assert p.output_directory == Path(tmpdir)
            assert p.frequency_hz == 2.0
            assert p.title == "Test run"


class TestProfilerStartStop:
    """Tests for Profiler start/stop and output generation."""

    def test_start_raises_when_setup_fails(self):
        """start() raises ProfilerSetupError immediately when packages or hardware unavailable."""
        try:
            Profiler.verify_setup()
        except ProfilerSetupError:
            # Setup fails - create profiler and verify start() raises
            with tempfile.TemporaryDirectory() as tmpdir:
                p = Profiler(tmpdir, gpu_memory_total_gb=16.376)
                with pytest.raises(ProfilerSetupError):
                    p.start()
            return
        # If we get here, setup works - nothing to test
        pytest.skip("Full setup available, cannot test failure path")

    def test_verify_setup(self):
        """verify_setup() raises ProfilerSetupError with clear message when setup fails."""
        try:
            Profiler.verify_setup()
        except ProfilerSetupError as e:
            assert "pip install" in str(e).lower() or "nvidia" in str(e).lower() or "pyjou" in str(e).lower()
            return
        # Setup works
        pass

    def test_start_stop_produces_csv_and_metadata(self):
        """Start then stop produces CSV and metadata files in run subdirectory."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0, title="Unit test")
            p.start()
            time.sleep(0.8)  # Allow a few samples at 4 Hz
            p.stop()
            assert (p._run_dir / "data.csv").exists()
            assert (p._run_dir / "metadata.json").exists()

    def test_start_stop_produces_png(self):
        """Start then stop produces PNG plot in run subdirectory."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.start()
            time.sleep(0.6)
            p.stop()
            assert (p._run_dir / "plot.png").exists()

    def test_csv_has_expected_columns(self):
        """CSV contains expected column headers."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.start()
            time.sleep(0.5)
            p.stop()
            with open(p._run_dir / "data.csv") as f:
                header = f.readline().strip()
            expected = [
                "time_s", "cpu_usage_percent", "cpu_power_w",
                "gpu_power_w", "gpu_usage_percent", "gpu_memory_gb",
                "gpu_memory_percent", "ram_usage_percent", "ram_used_gb", "ram_available_gb",
            ]
            for col in expected:
                assert col in header

    def test_metadata_has_required_fields(self):
        """Metadata JSON contains title, run_time_s, num_samples, frequency_hz."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0, title="My test")
            p.start()
            time.sleep(0.5)
            p.stop()
            with open(p._run_dir / "metadata.json") as f:
                meta = json.load(f)
            assert meta["title"] == "My test"
            assert "run_time_s" in meta
            assert "num_samples" in meta
            assert meta["frequency_hz"] == 4.0

    def test_stop_with_num_frames_adds_energy_per_frame(self):
        """stop(num_frames=N) adds energy_per_frame_j to metadata."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.start()
            time.sleep(0.6)
            p.stop(num_frames=10)
            with open(p._run_dir / "metadata.json") as f:
                meta = json.load(f)
            assert meta["num_frames"] == 10
            assert "energy_per_frame_j" in meta
            epf = meta["energy_per_frame_j"]
            assert "min" in epf and "max" in epf and "avg" in epf

    def test_energy_per_frame_not_zero_when_power_available(self):
        """energy_per_frame_j values must not all be 0 when num_frames given - indicates power data failure."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.start()
            time.sleep(0.8)
            p.stop(num_frames=10)
            with open(p._run_dir / "metadata.json") as f:
                meta = json.load(f)
            epf = meta.get("energy_per_frame_j")
            assert epf is not None, "energy_per_frame_j missing"
            min_e, max_e, avg_e = epf["min"], epf["max"], epf["avg"]
            assert not (min_e == 0 and max_e == 0 and avg_e == 0), (
                "energy_per_frame_j min/max/avg are all 0 - CPU/GPU power not recorded. "
                "Check RAPL and NVML permissions."
            )

    def test_stop_without_num_frames_no_energy_per_frame(self):
        """stop() without num_frames does not add energy_per_frame_j."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376)
            p.start()
            time.sleep(0.3)
            p.stop()
            with open(p._run_dir / "metadata.json") as f:
                meta = json.load(f)
            assert "energy_per_frame_j" not in meta
            assert "num_frames" not in meta

    def test_double_stop_idempotent(self):
        """Calling stop() twice does not raise."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376)
            p.start()
            time.sleep(0.3)
            p.stop()
            p.stop()  # Second stop should not raise

    def test_start_twice_then_stop(self):
        """Calling start() twice (without stop) is effectively a no-op for second start."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376)
            p.start()
            p.start()  # Second start should not raise
            time.sleep(0.3)
            p.stop()

    def test_run_directory_name_format(self):
        """Run directory is named {YYYY}_{MM}_{DD}_{HHMM}_{title_with_underscores}."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, title="SLAM inference test")
            p.start()
            time.sleep(0.3)
            p.stop()
            # e.g. 2026_02_19_1430_SLAM_inference_test
            assert p._run_dir.parent == Path(tmpdir)
            assert "_SLAM_inference_test" in p._run_dir.name
            assert p._run_dir.name.count("_") >= 4  # date + time + title parts


class TestProfilerDataQuality:
    """Tests that verify recorded data quality - catch empty cpu_power, gpu metrics, etc."""

    def test_csv_required_metrics_always_populated(self):
        """cpu_usage_percent and ram_* must have values in every data row (psutil always works)."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.start()
            time.sleep(0.8)
            p.stop()
            _, rows = _parse_csv(p._run_dir / "data.csv")
            assert len(rows) >= 2, "Need multiple samples to verify data"
            required = ["cpu_usage_percent", "ram_usage_percent", "ram_used_gb", "ram_available_gb"]
            for row in rows:
                for col in required:
                    assert row[col].strip(), f"Required column '{col}' empty in row {row}"
                    float(row[col])  # Must be parseable as number

    def test_csv_gpu_metrics_populated(self):
        """GPU columns must have data. Fails if no GPU, NVML permissions, or nvidia-ml-py not working."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.start()
            time.sleep(0.8)
            p.stop()
            _, rows = _parse_csv(p._run_dir / "data.csv")
            assert len(rows) >= 2
            # At least one of gpu_power_w, gpu_usage_percent, gpu_memory_gb should have data
            has_gpu_data = False
            for row in rows:
                if row.get("gpu_power_w", "").strip():
                    has_gpu_data = True
                    break
                if row.get("gpu_usage_percent", "").strip():
                    has_gpu_data = True
                    break
                if row.get("gpu_memory_gb", "").strip():
                    has_gpu_data = True
                    break
            assert has_gpu_data, (
                "gpu_power_w, gpu_usage_percent, gpu_memory_gb are all empty. "
                "Ensure NVIDIA GPU, nvidia-ml-py, and NVML permissions (e.g. video group)."
            )

    def test_csv_cpu_power_populated(self):
        """cpu_power_w must have data. Fails if RAPL not accessible (permissions) or pyJoules not working."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.start()
            time.sleep(0.8)
            p.stop()
            _, rows = _parse_csv(p._run_dir / "data.csv")
            assert len(rows) >= 2
            has_cpu_power = any(row.get("cpu_power_w", "").strip() for row in rows)
            assert has_cpu_power, (
                "cpu_power_w is empty. Ensure pyJoules can read /sys/class/powercap/ "
                "(often: sudo chmod -R a+r /sys/class/powercap/intel-rapl)."
            )

    def test_metadata_contains_metrics_available(self):
        """Metadata includes metrics_available for diagnostics."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376)
            p.start()
            time.sleep(0.5)
            p.stop()
            with open(p._run_dir / "metadata.json") as f:
                meta = json.load(f)
            assert "metrics_available" in meta
            ma = meta["metrics_available"]
            assert "cpu_power" in ma and "gpu_power" in ma and "gpu_usage" in ma and "gpu_memory" in ma

    def test_metadata_averages_include_stddev(self):
        """Metadata averages block includes key_stddev for each metric (run stdev when no reference)."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.start()
            time.sleep(0.8)  # enough samples for non-zero stddev possibility
            p.stop()
            with open(p._run_dir / "metadata.json") as f:
                meta = json.load(f)
            assert "averages" in meta, "averages block missing"
            av = meta["averages"]
            expected_keys = [
                "cpu_usage_percent", "cpu_power_w", "gpu_power_w", "gpu_usage_percent",
                "gpu_memory_gb", "gpu_memory_percent", "ram_usage_percent",
                "ram_used_gb", "ram_available_gb",
            ]
            for k in expected_keys:
                assert k in av, f"averages missing key {k}"
                stddev_key = k + "_stddev"
                assert stddev_key in av, f"averages missing {stddev_key}"
                val = av[stddev_key]
                assert val is None or isinstance(val, (int, float)), f"{stddev_key} must be number or null"
                if val is not None:
                    assert val >= 0, f"{stddev_key} must be non-negative"

    def test_reference_metadata_include_stddev(self):
        """Reference run metadata includes averages with _stddev for each metric."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0, title="Ref test")
            p.record_reference(2)  # 2 seconds of reference
            ref_meta_path = Path(tmpdir) / "reference" / "metadata.json"
            assert ref_meta_path.exists(), "reference metadata.json missing"
            with open(ref_meta_path) as f:
                meta = json.load(f)
            assert "averages" in meta
            av = meta["averages"]
            for k in ["cpu_usage_percent", "ram_usage_percent", "ram_used_gb", "ram_available_gb"]:
                assert k in av
                assert (k + "_stddev") in av
                std = av[k + "_stddev"]
                assert std is None or (isinstance(std, (int, float)) and std >= 0)

    def test_reference_adjusted_run_metadata_include_stddev(self):
        """When use_reference=True, averages_reference_adjusted includes _stddev (from reference)."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0, title="Adjusted test")
            p.record_reference(2)
            p.start(use_reference=True)
            time.sleep(0.6)
            p.stop()
            with open(p._run_dir / "metadata.json") as f:
                meta = json.load(f)
            assert "averages_reference_adjusted" in meta
            adj = meta["averages_reference_adjusted"]
            for k in ["cpu_usage_percent", "ram_usage_percent"]:
                assert k in adj
                stddev_key = k + "_stddev"
                assert stddev_key in adj, f"averages_reference_adjusted missing {stddev_key}"
                val = adj[stddev_key]
                assert val is None or (isinstance(val, (int, float)) and val >= 0)


class TestRefMetadataCopy:
    """Tests that ref_metadata.json is created/versioned correctly."""

    def test_stop_copies_ref_metadata_to_run_dir(self):
        """stop() copies reference metadata.json to run_dir/ref_metadata.json when reference exists."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0, title="Ref copy test")
            p.record_reference(2)
            p.start()
            time.sleep(0.4)
            p.stop()
            assert (p._run_dir / "ref_metadata.json").exists(), "ref_metadata.json not created in run dir"
            with open(p._run_dir / "ref_metadata.json") as f:
                ref_meta = json.load(f)
            assert "averages" in ref_meta

    def test_record_reference_does_not_create_ref_metadata(self):
        """record_reference() itself must NOT create a ref_metadata.json in the reference dir."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.record_reference(2)
            ref_dir = Path(tmpdir) / "reference"
            assert not (ref_dir / "ref_metadata.json").exists(), (
                "ref_metadata.json should not be created inside the reference directory itself"
            )

    def test_stop_no_ref_metadata_when_no_reference_folder(self):
        """stop() does NOT create ref_metadata.json when no reference folder exists."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0)
            p.start()
            time.sleep(0.4)
            p.stop()
            assert not (p._run_dir / "ref_metadata.json").exists(), (
                "ref_metadata.json should not be created when no reference folder exists"
            )

    def test_reprocess_data_creates_ref_metadata(self):
        """reprocess_data() copies reference metadata.json to target_dir/ref_metadata.json."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0, title="Reprocess test")
            p.record_reference(2)
            p.start()
            time.sleep(0.4)
            p.stop()
            run_dir = p._run_dir
            # Remove the ref_metadata.json created by stop() so we can test reprocess independently
            ref_meta_in_run = run_dir / "ref_metadata.json"
            if ref_meta_in_run.exists():
                ref_meta_in_run.unlink()
            p.reprocess_data(run_dir)
            assert (run_dir / "ref_metadata.json").exists(), "ref_metadata.json not created by reprocess_data"

    def test_reprocess_data_versions_ref_metadata(self):
        """reprocess_data() adds versioned ref_metadata_1.json, _2.json when called multiple times."""
        _require_full_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Profiler(tmpdir, gpu_memory_total_gb=16.376, frequency_hz=4.0, title="Version test")
            p.record_reference(2)
            p.start()
            time.sleep(0.4)
            p.stop()
            run_dir = p._run_dir
            # ref_metadata.json should already exist from stop()
            assert (run_dir / "ref_metadata.json").exists()
            # First reprocess: should create ref_metadata_1.json
            p.reprocess_data(run_dir)
            assert (run_dir / "ref_metadata_1.json").exists(), "ref_metadata_1.json not created on second copy"
            # Second reprocess: should create ref_metadata_2.json
            p.reprocess_data(run_dir)
            assert (run_dir / "ref_metadata_2.json").exists(), "ref_metadata_2.json not created on third copy"
