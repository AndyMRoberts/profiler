"""
Diagnostic script to test GPU and RAPL (CPU power) access.
Run: python -m profiler.diagnose
"""

import sys


def check_gpu():
    """Test NVIDIA GPU access via nvidia-ml-py."""
    print("1. NVIDIA GPU (nvidia-ml-py)")
    try:
        import pynvml
        print("   pynvml: imported OK")
    except ImportError as e:
        print(f"   FAIL: {e}")
        print("   Install: pip install nvidia-ml-py")
        return

    try:
        pynvml.nvmlInit()
        print("   nvmlInit: OK")
    except Exception as e:
        print(f"   FAIL nvmlInit: {e}")
        print("   Ensure NVIDIA drivers installed and nvidia-smi works")
        return

    try:
        count = pynvml.nvmlDeviceGetCount()
        print(f"   GPU count: {count}")
        if count == 0:
            print("   No GPUs found")
            pynvml.nvmlShutdown()
            return
    except Exception as e:
        print(f"   FAIL GetCount: {e}")
        pynvml.nvmlShutdown()
        return

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    try:
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"   GPU 0: {name.decode() if isinstance(name, bytes) else name}")
    except Exception as e:
        print(f"   GetName: {e}")

    for attr, fn, desc in [
        ("Power", lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, "nvmlDeviceGetPowerUsage"),
        ("Utilization", lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu, "nvmlDeviceGetUtilizationRates"),
        ("Memory", lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3), "nvmlDeviceGetMemoryInfo"),
    ]:
        try:
            val = fn()
            print(f"   {attr}: {val:.2f} OK ({desc})")
        except Exception as e:
            print(f"   {attr}: FAIL - {e}")

    pynvml.nvmlShutdown()


def check_rapl():
    """Test Intel RAPL (CPU power) via pyJoules."""
    print("\n2. CPU power (pyJoules / Intel RAPL)")
    try:
        from pyJoules.device.rapl_device import RaplPackageDomain
        from pyJoules.device import DeviceFactory
        from pyJoules.energy_meter import EnergyMeter
        print("   pyJoules: imported OK")
    except ImportError as e:
        print(f"   FAIL: {e}")
        print("   Install: pip install pyJoules")
        return

    if sys.platform != "linux":
        print("   RAPL is Linux-only")
        return

    try:
        devices = DeviceFactory.create_devices([RaplPackageDomain(0)])
        print(f"   RAPL devices: {len(devices)} OK")
    except Exception as e:
        print(f"   FAIL create_devices: {e}")
        print("   Try: sudo chmod -R a+r /sys/class/powercap/intel-rapl")
        return

    try:
        meter = EnergyMeter(devices)
        meter.start()
        meter.record(tag="test")
        meter.stop()
        trace = meter.get_trace()
        if trace and trace._samples:
            s = trace._samples[0]
            total_j = sum(s.energy.values()) if s.energy else 0
            print(f"   First sample energy: {total_j:.2f} J OK")
        else:
            print("   WARNING: No trace samples")
    except Exception as e:
        print(f"   FAIL EnergyMeter read: {e}")
        print("   RAPL may need: sudo chmod -R a+r /sys/class/powercap/intel-rapl")


def main():
    print("=== Profiler diagnostics ===\n")
    check_gpu()
    check_rapl()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
