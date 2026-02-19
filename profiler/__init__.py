"""Profiler package for recording power, CPU, GPU, and memory metrics during testing."""

from .profiler import Profiler, ProfilerSetupError, Sample

__all__ = ["Profiler", "ProfilerSetupError", "Sample"]
