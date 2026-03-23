#!/home/patrick/miniconda3/envs/fastsam/bin/python3
"""Compatibility entrypoint for legacy CLIP launch files."""

# Preload system libffi to solve symbol conflicts with PyTorch and system libraries (e.g. p11-kit) on some setups.
import ctypes
import os


def _preload_system_libffi():
    """Prefer system libffi.so.7 to satisfy system libp11-kit ABI requirements."""
    libffi_path = "/usr/lib/x86_64-linux-gnu/libffi.so.7"
    if os.path.exists(libffi_path):
        try:
            ctypes.CDLL(libffi_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Keep startup resilient; downstream import may still work on other setups.
            pass


_preload_system_libffi()

from clip_ros_node import main


if __name__ == "__main__":
    main()
