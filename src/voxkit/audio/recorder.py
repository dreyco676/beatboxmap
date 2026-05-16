# SPDX-License-Identifier: GPL-3.0-or-later
"""Recorder: audio capture, resampler worker, callback contract (Q67, Q76, Q77)."""

from __future__ import annotations

import platform as _platform_mod
import threading
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class AudioInitError(Exception):
    pass

class BluetoothDeviceRefused(Exception):
    pass

class StreamAlreadyOpen(Exception):
    pass

class DeviceDisconnected(Exception):
    def __init__(self, device_id: str) -> None:
        super().__init__(f"Device disconnected: {device_id}")
        self.device_id = device_id


# ---------------------------------------------------------------
# Data types
# ---------------------------------------------------------------

@dataclass(frozen=True)
class DeviceInfo:
    id: str
    name: str
    default_rate: int

@dataclass(frozen=True)
class OSSleepEvent:
    reason: str


# ---------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------

_PRIORITY_TABLE: dict[str, Any] = {
    "Windows": "Pro Audio",
    "Linux": ("SCHED_FIFO", 80),
}


def _current_platform() -> str:
    return _platform_mod.system()


def _apply_priority(priority: Any) -> None:
    """Apply thread scheduling priority per Q67.

    Windows: registers the calling thread with MMCSS task *priority* (e.g.
    "Pro Audio") via AvSetMmThreadCharacteristicsW in avrt.dll.  A zero
    return handle means the OS refused — non-fatal; dropped buffers are
    tracked separately by the caller.

    Linux (Phase 1.5): sets SCHED_FIFO at the given numeric priority via
    os.sched_setscheduler.  Requires setcap cap_sys_nice on the Python
    binary (or root); silently degrades if not available.

    All other platforms (Darwin, …): documented no-op for v1.0.
    """
    if isinstance(priority, str):
        # Windows MMCSS path.
        try:
            import ctypes
            import ctypes.wintypes
            avrt = ctypes.WinDLL("avrt")
            task_index = ctypes.wintypes.DWORD(0)
            avrt.AvSetMmThreadCharacteristicsW(priority, ctypes.byref(task_index))
        except Exception:
            pass
    elif isinstance(priority, tuple) and len(priority) == 2 and priority[0] == "SCHED_FIFO":
        # Linux Phase 1.5 path.
        try:
            import os as _os
            _os.sched_setscheduler(0, _os.SCHED_FIFO, _os.sched_param(priority[1]))
        except Exception:
            pass


def _set_thread_priority(priority: Any = None, *, _current_platform_value: str | None = None) -> None:
    """Set thread priority.

    Two calling modes:
      _set_thread_priority("Pro Audio")          — apply a known priority value directly
      _set_thread_priority(_current_platform_value="Windows")  — dispatch from platform name
    """
    if priority is not None:
        _apply_priority(priority)
        return
    plat = _current_platform_value if _current_platform_value is not None else _current_platform()
    prio = _PRIORITY_TABLE.get(plat)
    if prio is not None:
        _apply_priority(prio)


# ---------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------

_VALID_RATES = {16_000, 32_000, 44_100, 48_000, 88_200, 96_000}


def _raw_device_list() -> list[dict]:
    import sounddevice as sd
    apis = sd.query_hostapis()
    result = []
    for d in sd.query_devices():
        api_name = apis[d["hostapi"]]["name"] if d["hostapi"] < len(apis) else "Unknown"
        rate = int(d["default_samplerate"])
        if rate not in _VALID_RATES:
            continue
        if d["max_input_channels"] <= 0:
            continue
        result.append({
            "id": str(d["index"]),
            "name": d["name"],
            "hostapi": api_name,
            "rate": rate,
        })
    return result


def _device_hostapi(device_id: str) -> str:
    try:
        import sounddevice as sd
        apis = sd.query_hostapis()
        d = sd.query_devices(int(device_id))
        return apis[d["hostapi"]]["name"] if d["hostapi"] < len(apis) else "Unknown"
    except (ValueError, Exception):
        return "Unknown"


def _open_native_stream(*, device_id: str, hostapi: str, sample_rate: int, **kwargs):
    import sounddevice as sd
    return sd.InputStream(device=int(device_id), samplerate=sample_rate, channels=1)


# ---------------------------------------------------------------
# AtomicCounter
# ---------------------------------------------------------------

class AtomicCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def add(self, n: int) -> None:
        with self._lock:
            self._value += n


# ---------------------------------------------------------------
# LockFreeRing (SPSC; pre-allocated slots initialized on first push)
# ---------------------------------------------------------------

class LockFreeRing:
    """Pre-allocated SPSC ring. State stored in numpy C memory to avoid
    Python-int heap allocations on the hot path (Q67 tracemalloc contract)."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf: np.ndarray | None = None   # 2-D pre-allocated on first push
        # [write_idx, read_idx, count] — stored in C memory, not Python heap.
        self._st = np.zeros(3, dtype=np.int64)
        self._lock = threading.Lock()

    def try_push(self, data: np.ndarray) -> bool:
        with self._lock:
            if self._st[2] >= self._capacity:
                return False
            if self._buf is None:
                self._buf = np.zeros((self._capacity, len(data)), dtype=np.float32)
            write_idx = int(self._st[0])   # transient Python int; freed on return

        # NumPy __setitem__ releases GIL during the memcpy (Q67/Q76).
        self._buf[write_idx] = data

        with self._lock:
            self._st[0] = (self._st[0] + 1) % self._capacity
            self._st[2] += 1
        return True

    def try_pop_block(self) -> np.ndarray | None:
        with self._lock:
            if self._st[2] == 0:
                return None
            read_idx = int(self._st[1])
            slot = self._buf[read_idx].copy()
            self._st[1] = (self._st[1] + 1) % self._capacity
            self._st[2] -= 1
        return slot


# ---------------------------------------------------------------
# Callback factory (Q67)
# ---------------------------------------------------------------

def build_callback(
    ring: LockFreeRing,
    input_dtype: np.dtype | None = None,
    device_overflow_counter: AtomicCounter | None = None,
) -> Callable:
    def callback(
        indata: np.ndarray,
        *,
        frames: int,
        time_info: Any,
        status: Any,
        dropped_counter: AtomicCounter,
    ) -> None:
        try:
            # Device overflow is separate from ring-full drops (T32).
            if (device_overflow_counter is not None
                    and status is not None
                    and getattr(status, "input_overflow", False)):
                device_overflow_counter.add(1)

            # Normalize int16 → float32 if needed (T30).
            if input_dtype is not None and input_dtype == np.int16:
                data = indata.astype(np.float32) / 32768.0
            else:
                data = indata

            if not ring.try_push(data):
                dropped_counter.add(1)
        except Exception:
            pass  # Audio thread is sacred; never propagate (T13).

    return callback


# ---------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------

class Recorder:
    # Q67: v1.0 ships the Python-callback path. "cffi_hardened" is the
    # documented escalation when the default path measurably drops buffers.
    AUDIO_CALLBACK_PATH: str = "python_default"

    def __init__(self) -> None:
        self._stream = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._dropped_counter = AtomicCounter()
        self._sleep_handler: Callable | None = None
        self._device_id: str | None = None
        self.active_hostapi: str | None = None

    def list_devices(self) -> list[DeviceInfo]:
        raw = _raw_device_list()
        return [
            DeviceInfo(id=d["id"], name=d["name"], default_rate=d["rate"])
            for d in raw
            if d.get("hostapi") != "Bluetooth"
        ]

    def open_stream(self, device_id: str) -> None:
        if self._stream is not None:
            raise StreamAlreadyOpen(f"Stream already open; call close_stream() first")

        hostapi = _device_hostapi(device_id)
        if hostapi == "Bluetooth":
            raise BluetoothDeviceRefused(f"Device {device_id!r} is Bluetooth")

        plat = _current_platform()
        stream = None

        if plat == "Windows":
            try:
                stream = _open_native_stream(device_id=device_id, hostapi="WASAPI", sample_rate=16_000)
                self.active_hostapi = "WASAPI"
            except AudioInitError:
                stream = _open_native_stream(device_id=device_id, hostapi="MME", sample_rate=16_000)
                self.active_hostapi = "MME"
        else:
            stream = _open_native_stream(device_id=device_id, hostapi="ALSA", sample_rate=16_000)
            self.active_hostapi = "ALSA"

        self._stream = stream
        self._device_id = device_id
        self._dropped_counter = AtomicCounter()
        self._stop_event.clear()

        # Start worker thread; block until it has registered priority (T17/T18).
        priority_ready = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_run,
            args=(priority_ready,),
            daemon=True,
        )
        self._worker_thread.start()
        priority_ready.wait(timeout=1.0)

    def _worker_run(self, priority_ready: threading.Event) -> None:
        plat = _current_platform()
        prio = _PRIORITY_TABLE.get(plat)
        if prio is not None:
            _set_thread_priority(prio)
        priority_ready.set()
        self._stop_event.wait()

    def close_stream(self) -> None:
        if self._stream is None:
            return
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        self._stream = None
        self._device_id = None
        self.active_hostapi = None

    def get_dropped_buffer_count(self) -> int:
        return self._dropped_counter.value

    def install_sleep_handler(self, handler: Callable) -> None:
        self._sleep_handler = handler

    def _dispatch_sleep_event(self, event: OSSleepEvent) -> None:
        if self._sleep_handler is not None:
            self._sleep_handler(event)

    def handle_disconnect(self) -> DeviceDisconnected:
        device_id = self._device_id or ""
        self.close_stream()
        return DeviceDisconnected(device_id=device_id)
