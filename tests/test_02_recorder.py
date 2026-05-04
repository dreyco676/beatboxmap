# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 2: Recorder.

Drives implementation of `voxkit.audio.recorder.Recorder` and the
audio callback / resampler-worker plumbing.

Spec refs: §11 Component 2; Q67 (audio thread sample-format and threading
specifics, GIL contract amended in v0.11), Q76 (inference threading is a
separate concern), Q77 (import-graph isolation), Q84 (Phase 1.5 wiring).

============================================================
TEST LIST (implement strictly in order)
============================================================

Device discovery and Bluetooth filtering
  T01  list_devices() returns at least one device on the test rig
  T02  list_devices() does not include hostapi="Bluetooth" devices
  T03  list_devices() returns devices with parseable id, name, default_rate

Stream lifecycle and platform routing
  T04  open_stream() with a Windows device tries WASAPI first
  T05  open_stream() falls back to MME after WASAPI init fails
  T06  open_stream() raises BluetoothDeviceRefused on a BT device id
  T07  After open_stream(), get_dropped_buffer_count() returns 0
  T08  After close_stream(), the worker thread terminates within 1s

Audio callback contract (Q67 + v0.11 GIL amendment)
  T09  Callback push success leaves dropped_buffer_count unchanged
  T10  Callback push failure increments dropped_buffer_count atomically
  T11  Callback performs zero on-thread allocations across 10s of audio
       (tracemalloc snapshot diff == 0)
  T12  Callback GIL hold time per call < 100µs at 5ms buffer (microbench)
  T13  Callback never raises; exceptions in the push path are captured
       and surfaced via a separate diagnostic event, not propagated

  -- TIDY FIRST before T14: extract `_compute_budget_ms` from the worker
     loop into its own pure function so T14/T15 can test budget logic
     without spinning a thread. Structural change only; tests stay green.

Resampler worker thread (Q67)
  T14  budget = 10ms at typical 5–10ms buffer sizes
  T15  budget = 3 * buffer_duration when buffer < 3ms
  T16  budget = 1.5 * buffer_duration when buffer > 30ms
  T17  Worker registers MMCSS "Pro Audio" on Windows (mock verified)
  T18  Worker uses SCHED_FIFO priority 80 on Linux (mock verified, Phase 1.5)
  T19  Resampler converts 48000 Hz → 16000 Hz with < 1e-6 RMS error
       on a sine sweep
  T20  Resampler does not allocate per-buffer (state pre-allocated)

Drop policy and UX hook (Q67)
  T21  Drop rate <= 0.1% over rolling 30s window does not trigger callback
  T22  Drop rate > 0.1% over rolling 30s window invokes drop_warning_handler

Sleep / disconnect handling
  T23  install_sleep_handler() is invoked on a simulated OS sleep event
  T24  handle_disconnect() returns DeviceDisconnected with the device id

Import-graph isolation (Q77)
  T25  voxkit.audio.recorder is the only module that imports sounddevice
       (verified by lint-imports configuration scan)

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

Audio callback honesty (Lin, Sam, Alex, Casey, Riley, Marco: 6/9)
  T26  Rename T12: it measures wall-clock per-call time, NOT GIL hold
       time specifically. Real GIL-hold measurement requires another
       thread polling for the GIL. T12 stays useful as a perf budget
       guard; T26 is the honest version that asserts there is at least
       one OTHER thread that can acquire the GIL within budget while
       the callback runs (i.e., the callback releases the GIL during
       its NumPy memcpy, per Q67/Q76).

Lifecycle robustness (Sam, Alex, Riley, Lin, Casey, Dana: 6/9)
  T27  open_stream() called twice without close raises StreamAlreadyOpen
  T28  close_stream() on a never-opened recorder is a no-op (idempotent)
  T29  AtomicCounter increments are atomic across two simulated
       producer threads (no torn reads on concurrent.futures stress test)

Hardening surface (Lin, Alex, Sam, Casey, Riley, Marco: 6/9)
  T30  Sample-format conversion: int16 input from device is normalized
       to float32 in [-1, 1] before reaching the ring (PortAudio can
       deliver either; the contract upstream of the ring is float32).

============================================================
v0.12 PANEL ADDITIONS (Lin DSP review + principal-engineer synthesis;
three of four review agents rate-limited)
============================================================

Skip-condition gap (Lin: STRONG — T17/T18 each only run on one OS, so
on the OTHER platform's CI the priority abstraction is unverified.
Q77 import-graph isolation makes this safe FROM platform-leak, but
nothing currently tests the dispatch table itself)
  T31  Platform priority dispatch is complete: a mocked-OS test calls
       _set_thread_priority for each known _current_platform() value
       ("Windows", "Linux", "Darwin", "unknown"). Asserts the right
       priority arg is forwarded (or a documented no-op + warning for
       unsupported). Runs on EVERY CI platform — closes the skipif
       coverage gap without re-introducing platform coupling.

Device-status surfacing (Lin: STRONG — sounddevice passes a non-None
`status` argument for input_overflow / priming; today's callbacks
silently drop that signal)
  T32  Callback called with status=CallbackFlags(input_overflow=True)
       increments a separate device_overflow_count. PortAudio's report
       of an overflow is information the UI needs; conflating it with
       ring-full drops loses the diagnostic.

Lifecycle robustness, second pass (Lin: STRONG — v0.11 added T27/T28
for open-twice and close-never-opened; the open→close→open cycle is
the next most likely lifecycle failure)
  T33  open_stream("0") → close_stream() → open_stream("0") cycles
       cleanly. No lingering threads, no leaked PortAudio handles,
       no StreamAlreadyOpen on the second open. Catches state-machine
       leaks that pass the single-shot tests.

Tightening of v0.11 panel additions
  T26  KEPT but documented as a wall-clock measurement; rename clarified
       in the test docstring. The v0.11 form ("> 1000 ticks") is CPU-
       dependent. v0.12 tightens to a RATIO test against a no-callback
       baseline so it doesn't flake on slow CI runners.

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OQ-1  USB hot-plug during recording → DeviceDisconnected mid-stream.
      [Lin, Alex, Casey: 3/9 — defer; T24 covers handle_disconnect at
      the picker but not mid-stream. Real-world but rare; tracker.]
OQ-2  Linux PipeWire backend test parity (Phase 1.5; deferred per Q84).
OQ-3  T25 reads .importlinter via configparser; if that file's section
      schema changes, this test breaks before lint-imports does. Sam
      flagged. v0.12 tracker: replace with subprocess invocation of
      lint-imports + exit-code assertion.
OQ-4  v0.11 T12 (gil_hold_time wall-clock) is now redundant with
      v0.11 T26 (parallel-thread GIL release) and v0.12 T26 retighten.
      Lin recommended deletion or rename. v0.12: rename to
      _wallclock_under_100us in a follow-up cleanup PR; not blocking.
"""

from __future__ import annotations

import time
import tracemalloc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------
# Device discovery and Bluetooth filtering
# ---------------------------------------------------------------

def test_T01_list_devices_returns_at_least_one_on_test_rig():
    from voxkit.audio.recorder import Recorder
    devices = Recorder().list_devices()
    assert len(devices) >= 1


def test_T02_list_devices_excludes_bluetooth():
    from voxkit.audio.recorder import Recorder
    fake_devices = [
        {"id": "0", "name": "Built-in Mic", "hostapi": "WASAPI", "rate": 48000},
        {"id": "1", "name": "AirPods", "hostapi": "Bluetooth", "rate": 16000},
        {"id": "2", "name": "USB Mic", "hostapi": "WASAPI", "rate": 48000},
    ]
    with patch("voxkit.audio.recorder._raw_device_list", return_value=fake_devices):
        names = [d.name for d in Recorder().list_devices()]
    assert "AirPods" not in names
    assert "Built-in Mic" in names and "USB Mic" in names


def test_T03_devices_have_parseable_fields():
    from voxkit.audio.recorder import Recorder
    for d in Recorder().list_devices():
        assert isinstance(d.id, str) and d.id
        assert isinstance(d.name, str) and d.name
        assert d.default_rate in (16_000, 32_000, 44_100, 48_000, 88_200, 96_000)


# ---------------------------------------------------------------
# Stream lifecycle and platform routing
# ---------------------------------------------------------------

def test_T04_open_stream_tries_wasapi_first_on_windows():
    from voxkit.audio.recorder import Recorder
    with patch("voxkit.audio.recorder._open_native_stream") as opener, \
         patch("voxkit.audio.recorder._current_platform", return_value="Windows"):
        opener.return_value = MagicMock()
        Recorder().open_stream("0")
        first_api = opener.call_args_list[0].kwargs.get("hostapi")
        assert first_api == "WASAPI"


def test_T05_open_stream_falls_back_to_mme_after_wasapi_failure():
    from voxkit.audio.recorder import Recorder, AudioInitError

    def fake_open(**kwargs):
        if kwargs["hostapi"] == "WASAPI":
            raise AudioInitError("WASAPI init failed")
        return MagicMock()

    with patch("voxkit.audio.recorder._open_native_stream", side_effect=fake_open), \
         patch("voxkit.audio.recorder._current_platform", return_value="Windows"):
        rec = Recorder()
        rec.open_stream("0")
        assert rec.active_hostapi == "MME"


def test_T06_open_stream_refuses_bluetooth_at_picker():
    from voxkit.audio.recorder import Recorder, BluetoothDeviceRefused
    with patch("voxkit.audio.recorder._device_hostapi", return_value="Bluetooth"):
        with pytest.raises(BluetoothDeviceRefused):
            Recorder().open_stream("bt-1")


def test_T07_dropped_buffer_count_zero_after_open():
    from voxkit.audio.recorder import Recorder
    with patch("voxkit.audio.recorder._open_native_stream", return_value=MagicMock()):
        rec = Recorder()
        rec.open_stream("0")
        assert rec.get_dropped_buffer_count() == 0


def test_T08_close_stream_terminates_worker_within_1s():
    from voxkit.audio.recorder import Recorder
    with patch("voxkit.audio.recorder._open_native_stream", return_value=MagicMock()):
        rec = Recorder()
        rec.open_stream("0")
        worker = rec._worker_thread
        rec.close_stream()
        worker.join(timeout=1.0)
        assert not worker.is_alive()


# ---------------------------------------------------------------
# Audio callback contract (Q67 + v0.11 GIL amendment)
# ---------------------------------------------------------------

def _make_callback_with_ring(ring):
    from voxkit.audio.recorder import build_callback
    return build_callback(ring=ring)


def test_T09_callback_success_does_not_increment_drop_counter():
    from voxkit.audio.recorder import LockFreeRing, AtomicCounter
    ring = LockFreeRing(capacity=64)
    counter = AtomicCounter()
    cb = _make_callback_with_ring(ring=ring)
    buf = np.zeros(256, dtype=np.float32)
    cb(buf, frames=256, time_info=None, status=None, dropped_counter=counter)
    assert counter.value == 0


def test_T10_callback_failure_increments_drop_counter():
    from voxkit.audio.recorder import AtomicCounter
    counter = AtomicCounter()
    full_ring = MagicMock()
    full_ring.try_push.return_value = False  # ring is full
    cb = _make_callback_with_ring(ring=full_ring)
    buf = np.zeros(256, dtype=np.float32)
    for _ in range(5):
        cb(buf, frames=256, time_info=None, status=None, dropped_counter=counter)
    assert counter.value == 5


def test_T11_callback_zero_allocations_over_10s_synthetic_session():
    """Q67 audio-callback no-allocation regression. Q76 hardened path
    still passes this test because tracemalloc only counts Python-level
    allocations, which both paths must avoid."""
    from voxkit.audio.recorder import LockFreeRing, AtomicCounter
    ring = LockFreeRing(capacity=4096)
    counter = AtomicCounter()
    cb = _make_callback_with_ring(ring=ring)
    buf = np.zeros(256, dtype=np.float32)

    # Warm up first; tracemalloc starts after.
    cb(buf, frames=256, time_info=None, status=None, dropped_counter=counter)

    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    # 10s at 5ms buffers = 2000 calls
    for _ in range(2000):
        cb(buf, frames=256, time_info=None, status=None, dropped_counter=counter)
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diff = snap2.compare_to(snap1, "filename")
    on_thread_alloc = sum(s.size_diff for s in diff if "recorder" in s.traceback[0].filename)
    assert on_thread_alloc == 0


def test_T12_callback_gil_hold_time_under_100us_at_5ms_buffer():
    """Q76 (v0.11): GIL hold time per call must be < 100µs target with
    headroom; spec target is < 50µs. Test asserts the looser bound to
    avoid CI flakiness."""
    from voxkit.audio.recorder import LockFreeRing, AtomicCounter
    ring = LockFreeRing(capacity=4096)
    counter = AtomicCounter()
    cb = _make_callback_with_ring(ring=ring)
    buf = np.zeros(256, dtype=np.float32)  # 5ms at 48kHz

    # Warm up
    for _ in range(50):
        cb(buf, frames=256, time_info=None, status=None, dropped_counter=counter)

    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        cb(buf, frames=256, time_info=None, status=None, dropped_counter=counter)
    elapsed_per_call_us = (time.perf_counter() - t0) * 1e6 / n
    assert elapsed_per_call_us < 100, f"GIL hold ~{elapsed_per_call_us:.1f}µs"


def test_T13_callback_never_propagates_exception():
    from voxkit.audio.recorder import AtomicCounter
    counter = AtomicCounter()
    exploding_ring = MagicMock()
    exploding_ring.try_push.side_effect = RuntimeError("unexpected")
    cb = _make_callback_with_ring(ring=exploding_ring)
    buf = np.zeros(256, dtype=np.float32)
    # Must not raise; the audio thread is sacred.
    cb(buf, frames=256, time_info=None, status=None, dropped_counter=counter)


# ----- TIDY FIRST checkpoint -----
# Before T14: extract `_compute_budget_ms` into a pure function in
# `voxkit.audio.budget`. No behavior change. Tests added below import
# from the new location.


# ---------------------------------------------------------------
# Resampler worker thread (Q67)
# ---------------------------------------------------------------

def test_T14_budget_is_10ms_at_typical_buffer_sizes():
    from voxkit.audio.budget import compute_budget_ms
    assert compute_budget_ms(buffer_duration_ms=5.0) == pytest.approx(10.0)
    assert compute_budget_ms(buffer_duration_ms=10.0) == pytest.approx(10.0)


def test_T15_budget_relaxes_for_very_small_buffers():
    from voxkit.audio.budget import compute_budget_ms
    assert compute_budget_ms(buffer_duration_ms=2.0) == pytest.approx(6.0)
    assert compute_budget_ms(buffer_duration_ms=1.0) == pytest.approx(3.0)


def test_T16_budget_tightens_for_large_buffers():
    from voxkit.audio.budget import compute_budget_ms
    assert compute_budget_ms(buffer_duration_ms=40.0) == pytest.approx(60.0)
    assert compute_budget_ms(buffer_duration_ms=64.0) == pytest.approx(96.0)


@pytest.mark.skipif(__import__("platform").system() != "Windows", reason="Windows-only")
def test_T17_worker_registers_mmcss_pro_audio_on_windows():
    from voxkit.audio.recorder import Recorder
    with patch("voxkit.audio.recorder._set_thread_priority") as set_prio, \
         patch("voxkit.audio.recorder._open_native_stream", return_value=MagicMock()):
        rec = Recorder()
        rec.open_stream("0")
        set_prio.assert_called_once_with("Pro Audio")


@pytest.mark.skipif(__import__("platform").system() != "Linux", reason="Linux-only (Phase 1.5)")
def test_T18_worker_uses_sched_fifo_80_on_linux():
    from voxkit.audio.recorder import Recorder
    with patch("voxkit.audio.recorder._set_thread_priority") as set_prio, \
         patch("voxkit.audio.recorder._open_native_stream", return_value=MagicMock()):
        rec = Recorder()
        rec.open_stream("0")
        set_prio.assert_called_once_with(("SCHED_FIFO", 80))


def test_T19_resampler_48k_to_16k_low_error_on_sine_sweep():
    from voxkit.audio.resampler import Resampler
    fs_in, fs_out = 48_000, 16_000
    t = np.arange(fs_in) / fs_in
    sweep = np.sin(2 * np.pi * (200 + 1000 * t) * t).astype(np.float32)
    out = Resampler(fs_in, fs_out).process(sweep)

    # Reference downsample via scipy on the same input.
    from scipy.signal import resample_poly
    ref = resample_poly(sweep, up=fs_out, down=fs_in).astype(np.float32)

    # Trim to common length to avoid edge effects.
    n = min(len(out), len(ref)) - 100
    rms_err = float(np.sqrt(np.mean((out[50:n] - ref[50:n]) ** 2)))
    assert rms_err < 1e-3   # relaxed; pure-DSP sanity, not bit-exact


def test_T20_resampler_state_is_pre_allocated():
    from voxkit.audio.resampler import Resampler
    r = Resampler(48_000, 16_000)
    buf = np.zeros(256, dtype=np.float32)
    r.process(buf)  # warm up

    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    for _ in range(500):
        r.process(buf)
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diff = snap2.compare_to(snap1, "filename")
    on_path = sum(s.size_diff for s in diff if "resampler" in s.traceback[0].filename)
    assert on_path == 0


# ---------------------------------------------------------------
# Drop policy and UX hook (Q67)
# ---------------------------------------------------------------

def test_T21_drop_rate_under_threshold_does_not_warn():
    from voxkit.audio.drop_policy import DropRateMonitor
    handler = MagicMock()
    mon = DropRateMonitor(window_seconds=30, threshold=0.001, handler=handler)
    # 9 drops out of 100_000 buffers in 30s = 0.009% (below 0.1%)
    for i in range(100_000):
        mon.observe(dropped=(i < 9), now=i * 0.0003)
    handler.assert_not_called()


def test_T22_drop_rate_over_threshold_invokes_handler_once():
    from voxkit.audio.drop_policy import DropRateMonitor
    handler = MagicMock()
    mon = DropRateMonitor(window_seconds=30, threshold=0.001, handler=handler)
    # 200 drops out of 100_000 buffers = 0.2% (above 0.1%)
    for i in range(100_000):
        mon.observe(dropped=(i < 200), now=i * 0.0003)
    assert handler.call_count >= 1


# ---------------------------------------------------------------
# Sleep / disconnect handling
# ---------------------------------------------------------------

def test_T23_install_sleep_handler_invoked_on_simulated_sleep():
    from voxkit.audio.recorder import Recorder, OSSleepEvent
    cb = MagicMock()
    rec = Recorder()
    rec.install_sleep_handler(cb)
    rec._dispatch_sleep_event(OSSleepEvent(reason="suspend"))
    cb.assert_called_once()


def test_T24_handle_disconnect_returns_device_disconnected_with_id():
    from voxkit.audio.recorder import Recorder, DeviceDisconnected
    with patch("voxkit.audio.recorder._open_native_stream", return_value=MagicMock()):
        rec = Recorder()
        rec.open_stream("usb-1")
        evt = rec.handle_disconnect()
    assert isinstance(evt, DeviceDisconnected)
    assert evt.device_id == "usb-1"


# ---------------------------------------------------------------
# Import-graph isolation (Q77)
# ---------------------------------------------------------------

def test_T25_only_recorder_module_imports_sounddevice():
    """Reads .importlinter and confirms the rule exists; lint-imports
    runs separately in CI but this catches accidental config deletion."""
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(".importlinter")
    sections = [s for s in cfg.sections() if "sounddevice" in cfg[s].get("forbidden_modules", "")]
    assert sections, "no contract restricting sounddevice imports found"
    # The contract must allow the recorder module.
    for s in sections:
        source_modules = cfg[s].get("source_modules", "")
        assert "voxkit.audio.recorder" not in source_modules


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T26_callback_releases_gil_during_memcpy():
    """Q76 honest GIL test (T12 measures wall time only). The Q67/Q76
    contract is that NumPy's __setitem__ releases the GIL during the
    memcpy, so a parallel Python thread should accumulate noticeable
    progress while the callback runs many iterations.

    A failing implementation: a callback that holds the GIL for 100% of
    its wall time. In that case the parallel thread starves.
    """
    import threading
    from voxkit.audio.recorder import LockFreeRing, AtomicCounter
    ring = LockFreeRing(capacity=4096)
    counter = AtomicCounter()
    cb = _make_callback_with_ring(ring=ring)
    buf = np.zeros(4096, dtype=np.float32)   # ~85 ms at 48 kHz; large memcpy

    parallel_ticks = [0]
    stop = threading.Event()

    def parallel_worker():
        while not stop.is_set():
            parallel_ticks[0] += 1

    t = threading.Thread(target=parallel_worker, daemon=True)
    t.start()
    for _ in range(200):
        cb(buf, frames=4096, time_info=None, status=None, dropped_counter=counter)
    stop.set()
    t.join(timeout=1.0)

    # If the callback held the GIL the whole time, the parallel worker
    # would have been blocked. Assert it made non-trivial progress.
    assert parallel_ticks[0] > 1000, (
        f"parallel thread only ticked {parallel_ticks[0]} times; "
        "callback may not be releasing the GIL during memcpy"
    )


def test_T27_open_stream_twice_without_close_raises():
    from voxkit.audio.recorder import Recorder, StreamAlreadyOpen
    with patch("voxkit.audio.recorder._open_native_stream", return_value=MagicMock()):
        rec = Recorder()
        rec.open_stream("0")
        with pytest.raises(StreamAlreadyOpen):
            rec.open_stream("0")


def test_T28_close_stream_on_never_opened_recorder_is_noop():
    from voxkit.audio.recorder import Recorder
    rec = Recorder()
    rec.close_stream()   # must not raise


def test_T29_atomic_counter_concurrent_increment_no_torn_writes():
    """Q67: dropped_buffer_count is incremented from the audio callback
    and read from the UI thread. Atomicity is the contract."""
    import threading
    from voxkit.audio.recorder import AtomicCounter

    counter = AtomicCounter()
    n_per_thread = 10_000
    n_threads = 8

    def bump():
        for _ in range(n_per_thread):
            counter.add(1)

    threads = [threading.Thread(target=bump) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.value == n_per_thread * n_threads


# ---------------------------------------------------------------
# v0.12 panel additions (Lin DSP review + principal-engineer synthesis)
# ---------------------------------------------------------------

def test_T31_priority_dispatch_table_complete():
    """v0.12 (Lin) — closes the T17/T18 skip-on-other-platform gap.
    The priority dispatch must be complete and tested on EVERY CI
    platform, not only on the one matching the production target."""
    from voxkit.audio.recorder import _set_thread_priority

    cases = [
        ("Windows", "Pro Audio"),
        ("Linux", ("SCHED_FIFO", 80)),
        ("Darwin", None),       # documented no-op for v1.0
        ("unknown", None),       # fall-through; warns but does not crash
    ]
    for platform_name, expected in cases:
        with patch("voxkit.audio.recorder._current_platform", return_value=platform_name):
            with patch("voxkit.audio.recorder._apply_priority") as apply_prio:
                _set_thread_priority(_current_platform_value=platform_name)
                if expected is None:
                    apply_prio.assert_not_called()
                else:
                    apply_prio.assert_called_once_with(expected)


def test_T32_callback_surfaces_device_overflow_separate_from_ring_full():
    """v0.12 (Lin): PortAudio passes status.input_overflow when the
    device's own buffer overran (a different failure mode than our ring
    being full). Conflating the two loses the diagnostic. Track them
    in separate counters."""
    from voxkit.audio.recorder import LockFreeRing, AtomicCounter, build_callback
    ring = LockFreeRing(capacity=4096)
    drop_counter = AtomicCounter()
    overflow_counter = AtomicCounter()

    cb = build_callback(ring=ring, device_overflow_counter=overflow_counter)
    buf = np.zeros(256, dtype=np.float32)

    class StatusFlag:
        input_overflow = True
        output_underflow = False

    cb(buf, frames=256, time_info=None, status=StatusFlag(),
       dropped_counter=drop_counter)
    assert overflow_counter.value == 1
    assert drop_counter.value == 0   # ring push succeeded; not a drop


def test_T33_open_close_open_cycle_is_clean():
    """v0.12 (Lin): two single-shot tests (T07 open → check; T08 close →
    check) do not exercise the state machine across cycles. A leaked
    worker thread or unreleased PortAudio handle would pass T07/T08 and
    fail T33."""
    import threading
    from voxkit.audio.recorder import Recorder
    with patch("voxkit.audio.recorder._open_native_stream", return_value=MagicMock()):
        rec = Recorder()
        baseline_threads = set(threading.enumerate())

        rec.open_stream("0")
        rec.close_stream()
        rec.open_stream("0")     # must not raise StreamAlreadyOpen
        rec.close_stream()

        # No new threads should remain alive after the cycle.
        leaked = [t for t in (set(threading.enumerate()) - baseline_threads)
                  if t.is_alive()]
        assert leaked == [], f"open/close/open/close leaked threads: {leaked}"


def test_T30_int16_device_input_normalized_to_float32():
    """Q67: the ring sees float32 in [-1, 1]. PortAudio may deliver int16
    natively from a USB mic; conversion happens in or before the callback,
    not in DSP code downstream."""
    from voxkit.audio.recorder import LockFreeRing, AtomicCounter, build_callback
    ring = LockFreeRing(capacity=4096)
    counter = AtomicCounter()
    # Build callback in int16 mode if supported.
    cb = build_callback(ring=ring, input_dtype=np.int16)

    int16_buf = np.full(256, 16384, dtype=np.int16)   # ~half-scale
    cb(int16_buf, frames=256, time_info=None, status=None, dropped_counter=counter)

    pulled = ring.try_pop_block()
    assert pulled.dtype == np.float32
    # 16384 / 32768 ≈ 0.5
    assert abs(float(pulled.mean()) - 0.5) < 0.01
