# SPDX-License-Identifier: GPL-3.0-or-later
# VoxKit — Test Suite

Test files for VoxKit components, written following Kent Beck's TDD
methodology and Tidy First principles.

## File map

One file per spec component (12 total):

| File | Component | Spec ref |
|---|---|---|
| `test_01_project_session.py` | Project & Session | §11 Component 1 |
| `test_02_recorder.py` | Recording subsystem | §11 Component 2 |
| `test_03_click_bleed_handler.py` | Click bleed handler | §11 Component 3 |
| `test_04_onset_detector.py` | Onset detector | §11 Component 4 |
| `test_05_embedding_extractor.py` | Embedding extractor | §11 Component 5 |
| `test_06_classifier.py` | Classifier (composite gate) | §11 Component 6 |
| `test_07_calibration_manager.py` | Calibration manager | §11 Component 7 |
| `test_08_tempo_grid_engine.py` | Tempo & grid engine | §11 Component 8 |
| `test_09_midi_exporter.py` | MIDI exporter | §11 Component 9 |
| `test_10_playback_engine.py` | Playback engine | §11 Component 10 |
| `test_11_editor_ui.py` | Editor UI | §11 Component 11 |
| `test_12_eval_harness.py` | Eval harness (dev-only) | §11 Component 12 |

## How these files are organized

Each file's docstring is the **test list** — Beck's TODO list of
behaviors-to-verify, ordered simplest-first. Tests are then written
in the same order, so a developer can work top-to-bottom through the
file using the **Red → Green → Refactor** cycle.

### TDD discipline

For each test (T01, T02, ...):

1. **Red:** write the test as it appears in the file. Run it. It
   fails — usually with `ImportError` first, then with `AssertionError`
   once the import resolves.
2. **Green:** write the *minimum* implementation to make this test
   (and all prior tests in the file) pass. Resist over-engineering.
3. **Refactor:** clean up the implementation now that you have a green
   bar. The other tests guard against regressions.
4. Move to the next test.

### Tidy First markers

Inside each file, you'll find comments like:

```python
# ----- TIDY FIRST checkpoint -----
# Before T14: extract `_compute_budget_ms` into a pure function so
# T14/T15 can test budget logic without spinning a thread. Pure
# structural change; tests must stay green during the refactor.
```

These mark spots where Beck's "Tidy First" advice applies: a
**structural change** (extract, rename, move, inline) should land in
its own commit *before* the next behavioral test. The structural
change must keep all existing tests green; it is purely a shape
improvement that makes the next test cheaper to write.

Commit hygiene:
- `tidy: extract _compute_budget_ms` (structural; no behavior change)
- `feat: add buffer-budget tests T14, T15` (behavioral; new tests)
- `feat: implement Q67 buffer-budget logic` (behavioral; new code)

Never mix structural and behavioral changes in the same commit.

### Spec traceability

Every test file references the v0.11 spec questions (Q33, Q66, Q67,
Q68, Q70, Q71, Q72, Q73, Q75, Q76, Q77, Q78, Q79, Q80, Q81, Q85)
that drive its requirements. When a test fails in CI, the failure
message points to the spec question that motivated it, so you can
trace requirement → test → implementation.

### What's mocked vs. real

- **Audio I/O (sounddevice):** mocked. Recorder tests don't actually
  open a stream; they verify the contract.
- **ONNX models:** stubbed at the session level. The eval harness
  has slow tests (`@pytest.mark.slow`) that exercise the real ONNX
  files when present.
- **Qt:** the editor UI tests focus on **state machines**, not
  pixel rendering. The `InferenceWorker` tests run the actual worker
  thread to verify Q76's threading contract.
- **Numerics (Cholesky, Mahalanobis, LR):** real, on synthetic data.
  These are the highest-confidence-required tests; mocking them
  defeats the purpose.

### Markers and slow tests

```python
@pytest.mark.slow                       # exclude with `pytest -m "not slow"`
@pytest.mark.dataset_required("AVP")    # skip if the dataset is missing
```

CI runs the synthetic-tier tests (no slow, no dataset-required) on
every PR. PR-validation runs add the minimum-reproducible-tier tests.
Release validation runs everything.

## Running

```bash
# Fast tier — runs on every PR
pytest -m "not slow"

# PR-validation tier
pytest -m "not slow" --dataset=minimum-reproducible

# Full release tier
pytest --dataset=canonical
```

## Coverage expectations

Per Beck: tests drive design, not coverage. That said, components
1, 2, 3, 6 (the load-bearing ones per v0.11 §0) are expected to
reach > 90% line coverage by the time their tests are all green.
Components 8, 9, 10 (mostly mechanical) should reach > 95%.
Component 11 (UI) is intentionally state-machine-focused; coverage
of rendering code may be lower and that is fine.

## When to add a new test

- A bug report → write the failing test first that reproduces it,
  then fix.
- A new spec question (Q86, Q87, ...) → add a section to the
  appropriate file's test list, then drive implementation.
- A regression in CI → expand the test that was supposed to catch it.

Do not add tests for code that already works. Do not add tests
because "we should test more." Tests exist to drive design forward
and to catch regressions; they pay rent.
