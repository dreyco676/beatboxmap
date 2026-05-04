# VoxKit — Application Specification v0.11

**Document status:** Draft v0.11 — incorporates principal-engineer panel review of v0.10. Eight strong-consensus and four weak-consensus recommendations adopted. Three weak-consensus items recorded as open-question tracking. Three items rejected with rationale.
**Author:** Principal-engineer review (audio DSP / data science / software architecture / open-source maintainer / pragmatist perspectives).
**Supersedes:** v0.10.

**Project license:** GPL v3-or-later (unchanged from v0.10). Q59, Q60, Q61 carry forward.

This document is a delta from v0.10. v0.10 sections not referenced here are unchanged. v0.10 in turn deltas v0.9; the cumulative reading order is v0.9 → v0.10 → v0.11.

---

## Changelog from v0.10

| # | Item | v0.10 | v0.11 |
|---|---|---|---|
| 1 | Click-bleed handler risk surfacing | Component 3 marked "unchanged from v0.9"; no risk-register row; no week-1 validation step | **Click-bleed handler flagged as the highest single technical risk in §6.** New §6 row: "Bleed IR estimation fails on novel headphone/mic setups." Mitigation: week-1 tracer-bullet implementation on the maintainer's own setup with a deliberately leaky pair of open-back headphones; if the simplest LMS/Wiener-based IR estimation cannot achieve > 20 dB null on the click-only segment after 2 seconds of adaptation, escalate to a more sophisticated IR model before continuing with downstream components. The bleed handler being a v0.9 carry-over is precisely why it has not been re-stress-tested at the v0.10 / v0.11 architecture; the tracer bullet exists to confirm the v0.9 design holds up under the v0.10 threading and sample-rate contract. See §6 and §9 week 1. |
| 2 | Audio callback Python / GIL contract | Q67 specifies "no allocation, no logging, no system calls" but is silent on GIL behavior under sounddevice's PortAudio binding | **GIL contract specified.** Two acceptable implementations: (a) **Default path:** Python callback that operates only on a pre-allocated NumPy buffer copy via `out[:] = indata` style assignment plus an atomic counter; NumPy's array operations release the GIL for the copy itself. The callback acquires the GIL only to enter the function and to schedule the Python-side counter increment. Total GIL-held time at typical 5–10 ms buffers is < 50 µs on the reference CPU, well inside the audio thread's slack. (b) **Hardened path:** sounddevice `RawInputStream` with a CFFI-level callback that does not acquire the GIL at all, pushing into a C-level lock-free ring (e.g., `cffi`-bound SPSC ring). Use (b) only if (a) measurably drops buffers under load on the reference CI hardware; (a) is the v1.0 default. Documentation block in `voxkit/audio/recorder.py` records which path is in effect. See §5.1, §11 Component 2. |
| 3 | Inference pipeline threading | Implicit; only the resampler-worker thread is specified | **Inference pipeline runs on a dedicated `InferenceWorker` thread.** The thread executes the three Q73 phases (onset → embedding → classification) and emits Qt signals (PyQt6) or equivalents (PySide6) for progress updates and completion. The main thread runs only the Qt event loop and never blocks on inference. Cancel is implemented via a thread-safe `Event` flag checked between phases and at the top of each per-onset iteration inside the embedding-extraction loop; cancellation is best-effort with a worst-case latency of one onset's embedding extraction (~50 ms). The cancel path preserves the recorded audio buffer in the Session. Inference worker priority is normal (not real-time); embedding extraction is the dominant cost and is CPU-bound, not latency-sensitive. See §5.8, §11 Component 11. |
| 4 | Click-bleed quality indicator metric | Indicator named in v0.9 / v0.10 but not defined | **Metric defined.** The indicator displays the *post-subtraction click residual ratio*: the RMS energy of the cleaned audio in click-aligned 50 ms windows, divided by the RMS energy of the click-only calibration recording in the same window length, expressed in dB. Range mapped to UI as: green ≥ 20 dB attenuation, yellow 10–20 dB, red < 10 dB. Below 10 dB the bleed banner offers re-running calibration or accepting the override. The numeric value is shown alongside the colored bar for advanced users. See §5.1.1, §11 Component 3. |
| 5 | Q62 import-graph isolation enforcement | "Win32-isolation should be enforced via lint or import-graph checks, not just convention" recorded as a Sam review note in v0.10 §8; not promoted to a tooling commitment | **Promoted to enforced check.** `import-linter` configuration committed to repo as `.importlinter` with the rules: (a) `voxkit.audio.recorder` is the only module that may import `sounddevice` or any platform audio library; (b) no module under `voxkit.core`, `voxkit.classifier`, `voxkit.eval`, or `voxkit.ui` may import `win32`, `pywin32`, `ctypes.windll`, `Foundation`, `AppKit`, or any platform-specific Linux library by name; (c) `voxkit.core` may not import `voxkit.ui`. CI runs `lint-imports` on every PR; violations block merge. See §9 week 1, §11 Component 2. |
| 6 | CONTRIBUTING.md | Implicit; mentioned nowhere | **CONTRIBUTING.md committed in week 1** covering: (a) environment setup (Python version, virtual environment, `pip install -e .[dev]`), (b) running the synthetic-dataset CI tier locally, (c) obtaining the minimum-reproducible dataset (the project-hosted subset per Q63) for PR validation, (d) license expectations including SPDX header on every new source file, (e) PR checklist (tests pass, `lint-imports` clean, no SPDX-header drift), (f) where to ask questions (GitHub Discussions, not Issues). Skeleton committed in week 1 even if some sections are stubs to be filled out in week 2 and beyond. See §9 week 1. |
| 7 | Format version field in manifest.json | Migration v0.4–v0.9 → v0.10 implicit on field presence/absence; no version field | **Explicit version field.** `manifest.json` gains a top-level `voxkit_format_version` string field (e.g., `"0.11"`). Migration code dispatches on this field via a registered table of `(from_version, to_version) -> migrator` functions; legacy bundles with no field are treated as `"0.4"` for migration purposes (matches the earliest known format). The Cholesky conversion from v0.10 becomes one entry in this table rather than implicit at-load behavior. See §11 Component 1. |
| 8 | CalibrationRejected dialog wording | "Your recent calibration didn't improve the model on the held-out test set. Try recording in a quieter environment or adjusting your microphone." | **Wording corrected to be honest about the cause and complete on remediation.** New text: "VoxKit's accuracy check found that the most recent calibration didn't improve classification on the held-out test set; the previous calibration has been restored. This usually means the calibration recording was very different from your typical use (very few samples, unusual background noise, or a different microphone), and the model would have generalized worse with it. You can try again with more or quieter samples, or continue using the previous calibration." Diagnostic file always records the macro-F1 delta that caused the rejection so support requests can reference real numbers. See §11 Component 6. |
| 9 | Phase 1.5 Linux commitment timing | "Linux build ships in Phase 1.5 (~4–6 weeks after Windows beta)"; framed as a calendar commitment | **Reframed to "as bandwidth permits."** The architectural constraint of Q62 (and now Q77, the linter enforcement) is the real Phase 1 deliverable; the calendar slot for the Linux build is reframed as "ships when there is time, with the architectural lock-in ensuring it remains tractable when that time comes." This is honest about hobby pace and avoids creating a contributor expectation that gets repeatedly broken. The Phase 1.5 contents (ALSA / PipeWire wiring, `SCHED_FIFO` install docs, AppImage / Flatpak packaging, Linux device test matrix) are unchanged in shape; only the calendar wording is softened. See §3, §9. |
| 10 | Synthetic dataset purpose | Q63 tier (c) labeled "absolute floor for CI" without explanation of what that does and does not validate | **Purpose made explicit in spec text.** The synthetic dataset is procedurally generated drum samples; it does not represent vocal-percussion acoustics and is intentionally simpler than real data. CI on the synthetic tier validates only that (a) the pipeline runs end-to-end without errors, (b) imports are valid, (c) Cholesky factoring round-trips, (d) MIDI export produces a parseable file, (e) the eval harness emits its expected JSON shape. CI green on synthetic does **not** validate model quality; PR validation against the minimum-reproducible tier is the lowest level of quality validation, and release-tier validation against the full canonical dataset is what gates v1.0. README and the eval-harness `--help` text both repeat this caveat. See §7.10. |
| 11 | SPDX header CI enforcement | SPDX headers added to source files in week 1; enforcement not specified | **CI enforcement added.** `pre-commit` hook plus a CI job runs the `reuse` tool (or an equivalent custom check) verifying every `.py`, `.pyx`, and `.toml` source file has `SPDX-License-Identifier: GPL-3.0-or-later`. Generated files (e.g., compiled extensions, lockfiles, dataset metadata JSON) are listed in `.reuse/dep5` as exempt. Violations block merge. Cost: roughly an hour to set up plus a one-time pass to add headers; recurring cost effectively zero. See §9 week 1. |
| 12 | Q60 license-review memo template | Lists six fields: code license, weight license, commercial-use restrictions, fine-tuning rights, attribution, redistribution | **Seventh field added: pre-training data license propagation.** Specifically: "If the model was pre-trained on a dataset with usage restrictions (e.g., AudioSet's YouTube-derived non-commercial-research framing), do those restrictions propagate to the released model weights for downstream redistribution under GPL v3?" This is a legal grey area; the memo's role is to surface the question so the project doesn't ship redistributable weights with an unaddressed pre-training-license concern. PANNs and BEATs are both AudioSet-pre-trained; the memo will need to document the project's interpretation. See §9 week 1. |

Twelve adopted items. Items 1–8 strong consensus (≥ 6/9 panel votes). Items 9–12 weak consensus (4–5 votes), adopted because each is cheap, none drew dissent, and each removes a small but real friction.

**Three items recorded as weak-consensus tracking** (not adopted as committed v0.11 changes; carried forward as open questions in §10):

- *Lock-free SPSC ring buffer named implementation reference* — Lin's recommendation to commit to a specific lock-free ring implementation (rather than describing the contract). Reasonable but premature; the v0.10 callback contract (now Q67 + the Q76 GIL clarification) is enough to constrain implementation. Revisit if the audio-thread no-allocation tracemalloc test or the dropped-buffer-rate metric flags a problem under real-world load. Recorded as §10 item 21.
- *Per-class F1 reporting alongside macro-F1 in eval* — Marco / Priya / Alex's recommendation. Defensible but the existing macro-F1 + per-class operating-point selection in §7.3 already lets a closed/open-hat regression be detected; adding per-class F1 to the release-gate output is cheap but not load-bearing. Recorded as §10 item 22.
- *Re-run inference from the editor without re-record* — Jordan / Marco's UX recommendation. Real value but adds Component 11 surface area; deserves a deliberate v1.1 design pass rather than a v1.0 retrofit. Recorded as §10 item 23.

**Three items rejected** (with rationale, recorded in §8):

- *AVP class-imbalance handling specified explicitly in spec* — Priya's note. Verified: AVP is approximately balanced across the four classes after subject-level pooling. The class-balance question is settled by data inspection, not by spec language. No change needed.
- *Re-specifying `bleed_ir_history` "two protected slots" inline in v0.11* — Sam's hygiene note. Sympathetic but the v0.9 specification is correct and stable; re-stating in every version invites accidental drift. Better to leave the v0.9 reference and add a one-line gloss only if a contributor flags confusion in practice.
- *Cutting canonical dataset hosting from v1.0* — Casey's scope-down suggestion. Riley dissents from the contributor-experience side: removing the hosted canonical tier turns "first-time contributor reproducing the release-gate eval" from a one-command flow into a multi-step ToS / download / placement flow that disproportionately bounces newcomers. Casey's pragmatism point holds for the canonical *being maintained at scale* but not for it existing at all. Keep the three-tier plan from Q63.

---

## 0. How to read this document

The architectural changes from v0.10 are confined to: §5.1 / §11 Component 2 (audio callback GIL contract, import-graph isolation enforcement), §5.8 / §11 Component 11 (inference-worker threading, recording-session UX correction), §11 Component 1 (format version field, dispatch-table migration), §11 Component 3 (click-bleed quality indicator metric definition), §11 Component 6 (CalibrationRejected wording), and §6 (click-bleed risk row). Everything else is hygiene (CONTRIBUTING.md, SPDX enforcement, license memo template, synthetic-dataset purpose) or honest re-framing (Phase 1.5).

**The load-bearing v0.11 changes are #1 (click-bleed risk surfacing — the bleed handler is the most uncertain DSP component and has not been stress-tested against the v0.10 threading contract), #2 (GIL contract — the difference between "MMCSS works as intended" and "MMCSS works but Python's callback drops buffers anyway"), #3 (inference threading — without it, Q73's cancel button is a lie), and #5 (import-graph enforcement — convention-only isolation does not survive a project's third contributor).** The remaining items are correctness (Q73 cancel semantics, format version), methodology (license memo, synthetic purpose), or operational hygiene (CONTRIBUTING.md, SPDX CI, dialog wording).

**Hobby-project framing carried from v0.10:** the cost of adopting all twelve items in v0.11 is small. The CONTRIBUTING.md, SPDX CI, format version, and dialog wording items are each under an hour of work. The import-graph linter is roughly half a day. The GIL contract clarification is documentation plus one tracemalloc test. The inference-worker threading is real but already implied by Q73; the v0.11 spec just makes it explicit. The click-bleed tracer bullet in week 1 is the most expensive item, and that one is genuine new work — but it is replacing the implicit assumption that v0.9's design still holds with a measured confirmation, which is the right tradeoff.

---

## 1. Executive summary

Unchanged from v0.10 in product shape. Changes for v0.11:

- Click-bleed handler flagged as the highest single technical risk; week-1 tracer bullet added.
- Audio callback Python / GIL contract specified, with a hardened CFFI-callback fallback if measured drops require it.
- Inference pipeline runs on a dedicated `InferenceWorker` thread with cancellable progress; main thread runs the Qt event loop only.
- Click-bleed quality indicator defined as post-subtraction click residual ratio (dB).
- Q62 platform isolation enforced via `import-linter` in CI.
- CONTRIBUTING.md committed in week 1.
- `manifest.json` gains an explicit `voxkit_format_version` field; migrations dispatched via a registered table.
- CalibrationRejected dialog wording corrected to be honest about the cause and complete on remediation.
- Phase 1.5 Linux build re-framed to "as bandwidth permits"; architectural lock-in is the real Phase 1 deliverable.
- Synthetic dataset purpose (smoke not quality) made explicit in spec text and `--help`.
- SPDX header enforcement added to CI via `reuse`.
- Q60 license-review memo template extended with a pre-training-data license propagation field.

---

## 2. Resolved decisions

v0.10 decisions Q1–Q75 carry over. Amendments and additions for v0.11:

| # | Decision | Resolution |
|---|---|---|
| Q60 *(amended)* | License-review memo scope | Seventh field added: "If the model was pre-trained on a dataset with usage restrictions (e.g., AudioSet's non-commercial-research framing), do those restrictions propagate to the released model weights for downstream GPL v3 redistribution?" Memos in `docs/licenses/` must answer this for PANNs and BEATs explicitly. |
| Q67 *(amended)* | Audio-thread sample-format and threading specifics | GIL contract added: default path is Python callback with NumPy-buffer copy and atomic counter (GIL held < 50 µs/call typical); hardened path is sounddevice `RawInputStream` with CFFI-level callback when (a) measurably drops buffers under load. v1.0 ships (a); (b) is the documented escalation path. The non-Python parts of Q67 (MMCSS, drop policy, latency budget) are unchanged. |
| Q76 *(new)* | Inference pipeline threading | Onset detection, embedding extraction, and classification run on a dedicated `InferenceWorker` thread (Python `threading.Thread` plus Qt signals, or `QThread` directly — implementer's choice). The main thread runs the Qt event loop only. Cancellation via thread-safe `threading.Event`; worst-case cancel latency is one embedding-extraction call (~50 ms reference). On cancel, the recorded audio buffer is preserved in the Session; partial events are discarded. Worker priority is normal (CPU-bound, not latency-sensitive). |
| Q77 *(new)* | Platform-isolation enforcement | `import-linter` configuration in `.importlinter` committed to repo with rules: (a) `voxkit.audio.recorder` is the only module permitted to import platform audio libraries (`sounddevice`, future `pyaudio`, future `coreaudio`); (b) `voxkit.core`, `voxkit.classifier`, `voxkit.eval`, `voxkit.ui` may not import any platform-specific module by name; (c) `voxkit.core` may not import `voxkit.ui`. CI runs `lint-imports` on every PR; violations block merge. |
| Q78 *(new)* | Format version field | `manifest.json` adds `voxkit_format_version: str` at top level. Migration is dispatched via a registered table of `(from_version, to_version) -> migrator`. Bundles missing the field are treated as `"0.4"`. The Q68 Cholesky conversion becomes one entry in this table. |
| Q79 *(new)* | Click-bleed quality indicator metric | Post-subtraction click residual ratio in dB: `20 * log10(rms(cleaned_audio[click_aligned_windows]) / rms(click_calibration[click_aligned_windows]))`, more negative is better, displayed as positive attenuation in dB on the UI. Mapped to color: green ≥ 20 dB, yellow 10–20 dB, red < 10 dB. Below 10 dB triggers the bleed banner. Numeric value shown alongside the bar. |
| Q80 *(new)* | Click-bleed handler tracer bullet | Week 1 task: implement the simplest LMS / Wiener-based IR estimation against a 2-second click-only calibration recording on the maintainer's own setup with deliberately leaky open-back headphones. Acceptance: > 20 dB null on the click-only segment after 2 seconds of adaptation. If the simplest approach fails, escalate to a more sophisticated IR model (e.g., NLMS with regularization, or RLS) before continuing with downstream components. The tracer bullet exists to confirm the v0.9 design assumption holds under the v0.10 + v0.11 threading and sample-rate contract. |
| Q81 *(new)* | CalibrationRejected dialog text | "VoxKit's accuracy check found that the most recent calibration didn't improve classification on the held-out test set; the previous calibration has been restored. This usually means the calibration recording was very different from your typical use (very few samples, unusual background noise, or a different microphone), and the model would have generalized worse with it. You can try again with more or quieter samples, or continue using the previous calibration." Diagnostic file records the macro-F1 delta. |
| Q82 *(new)* | SPDX header enforcement | Pre-commit hook plus CI job runs `reuse` (or equivalent) on every PR; missing or wrong SPDX header on a tracked source file blocks merge. `.reuse/dep5` lists exempted generated files. |
| Q83 *(new)* | CONTRIBUTING.md | Skeleton committed week 1 covering environment setup, synthetic-tier CI invocation, dataset acquisition, license expectations, PR checklist, and where to ask questions. Some sections may be stubs in week 1, filled out by week 4. |
| Q84 *(new)* | Phase 1.5 Linux framing | "Ships as bandwidth permits, with the architectural constraint of Q62 / Q77 ensuring it remains tractable when that time comes." Architectural lock-in is the deliverable; the calendar slot is honestly conditional. Phase 1.5 contents (Q62) unchanged in shape. |
| Q85 *(new)* | Synthetic dataset role | Synthetic tier validates pipeline runs, imports, Cholesky round-trip, MIDI parseability, and eval-harness JSON shape. Does NOT validate model quality. README and eval-harness `--help` both repeat this caveat. |

---

## 3. Scope

Phase 1 — Windows-only (architecturally portable per Q62 / Q77). **Phase 1.5 — Linux build ships as bandwidth permits**, not on a calendar commitment; the architectural constraint is what locks in the option (Q84). Phase 2 — macOS build.

The Linux-before-Mac priority order is unchanged: OSS contributor distribution skews Linux, so the architectural readiness is sized to that audience first.

Other scope items unchanged from v0.10.

---

## 4. Architecture

### 4.1 Top-level shape

Unchanged.

### 4.2 Stack (updated rows)

| Layer | Choice | Why |
|---|---|---|
| Audio I/O | `sounddevice` with WASAPI default, MME fallback (Windows). Audio callback per Q67 amended (Q76 GIL contract). Default Python-callback path; CFFI-callback path documented as escalation if measured drops require it. | Q67 amended. |
| Inference threading | `InferenceWorker` thread executing onset + embedding + classification; main thread is Qt event loop only. Q76. | Q73's cancellable progress dialog requires this. |
| Platform isolation | Enforced via `import-linter` in CI per Q77; not convention. | Q62 lock-in only survives if it's a check, not a convention. |

Other rows unchanged from v0.10.

### 4.3 Why offline

Unchanged.

### 4.4 Phase 1.5 Linux framing

Phase 1.5 is now an *architecturally enabled, calendar-flexible* phase. The architectural constraint (Q62 + the Q77 enforcement) ensures that when bandwidth allows the Linux build, it is a wiring exercise rather than a refactor. The contents of the Linux build (ALSA / PipeWire backend, `SCHED_FIFO` install docs, AppImage / Flatpak packaging, Linux device test matrix) are unchanged from v0.10's Phase 1.5 description.

---

## 5. Component specifications (high-level)

### 5.1 Recorder

Per v0.10, plus:

- **Audio callback GIL contract (Q67 amended):** the default v1.0 callback is implemented as a Python function that performs only a NumPy buffer copy (`out[:] = indata`) and an atomic increment of `dropped_buffers` on `try_push` failure. NumPy's `__setitem__` for an array-to-array copy releases the GIL during the memcpy itself; the GIL is held only during function-entry and counter-update bookkeeping, totaling < 50 µs per call at typical buffer sizes on the reference CPU. If the audio-thread no-allocation tracemalloc test or the dropped-buffer-rate metric flags problems under real-world load, the documented escalation is to switch to `sounddevice.RawInputStream` with a CFFI-level callback that does not acquire the GIL at all and pushes directly into a C-level lock-free ring. The callback path in effect is recorded in a docstring at the top of `voxkit/audio/recorder.py`.
- **Import-graph isolation (Q77):** `voxkit.audio.recorder` is the *only* module permitted to import `sounddevice` or any future platform audio library. CI runs `lint-imports` on every PR. Violations block merge.
- **Sleep-handler abstraction:** unchanged; carry from v0.10's portability constraint.

Other Recorder behavior unchanged from v0.10.

### 5.1.1 Click-bleed quality indicator

The indicator displays the post-subtraction click residual ratio as defined in Q79: 20·log₁₀(RMS of cleaned audio in click-aligned 50 ms windows / RMS of the click-only calibration recording in the same windows), reported as positive attenuation in dB. Color thresholds: green ≥ 20 dB, yellow 10–20 dB, red < 10 dB. The numeric value is shown alongside the colored bar; tooltip on hover explains the metric in plain language. Below 10 dB triggers the bleed banner (carry from v0.9), now offering re-running calibration as the primary remediation.

### 5.2 Onset detector

Unchanged from v0.10.

### 5.2.1 Headphone-bleed setup step

Unchanged from v0.9, with the v0.11 metric (Q79) used for the per-step quality reading.

### 5.3 Embedding extractor

Unchanged from v0.10.

### 5.4 Classifier (composite unknown gate, refined further)

Unchanged in shape from v0.10. Wording change only:

- **CalibrationRejected dialog text (Q81):** see Q81 for full text. The diagnostic file always records the macro-F1 delta that caused the rejection.

### 5.5 Tempo & grid engine

Unchanged.

### 5.6 Calibration manager

Unchanged from v0.10 except for the dialog wording correction in §5.4.

### 5.7 Telemetry

Unchanged from v0.10. The local diagnostic file format gains one new event type:

```json
{"event": "calibration_overfit_guard_triggered",
 "details": {"f1_calibrated": ..., "f1_baseline": ..., "delta": ...}}
```

(Diagnostic schema is intentionally additive; readers ignore unknown fields.)

### 5.8 Editor (UI)

Per v0.10, plus:

- **Inference pipeline threading (Q76):** the recording-session progress dialog from Q73 is now backed by a dedicated `InferenceWorker` thread. The main thread runs only the Qt event loop. Phase progress is reported via Qt signals; the cancel button sets a `threading.Event` flag checked between phases and at the top of each per-onset iteration inside embedding extraction. Worst-case cancel latency is one embedding-extraction call (~50 ms reference). On cancel, the recorded audio buffer is preserved in the Session; partial events are discarded.

Other v0.10 editor behavior unchanged.

### 5.9 MIDI exporter

Unchanged from v0.10.

### 5.10 Project file

Schema additions on top of v0.10:

- `manifest.json` gains `voxkit_format_version: str` at top level (Q78). Migration code dispatches on this field via a registered table.
- v0.10 → v0.11 migration: bundles loaded with no version field are treated as `"0.4"` and walked through the sequential migration table (which now includes the v0.10 Cholesky-conversion step explicitly). Bundles with `voxkit_format_version: "0.10"` get only the version-field-stamping pass. No other on-disk field changes in v0.11.

Migration matrix tests extended to v0.11. Sunset cadence per Q55 unchanged.

---

## 6. Risk register (deltas from v0.10 in **bold**)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Click-bleed handler IR estimation fails on novel headphone/mic setups (NEW)** | **Medium** | **High (cascade: bleed misclassified as percussion, all downstream metrics degrade)** | **Q80 week-1 tracer bullet on maintainer's setup with deliberately leaky open-back headphones. Acceptance threshold: > 20 dB null after 2 seconds. Escalation path documented (NLMS / RLS) if simplest approach fails.** |
| **Audio callback GIL contention causes buffer drops despite MMCSS (NEW)** | **Medium without fix** | **Medium** | **Q67 amended: default path keeps GIL-held time < 50 µs/call via NumPy buffer copy; hardened CFFI-callback path documented as escalation if measured drops require it.** |
| **Inference pipeline blocks main thread during recording-stop processing (NEW)** | **High without fix** | **Medium (Q73 cancel button non-functional, UI freeze)** | **Q76: dedicated `InferenceWorker` thread; main thread runs Qt event loop only; cancellation via `threading.Event` checked between phases and per-onset.** |
| **Platform-specific code leaks beyond `Recorder` over time as contributors join (NEW)** | **High without fix** | **Medium (Phase 1.5 Linux becomes a refactor, not a wiring exercise)** | **Q77: `import-linter` enforces isolation in CI; violations block merge.** |
| **First-time contributors cannot run any tests because dataset access is unclear (NEW)** | **Medium** | **Medium (contribution velocity collapses)** | **Q83 CONTRIBUTING.md committed week 1 documents the synthetic-tier-only quickstart path.** |
| **Source files drift out of SPDX compliance over time (NEW)** | **Medium without fix** | **Low–Medium** | **Q82: pre-commit hook + CI `reuse` check blocks merge.** |
| **Pre-training data license (AudioSet) propagation to redistributable model weights unaddressed (NEW)** | **Medium** | **Medium (potential redistribution legal grey area)** | **Q60 amended: memo template seventh field surfaces the question for explicit disposition.** |

Other v0.10 risks unchanged.

---

## 7. Test strategy

### 7.1 — 7.7

Unchanged in shape.

### 7.8 Onset-detection release gate

Unchanged from v0.10.

### 7.9 OOD validation

Unchanged from v0.10.

### 7.10 Tiered eval cadence

Unchanged in tier structure. **Synthetic-tier purpose made explicit (Q85):**

The synthetic tier validates pipeline runs, imports, Cholesky round-trip, MIDI parseability, and eval-harness JSON shape. It does NOT validate model quality. Quality validation begins at the minimum-reproducible tier (PR validation) and is final at the canonical tier (release validation). The README and `python -m voxkit.eval --help` both repeat this caveat; the synthetic-tier CI report names the tier in its output banner so contributors do not mistake "CI green" for "quality validated."

### 7.11 New CI checks (v0.11)

- **Import-graph linter (Q77):** `lint-imports` runs on every PR; violation blocks merge.
- **SPDX header check (Q82):** `reuse` lint runs on every PR; missing or wrong header blocks merge.
- **Audio callback no-allocation regression test (carry from v0.10, hardened):** `tracemalloc` against the audio thread on a 10-minute synthetic session, validating zero on-thread allocations after `open_stream`. v0.11 adds a second invocation with the hardened CFFI-callback path enabled if it has been wired in, even though v1.0 default is the Python path.
- **Migration table round-trip (Q78):** for each registered `(from, to)` migrator, a synthetic bundle in the `from` schema is migrated, then re-loaded, then re-saved; the resulting bundle equals what fresh-saving the same logical data would have produced. Catches half-migrations and unregistered legacy fields.

---

## 8. Multi-perspective review (v0.11 panel)

v0.5 / v0.6 / v0.7 / v0.8 / v0.9 / v0.10 reviews stand. v0.11 panel notes (nine reviewers, voted per item; ≥ 6/9 = strong consensus, 4–5/9 = weak consensus):

- **Lin (DSP):** v0.10 specifies MMCSS but is silent on Python's GIL behavior in the audio callback. Without explicit guidance, a contributor will land a logging call or dict access in the callback and we will see drops that are not MMCSS's fault. Q67 amendment is the correct fix. Click-bleed handler being unchanged from v0.9 across two architectural revisions is a hidden risk; week-1 tracer bullet is overdue. Onset-detector tongue-click handling and named lock-free ring implementation are nice-to-haves; defer.
- **Priya (ML):** Nothing in v0.10's ML core needs a v0.11 fix. The CalibrationRejected wording is the small but genuine issue: blaming the user's environment when the actual cause is statistical drift on the held-out fold is dishonest. Q81 corrects it. Pre-training data license propagation (AudioSet → released weights) is the legal grey area worth surfacing in Q60. Per-class F1 reporting is cheap; happy to defer.
- **Sam (Architecture):** Q77 (import-linter) is the most operationally important architectural change in v0.11. Convention-only platform isolation does not survive a project's third contributor. Q78 (format version field) is overdue; v0.4–v0.10 migration has been implicit on field presence/absence, which is fine for two versions and brittle by version five. Q76 (inference threading) is the right fix to a v0.10 oversight: Q73 promises a cancel button that doesn't work without an `InferenceWorker`. The bleed_ir_history "two slots" re-spec is a small concern not worth a v0.11 entry; agree to defer.
- **Jordan (UX/PM):** Click-bleed quality indicator getting a real metric definition (Q79) means users can interpret the colored bar; without this, the indicator is theater. CalibrationRejected dialog text correction (Q81) is the kind of small honesty item that adds up to user trust. Re-running inference from the editor without re-recording is real value but deserves a deliberate v1.1 design pass; happy to track in §10.
- **Marco (Domain):** No drumming-specific changes in v0.11; v0.10 already covered the percussion-relevant items (alignment MAE, taxonomy parameterization). Click-bleed tracer bullet matters more than it sounds — if click bleed is misclassified as percussion, every musician using VoxKit for the first time will see false hits and conclude it doesn't work, regardless of how good the trained classifier is. Per-class F1 in eval would help but defer.
- **Alex (QA):** Q77, Q82, and the migration table round-trip test are the quality-gate additions that matter. Synthetic-tier purpose disambiguation (Q85) is small but prevents the single most predictable contributor mistake (assuming CI green = ready to merge a model change). The audio-callback no-allocation regression test was already specified in v0.10; v0.11 hardens it with a second-path invocation if the CFFI path is wired.
- **Dana (Security/Legal):** Q60 amendment (pre-training data license propagation field) closes a real grey area for AudioSet-pre-trained weights. Q82 SPDX CI enforcement is appropriate due-diligence; an unmaintained SPDX state is worse than no SPDX claim. License-review memos in `docs/licenses/` carry forward from v0.10 with the new field.
- **Riley (OSS maintainer):** CONTRIBUTING.md in week 1 (Q83) is the highest-leverage single OSS-hygiene item not yet captured. Phase 1.5 Linux re-framing (Q84) is honest and avoids creating a contributor expectation that gets repeatedly broken by hobby pace. Import-graph enforcement (Q77) is what makes the architectural lock-in real for downstream contributors.
- **Casey (Pragmatist):** The twelve adopted items in v0.11 are individually small (CONTRIBUTING.md skeleton, SPDX CI, format version field, dialog wording, synthetic-purpose docs are each under an hour) plus three real items (Q77 import-linter is half a day; Q76 inference worker is a few hours; Q80 click-bleed tracer bullet is genuine new work, on the order of two days). Total v0.11 cost is roughly one work-week, much smaller than v0.10's. The bleed tracer bullet is the only schedule-meaningful item; everything else is hygiene. Comfortable with v0.11 from a hobby-scope perspective.

**Rejected items recorded:**

- *Specifying AVP class-imbalance handling explicitly in spec* — Priya's note. Verified: AVP is approximately balanced after subject-level pooling. The class-balance question is settled by data inspection, not spec language. No change.
- *Re-specifying `bleed_ir_history` "two protected slots" inline in v0.11* — Sam's hygiene note. v0.9 specification is correct and stable; re-stating in every version invites accidental drift. Better to add a one-line gloss only if a contributor flags confusion in practice.
- *Cutting canonical dataset hosting from v1.0* — Casey's scope-down suggestion. Riley dissents from the contributor-experience side: removing the hosted canonical tier turns "first-time contributor reproducing the release-gate eval" from a one-command flow into a multi-step ToS / download / placement flow that disproportionately bounces newcomers. Casey's concern is about *maintaining* the canonical tier at scale; the v1.0 cost of *creating* it is bounded by Q63's existing scope.

**Carry-forward rejected items from v0.7 / v0.8 / v0.9 / v0.10 (all still rejected):**
- Cut mid-session re-estimation entirely.
- Replace LR head with fine-tuned final layers.
- Per-user `unknown_threshold` UI in v1.0.
- Throat-bass as a 6th trained class in v1.0 (config change in v1.1 per Q66).
- Joint sweep of L2 regularization C with calibration weight.
- Cleaner `TrainedClass` / `RuntimeClass` type split.
- Adding macOS to Phase 1 in addition to Linux.
- Bluetooth full reversal (preserved as Phase 2 prototype, §10 item 17).

---

## 9. Phased delivery plan

Phase 1 contents updated for v0.11:

**Phase 1 — Analysis pipeline**

*Week 1 (additions on v0.10):*
- **CONTRIBUTING.md skeleton committed** (Q83). Some sections may be stubs filled out by week 4.
- **`.importlinter` configuration committed** with Q77 rules; CI job `lint-imports` wired into PR check set.
- **`reuse` configuration and CI job for SPDX header enforcement** (Q82); `.reuse/dep5` listing exempted generated files.
- **`voxkit_format_version` field added to `manifest.json`** (Q78); migration dispatch table populated with the v0.10 Cholesky-conversion entry as the first registered migrator.
- **License-review memos extended with the seventh field** (Q60 amended): pre-training data license propagation to released weights; PANNs and BEATs memos must answer this explicitly.
- **Click-bleed handler tracer bullet** (Q80): simplest LMS / Wiener-based IR estimation, 2-second adaptation against click-only calibration, acceptance > 20 dB null on a deliberately leaky headphone setup. If it fails, escalate to NLMS / RLS before continuing with downstream components. This is the only schedule-meaningful v0.11 item.
- **Inference-worker scaffolding** (Q76): `InferenceWorker` thread class with `start()`, `cancel()`, and Qt-signal hooks; not wired to UI yet, but contract published.
- All v0.10 week-1 items (license file, SPDX headers, A/B time-stretch quality test, dataset access plan, 5-subject OOD pilot recruitment, release-gate justification methodology document, OOD power sensitivity analysis, Component 1 / Component 2 with MMCSS, Linux CI smoke test, tiered eval harness, silent-window test fixtures) carry forward.

*Week 2 (additions on v0.10):*
- Component 6: GIL-contract documentation block in `voxkit/audio/recorder.py` (Q67 amended) recording the active callback path.
- Component 11: `InferenceWorker` wired to the recording-session progress dialog with cancel semantics per Q76.
- Component 11: CalibrationRejected dialog wording updated per Q81; diagnostic-file logging of the macro-F1 delta wired up.
- Component 1: `voxkit_format_version` write path on save; migration table entry tested via the round-trip test (§7.11).
- All v0.10 week-2 items (Q33 substrate decision, Q43 PCA-64 decision, Q42 / Q65 calibration weighting, CPU performance benchmark per substrate, Component 3 silent-window re-estimation, Component 6 with Cholesky / self-test guard / TaxonomyConfig, OOD subject recruitment, user-impact study recruiting) carry forward.

*Week 3–8:* unchanged from v0.10.

**Phase 1.5 — Linux build (as bandwidth permits):** unchanged in shape; calendar wording softened per Q84. Architectural lock-in via Q62 + Q77 ensures the build is a wiring exercise when bandwidth allows.

**Phase 2 — macOS build:** unchanged.

**Phase 1 critical-path implications (v0.11 additions):**
- `import-linter` and `reuse` CI jobs from week 1; both block merge.
- `manifest.json` version field and migration table from week 1.
- Click-bleed tracer bullet from week 1; escalation path defined if simplest IR estimation fails the > 20 dB acceptance threshold.
- `InferenceWorker` scaffolding from week 1, wired to UI in week 2.
- All v0.10 critical-path items unchanged.

---

## 10. Remaining open questions

v0.7 / v0.8 / v0.9 / v0.10 questions carry forward except as noted. New / updated for v0.11:

7.–20. Unchanged from v0.10.

21. **(new — v0.11 weak-consensus tracking)** Lock-free SPSC ring buffer named implementation reference. v0.11 keeps the contract-only specification (Q67 + Q76); commit to a named implementation if the audio-thread no-allocation tracemalloc test or dropped-buffer-rate metric flags a problem under real-world load.

22. **(new — v0.11 weak-consensus tracking)** Per-class F1 reporting alongside macro-F1 in release-gate output. Cheap; deferred because macro-F1 + per-class operating-point selection in §7.3 already lets a closed/open-hat regression be detected. Revisit if vocal-percussion-specific confusability shows up in v1.0 telemetry.

23. **(new — v0.11 weak-consensus tracking)** Re-run inference from the editor without re-recording. Real UX value but adds Component 11 surface area; deserves a deliberate v1.1 design pass rather than a v1.0 retrofit.

24. **(carried)** v1.0 user-facing class taxonomy editing (was item 18 in v0.10 §10).

25. **(carried)** Distribution mechanism for canonical eval datasets if AVP / OOD redistribution rights are not confirmed in week 1 (was item 19 in v0.10 §10).

26. **(carried)** Qt binding choice: PyQt6 vs PySide6 (was item 20 in v0.10 §10).

---

## 11. Component specifications (build-ready)

### Component diagram (data flow, left to right)

Unchanged from v0.10 in shape. Annotations for v0.11:

```
[Recorder] ──▶ raw_audio_buffer
   │           (Q67 amended: default Python callback with NumPy
   │            buffer copy + atomic counter, GIL held < 50 µs/call;
   │            CFFI-callback path documented as escalation)
   │           (Q77: voxkit.audio.recorder is the only module
   │            permitted to import sounddevice — enforced by
   │            import-linter in CI)
   ▼
[ResamplerWorker]  (DEDICATED THREAD, MMCSS "Pro Audio")
   │            (per Q67, unchanged from v0.10)
   ▼
[ClickBleedHandler] ──▶ cleaned_audio
   │  ▲    (Q80 tracer bullet validates IR estimation in week 1;
   │  │     Q79 quality indicator metric: post-subtraction click
   │  │     residual ratio in dB)
   │  └─── re-estimation trigger
   ▼
[OnsetDetector] ──▶ onset_samples[]      ┐
   │                                     │
   ▼                                     │ Q76: ALL THREE
[EmbeddingExtractor] ──▶ embeddings[N×D] ├─ run on the dedicated
   │                                     │ InferenceWorker thread.
   │                                     │ Cancel via threading.Event
   │                                     │ checked between phases and
   ▼                                     │ per-onset; main thread runs
[Classifier] ──▶ (class_id, score)[]     │ Qt event loop only.
   │       ◀── [CalibrationManager]      ┘
   │            (Q81: corrected
   │            CalibrationRejected
   │            dialog wording)
   ▼
[TempoGridEngine] ──▶ quantized_events[]
   ▼
[MIDIExporter] ──▶ .mid file
   ▼
[Editor UI] ◀── [PlaybackEngine]
   │   (Q73 progress dialog now backed by InferenceWorker per Q76;
   │    cancel preserves recorded audio buffer)
   ▼
[ProjectFile (Q78: voxkit_format_version stamped on save;
   │           migration via registered (from,to) → migrator table)]
```

### Component 1: Project & Session

**Public API additions for v0.11:**

```python
@dataclass
class ProjectManifest:
    """v0.11 (Q78): explicit format version field."""
    voxkit_format_version: str   # e.g., "0.11"
    # ...all other v0.10 fields...

# Migration dispatch table (Q78)
Migrator = Callable[[dict], dict]
MIGRATIONS: dict[tuple[str, str], Migrator] = {
    ("0.4",  "0.5"):  migrate_0_4_to_0_5,
    ("0.5",  "0.6"):  migrate_0_5_to_0_6,
    # ...
    ("0.9",  "0.10"): migrate_0_9_to_0_10_cholesky,
    ("0.10", "0.11"): migrate_0_10_to_0_11_stamp_version,
}

def load_session(path: Path) -> Session:
    raw = read_bundle(path)
    manifest = raw.get("manifest", {})
    from_version = manifest.get("voxkit_format_version", "0.4")
    raw = walk_migrations(raw, from_version, target="0.11", table=MIGRATIONS)
    return parse_session(raw)
```

`migrate_0_10_to_0_11_stamp_version` is a no-op on data, only setting `voxkit_format_version: "0.11"` and (if missing) populating the field on legacy bundles. The Cholesky conversion remains in `migrate_0_9_to_0_10_cholesky` per v0.10's specification, now formalized as a registered table entry rather than implicit at-load behavior.

**Tests (additions on v0.10):**
- Migration table round-trip (§7.11): for each registered `(from, to)`, a synthetic bundle in the `from` schema is migrated, re-loaded, re-saved, and compared structurally to a fresh save of the same logical data.
- Legacy bundle (no `voxkit_format_version`) loads as `"0.4"` and walks all migrations.
- v0.10 bundle loads with one-step migration to v0.11.

### Component 2: Recording subsystem

**Public API additions for v0.11:**

```python
class Recorder:
    # All v0.10 attributes unchanged.

    AUDIO_CALLBACK_PATH: Literal["python_default", "cffi_hardened"] = "python_default"
    # Q67 amended (Q76 GIL contract). The active path is documented in the
    # docstring at the top of voxkit/audio/recorder.py and reflected in this
    # class attribute. v1.0 ships "python_default"; "cffi_hardened" is the
    # documented escalation path.
```

**Audio callback contract (Q67 amended with Q76 GIL guidance):**

```python
# Default path (v1.0):
def _python_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    # GIL-held time per call: < 50 µs typical at 5–10 ms buffers on reference CPU.
    # NumPy releases the GIL during the memcpy in __setitem__.
    if not ring.try_push_via_view(indata):
        # Atomic counter increment; no Python-level allocation.
        dropped_buffers.add(1)

# Hardened path (escalation, if measured drops require):
# Use sounddevice.RawInputStream with a CFFI-bound callback that does not
# acquire the GIL at all. Implementation reference: documented in
# voxkit/audio/cffi_callback.py if/when wired.
```

**Import-graph isolation (Q77):** `voxkit.audio.recorder` is the only module that may import `sounddevice` or any future platform audio library. Enforced via `import-linter` in CI.

**Tests (additions on v0.10):**
- Audio-callback no-allocation regression (carry from v0.10): `tracemalloc` against the audio thread on a 10-minute synthetic session; expected zero on-thread allocations after `open_stream`.
- GIL-held-time micro-benchmark on the reference CPU: median per-call GIL hold time < 100 µs at 5 ms buffer size (target < 50 µs but with headroom).
- `import-linter` integration test: synthetic violation (a stub `voxkit.core` file that imports `sounddevice`) reproduces a CI failure locally.

### Component 3: Click bleed handler

Per v0.10, plus v0.11 additions:

- **Click-bleed quality indicator metric (Q79)** computed and exposed via the public API:
  ```python
  def get_quality_attenuation_db(self) -> float:
      """Returns post-subtraction click residual ratio in dB.
      More positive = better attenuation.
      Range typically -3 (worse than no subtraction) to +35 (excellent).
      Mapped to UI: green ≥ 20, yellow 10–20, red < 10."""
  ```
- **Tracer bullet (Q80) acceptance criterion** validated in week 1: simplest LMS/Wiener-based IR estimation, 2-second adaptation, > 20 dB null on the click-only calibration on the reference setup with deliberately leaky open-back headphones.

Other Component 3 behavior unchanged from v0.9.

### Component 4: Onset detector

Unchanged from v0.10.

### Component 5: Embedding extractor

Unchanged from v0.10.

### Component 6: Classifier (composite gate, refined further)

Per v0.10, plus v0.11 additions:

- **CalibrationRejected dialog text (Q81):** see Q81 for full text. The diagnostic file entry now includes `f1_calibrated`, `f1_baseline`, and `delta`:
  ```python
  raise CalibrationRejected(
      message=Q81_DIALOG_TEXT,
      diagnostics={
          "f1_calibrated": f1_calibrated,
          "f1_baseline": f1_baseline,
          "delta": f1_calibrated - f1_baseline,
      },
  )
  ```
- The Editor catches `CalibrationRejected` and surfaces the dialog with the Q81 text; the diagnostics dict is logged to the local diagnostic file (Q61).

Other Component 6 behavior unchanged from v0.10.

### Component 7: Calibration manager

Unchanged from v0.10 except for the dialog wording correction propagated into Component 6.

### Component 8: Tempo & grid engine

Unchanged.

### Component 9: MIDI exporter

Unchanged from v0.10.

### Component 10: Playback engine

Unchanged.

### Component 11: Editor UI

Per v0.10, plus v0.11 additions:

- **InferenceWorker (Q76):** the recording-session progress dialog from Q73 is now backed by a dedicated `InferenceWorker` thread.
  ```python
  class InferenceWorker(QThread):
      phase_changed = Signal(str)              # "onset" | "embedding" | "classify"
      progress = Signal(float)                  # 0.0 to 1.0 within current phase
      completed = Signal(list)                  # list[Event] on success
      failed = Signal(str)                      # error message on failure
      cancelled = Signal()                      # emitted on user cancel

      def __init__(self, audio: np.ndarray, model: CalibratedModel) -> None: ...
      def run(self) -> None:
          # 1. Onset detection. Check self._cancel_flag before phase end.
          # 2. Embedding extraction. Check self._cancel_flag at top of each
          #    per-onset iteration. Worst-case cancel latency = one
          #    embedding extraction call (~50 ms reference).
          # 3. Classification. Check self._cancel_flag before phase end.
          ...
      def cancel(self) -> None:
          self._cancel_flag.set()
  ```
- **Main thread runs only the Qt event loop.** The progress dialog is a `QDialog` whose progress bar is updated by `phase_changed` and `progress` signals; the cancel button calls `worker.cancel()` and waits for the `cancelled` signal before dismissing the dialog.
- **On cancel:** the recorded audio buffer is preserved in the Session (no re-record needed); partial events are discarded; the editor returns to the post-recording state without events.

Other v0.10 editor behavior unchanged.

### Component 12: Eval harness (dev-only)

Per v0.10, plus v0.11 additions:

- **Synthetic-tier banner (Q85):** every invocation of the eval harness against the synthetic tier prints a banner:
  ```
  WARNING: Running against the synthetic tier. This validates pipeline
  runs only — NOT model quality. For quality validation, use --tier
  minimum-reproducible (PR validation) or --tier canonical (release).
  ```
  README repeats the same caveat; `--help` text repeats the same caveat in compact form.
- **Migration table round-trip test** (§7.11) added to the eval-harness CI invocation.

---

## End of document.

**v0.11 document graph for the build:**
- v0.11 changelog → tells the team what's new since v0.10.
- §5.1 + §11 Component 2 → Audio callback Python/GIL contract + import-graph isolation enforcement.
- §5.8 + §11 Component 11 → InferenceWorker threading; Q73 cancel semantics.
- §11 Component 1 → format version field, migration dispatch table.
- §11 Component 3 → click-bleed quality indicator metric + tracer-bullet acceptance.
- §11 Component 6 → CalibrationRejected wording.
- §6 → click-bleed risk row; six other new risk rows for the v0.11 changes.
- §9 → week 1 additions: CONTRIBUTING.md, `import-linter` and `reuse` CI, format-version field, license-memo seventh field, click-bleed tracer bullet, InferenceWorker scaffolding.
- §3 / §4.4 / Q84 → Phase 1.5 Linux re-framed to "as bandwidth permits."
- §7.10 / §7.11 / Q85 → synthetic-tier purpose explicit; new CI checks (import-linter, SPDX, migration round-trip).

**Critical-path additions for v0.11 (on top of v0.10 critical path):**
- CONTRIBUTING.md skeleton, `import-linter`, `reuse` CI, `voxkit_format_version` field, license-memo seventh field, click-bleed tracer bullet, InferenceWorker scaffolding — all week 1.
- InferenceWorker UI wiring, GIL-contract documentation block, CalibrationRejected wording, migration round-trip test — week 2.

End.
