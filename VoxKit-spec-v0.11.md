# VoxKit — Application Specification

**Document status:** Cohesive specification consolidating v0.3 through v0.11. Build-ready.
**Author:** Principal-engineer review (audio DSP / data science / software architecture / open-source maintainer / pragmatist perspectives).

**Project license:** GPL v3-or-later. All dependency choices in this specification are evaluated against GPL v3 compatibility. Two consequences worth surfacing up front: (a) Rubber Band's GPL v2-or-later licensing is compatible with GPL v3, so it is acceptable as a time-stretch fallback (`signalsmith-stretch` remains the primary on technical grounds, not licensing grounds); (b) any future Apache 2.0 dependency is compatible with GPL v3 but not GPL v2, so the "or-later" qualifier on the project license preserves that flexibility. Qt-side: both PyQt6 (GPL v3) and PySide6 (LGPL v3) are clean under this license; the implementer chooses.

**Project context:** VoxKit is a personal / hobby open-source project. The technical recommendations in this spec are sized for a properly-staffed v1.0 launch; a single-maintainer pace can sensibly relax some of the operational rigor without invalidating the architecture. Specifically, the formal user-impact study (Q64), the three-tier dataset infrastructure (Q63), and the Phase 1.5 commitment (Q62) are good practice but can be scaled down (a friends-and-family quality check, a single permissive dataset tier, and a "Linux when there's time" commitment respectively) without compromising the technical decisions. The architectural items — Cholesky storage, taxonomy parameterization, threading model, full-dim Mahalanobis, off-thread resampling, audio-callback GIL contract, import-graph isolation, inference-worker threading, click-bleed handler — are not optional in the same way; they're load-bearing for whether the product works at all.

---

## 0. How to read this document

This is a build-ready specification. The valuable sections are §5 (component specifications, high-level), §7 (test strategy and release gates), §11 (component specifications, build-ready), and §8 (multi-perspective review) for the audit trail.

The load-bearing technical commitments are:

1. **Click-bleed handler is the highest single technical risk** (§6, §11 Component 3). The bleed handler is the most uncertain DSP component; the week-1 tracer bullet (Q80) confirms the design holds before downstream components depend on it.
2. **Audio callback Python/GIL contract** (§5.1, §11 Component 2). Without explicit GIL-time guidance, a contributor will land a logging call in the callback and we'll see drops that aren't MMCSS's fault.
3. **Inference pipeline threading** (§5.8, §11 Component 11). Without a dedicated `InferenceWorker` thread, Q73's cancellable progress dialog is a lie.
4. **Platform isolation enforced via `import-linter`** (§9, §11 Component 2). Convention-only isolation does not survive a project's third contributor.
5. **Click-bleed FIR subtraction with mid-session re-estimation** (§5.2.1, §11 Component 3). The bleed path is not LTI on a session timescale.
6. **Full-dim Mahalanobis OOD detection** (§11 Component 6). PCA-64 silently disables OOD detection for samples differing along discarded directions.
7. **Two-tier onset metric** (§7.8). Detection F-measure alone hides timing-drift regressions invisible at IOU = 50 ms.
8. **Cholesky storage of pooled covariance** (§11 Components 1, 6). Half the size, better numerical stability.
9. **Parameterized class taxonomy** (§5.4, §11 Component 1). The v1.1 cost of un-hardcoding 4 classes is significantly larger than the v1.0 cost of doing it once.
10. **Architectural portability constraint** (§4.4, Q62). Linux/Mac portability is much cheaper to bake in at Phase 1 than to retrofit.

---

## 1. Executive summary

**VoxKit** is a Windows desktop tool (Phase 1; Linux ships in Phase 1.5 as bandwidth permits; macOS in Phase 2) for converting vocal percussion into MIDI drum patterns.

User flow: pick BPM and time signature → record a fixed-length take while a click plays in headphones → tool detects onsets, classifies each as kick/snare/closed_hat/open_hat (or unknown) using a frozen PANNs CNN14 (or BEATs) embedding plus a calibrated linear head, places them on a tempo grid → user audits and edits in a four-or-five-lane piano roll while looping the original audio for reference → exports a Standard MIDI File for FL Studio.

Three architectural commitments shape everything:

1. **Offline analysis, not realtime.** Record, then analyze. The calibration UI provides a near-realtime preview classifier, but full-signal analysis remains an offline batch step.
2. **Tempo is given, not inferred.** User-entered BPM, click in headphones, recording starts on bar 1 of the post-count-in buffer.
3. **Per-user calibration is mandatory before first analysis.** AVP-only models are mediocre on out-of-distribution voices. Calibration is the leverage point.

Additional product-shape commitments:

- Frozen PANNs CNN14 embedding (2048-dim, ONNX) or BEATs embedding (768-dim, ONNX) — substrate locked in Phase 1 week 2 — plus a logistic-regression head with temperature-scaled softmax output.
- 4 trained classes (kick, snare, closed_hat, open_hat) plus a runtime "unknown" class routed via a composite gate (softmax confidence threshold + full-dim Mahalanobis distance to nearest class centroid).
- Mandatory headphone-bleed setup at first run with FIR-based subtraction; mid-session re-estimation from active and passive silent windows.
- OOD-validated on ≥ 15 subjects with bootstrap CIs on the AVP-vs-OOD gap.
- GPL v3-or-later project license; local-file diagnostics only (no network telemetry in v1.x).

Provisional accuracy targets, validated empirically before release: 85–92% LOSO macro-F1 uncalibrated with the PANNs embedding; 92–96% with calibration. The 95.55% figure from Sinyor 2005 is on a different (smaller, single-source) dataset and does not transfer; the AVP cross-subject baseline is set empirically on first model train.


---

## 2. Resolved decisions

| # | Decision | Resolution |
|---|---|---|
| Q1 | Realtime vs offline | Offline. |
| Q2 | How tempo is known | User-entered BPM + click-in-headphones during recording. |
| Q3 | Where bar 1 starts | t=0 of the recording buffer, post count-in. |
| Q4 | Time signatures | 4/4 and 3/4. |
| Q5 | Quantization grid | 1/8, 1/16, 1/32, 1/16T, 1/8T. Default 1/16. |
| Q6 | Quantization strength | Slider 0–100%, default 100%. Original timestamps preserved. |
| Q7 | Class set | 4 trained classes (kick, snare, closed_hat, open_hat) plus runtime "unknown". |
| Q8 | Playback sound source | Four bundled WAV samples (one per trained class). |
| Q9 | Variable playback speed | 50%–200% with pitch preservation via `signalsmith-stretch` (Rubber Band fallback if A/B quality test rejects), debounced. |
| Q10 | Platform | Windows 10/11 (x64) for Phase 1; Linux in Phase 1.5 as bandwidth permits; macOS in Phase 2. |
| Q11 | Manual swing adjustment | Free-drag MIDI events on a piano roll, with original waveform shown beneath. Snap toggleable. |
| Q12 | Recording length | Fixed: 1, 2, 4, or 8 bars. Mandatory count-in. |
| Q13 | Calibration | Mandatory before first analysis. 3 examples per class minimum; "more samples = better results" hint between 3 and 10. |
| Q14 | Training dataset | AVP-LVT v4 (Zenodo 5036529 / 5578744), CC-BY-4.0. Personal-imitation modality preferred over fixed-imitation. |
| Q15 | Classifier | PANNs CNN14 frozen embedding (or BEATs) via ONNX Runtime + scikit-learn `LogisticRegression` (multinomial, L2) head with temperature scaling on softmax. AdaBoost+CART, SVM-RBF, raw-mel-spec linear head retained as dev-only benchmarks. |
| Q16 | Eval protocol | Leave-One-Subject-Out on AVP, 28 folds. Stratified k-fold retained as sanity check only. |
| Q17 | Click bleed handling | FIR subtraction (32–128 taps; default 64) of the click signal, with bleed-impulse-response measured during the headphone-bleed setup step. |
| Q18 | Embedding model distribution | Bundle the CNN14 16 kHz checkpoint exported to ONNX (~80 MB ONNX, ~150 MB total runtime including ONNX Runtime). Ship under MIT + AudioSet attribution. |
| Q19 | Onset-detection release gate | F-measure ≥ 0.92 on AVP at IOU = 50 ms; F-measure ≥ 0.88 on OOD at the same IOU. Median absolute timing error on true positives ≤ 15 ms AVP / ≤ 25 ms OOD. Both detection and alignment tiers must hold (Q70). Numbers provisional pending Q64 justification memo. |
| Q20 | Class taxonomy granularity | 4 trained classes (kick, snare, closed_hat, open_hat). Drum map 36 / 38 / 42 / 46. |
| Q21 | Embedding pre-processing | No pre-emphasis filter. Pad short windows with surrounding audio context; zero-pad only at buffer boundaries (within 800 ms of recording start/end). |
| Q22 | OOD validation | ≥ 15 subjects from outside AVP-LVT v4. Bootstrap 95% CI reported on the OOD-vs-AVP-LOSO macro-F1 gap (subject-level resampling, 1000 iterations). Pass criteria per Q50. |
| Q23 | Bleed-gate override | User may proceed past the bleed-gate refusal with explicit acknowledgment; flag persists in session metadata. |
| Q24 | Audio device support | Wired USB, built-in, and audio-interface devices supported. Bluetooth devices refused at the device picker with explanatory text. |
| Q25 | Mid-session bleed re-estimation | Triggered by elapsed-bar count (every 32 bars) or by click-window guard firing > 2% of click positions in the most recent 8 bars. Uses the silent-window rolling buffer per Q35 (active + passive detection per Q47/Q48). |
| Q26 | Calibration refit weighting | `class_weight=None` in the user refit; calibration weight (default per Q42 sensitivity study) is the only weighting mechanism. `class_weight='balanced'` still applied to the base AVP fit at build time. |
| Q27 | Output calibration | Temperature scaling on the softmax. Single scalar T fit by NLL minimization on a held-out validation fold disjoint from LR training and Mahalanobis covariance estimation data; applied as `softmax(logits / T)`. UI label is "confidence"; API field is `score`. |
| Q28 | `commit()` semantics | `CalibrationManager.commit()` returns a `CommitHandle` with `cancel()`, `wait()`, `status`, `error` properties. Newer commits cancel in-flight predecessors. Last-write-wins. |
| Q29 | v0.4 session migration | Up-convert metadata; prompt for fresh bleed setup on first open. Migrated sessions tagged as `bleed_ir_origin: "migrated_pending_recapture"` until re-estimated. Banner persistent until re-setup. |
| Q30 | First-run calibration minimum | 3 per class (12 total) minimum to commit. Drop-off metric logged to local diagnostic file. |
| Q31 | Taxonomy disclosure | Explicit user-facing text in calibration screen header and editor first-open banner; reclassify-on-drag tooltip on first drag of each session. |
| Q32 | Phase 1 platform support | Windows 10/11 (x64) only. Mac and Linux on roadmap per Q40 / Q62. |
| Q33 | Embedding substrate decision | PANNs CNN14 or BEATs, decided in Phase 1 week 2 on AVP smoke-tier LOSO macro-F1. Tiebreaker (per Q74): substrate wins outright if 95% bootstrap CI of AVP-LOSO macro-F1 (1000 resamples over LOSO folds) does not overlap with the other substrate's; otherwise pilot OOD (5 subjects) is the tiebreaker. |
| Q34 | Unknown-class detection | Composite gate: softmax confidence threshold (default 0.45) OR full-dim Mahalanobis distance to nearest class centroid exceeds per-class 95th-percentile threshold. Both thresholds eval-swept per §7.3. Mahalanobis runs on the full-dim embedding (never PCA-projected). |
| Q35 | Mid-session re-estimation input | Silent windows only. A window qualifies as actively silent if post-click RMS (excluding the click itself) is below noise_floor + 6 dB. If a trigger fires but fewer than 8 silent windows are available in the recent 16 bars, the user is prompted for a 4-bar silent re-capture. |
| Q36 | OOD power sensitivity analysis | One-page memo (week 1, ~half day) documenting variance assumption, MDG, and N recommendation. Drives §7.9 sample size and pass criterion. |
| Q37 | Audio device disconnect | Typed `DeviceDisconnected` exception with `device_id` and `last_good_sample_index`. Recording paused; blocking modal: "Audio device disconnected. Recording paused at bar X. Reconnect the device or select a different one." [Reconnect/Select] [Save and exit]. |
| Q38 | OS sleep mid-session | Typed `OSSleepEvent` raised on platform sleep notification. Recording state paused. On wake, device re-validation runs; non-blocking toast: "Device validated after sleep. Bleed setup may have changed (volume, headphone position). [Re-run bleed setup] [Continue]." |
| Q39 | `bleed_ir_history` cap | FIFO cap at 20 entries; two protected slots per Q49 (most recent setup-origin + most recent active-silent re-estimation). Both exempt from FIFO eviction. |
| Q40 | Phase 1 platform deliverable | Windows-first by week 4. Linux in Phase 1.5 as bandwidth permits. macOS in Phase 2. |
| Q41 | Network telemetry posture | Per Q61: local-file diagnostics only in v1.x. Reintroduction requires explicit opt-in with data preview, build-time disable flag, and correct operation under no-outbound-network configuration. |
| Q42 | Calibration weighting | Default selected from sweep `{1, 5, 25, 50, 125, 625}` (geometric with 50× probe). Per Q65 sensitivity study, default weight is the largest weight at which `drift(noisy)/drift(clean) < 2.0` across noise levels σ ∈ {0.1, 0.5, 1.0} × per-feature std on synthetic noisy calibration. Default provisional pending the sensitivity study. |
| Q43 | PCA-64 user refit | LR head can optionally project full-dim embedding through a learned PCA-64 matrix. Bake-off in week 2 decides; PCA-64 ships only if it does not regress LOSO macro-F1 by > 0.5 points. Mahalanobis is excluded from PCA: it always runs on full-dim embeddings. Q57 PCA-64 per-class Mahalanobis sweep is closed (Q69 cuts it from v1.0). |
| Q44 | Distribution-shift telemetry | If user's first-100-events median score < 0.5, the orchestrator surfaces a non-blocking recalibration prompt. Once per session if condition met. |
| Q45 | Distribution-shift warning threshold | AVP-derived: median score on AVP held-out folds × 0.7 (i.e., a 30% drop in median score relative to AVP). Stored in model bundle. |
| Q46 | Time-stretch library | `signalsmith-stretch` (MIT) is the primary on technical grounds (smaller dependency footprint, no second time-stretch implementation to maintain, code-base homogeneity). Rubber Band (GPL v2-or-later, acceptable under GPL v3-or-later project licensing) is the fallback if `signalsmith-stretch` fails the Phase 1 week-1 audible-quality A/B test on a 10-clip set. |
| Q47 | Active silent-window detection | A click-aligned window is "actively silent" if its post-click RMS (excluding ±15 ms around the click position) is below `noise_floor + 6 dB`. |
| Q48 | Passive silent-window detection | A click-aligned window is "passively silent" if a run of ≥ 4 consecutive active-silent windows precedes it (~200 ms of consecutive silence at the click cadence). The passive sliding window is over 8 click-aligned windows of 50 ms each. |
| Q49 | `bleed_ir_history` protected slots | Two protected slots: most recent setup-origin entry + most recent active-silent re-estimation entry. Both are exempt from FIFO eviction. |
| Q50 | OOD release-gate criterion | (a) Missed-unknown-rate on full OOD ≤ 25% (i.e., ≥ 75% of OOD events correctly routed to unknown). (b) False-unknown-rate on AVP held-out ≤ 5% (i.e., ≤ 5% of in-distribution events wrongly routed to unknown). Both must hold simultaneously at the operating-point `(softmax_threshold, distance_percentile)` pair selected by the §7.3 sweep. |
| Q51 | Resampler thread placement | Resampler runs on a dedicated worker thread. Audio callback only copies device buffers into a lock-free SPSC ring buffer. Filter state pre-allocated at picker time. Worker latency budget: 10 ms (alarm if exceeded for > 100 consecutive buffers). |
| Q52 | Mahalanobis covariance source under weighted calibration | Centroids: AVP + weighted calibration data. Pooled covariance and per-class distance thresholds: AVP only, unweighted. |
| Q53 | Onset-detection release gate (numeric) | F-measure ≥ 0.92 on AVP at IOU = 50 ms; F-measure ≥ 0.88 on OOD at the same IOU. Both must hold. (See also Q70 for the alignment-MAE tier.) |
| Q54 | First-run guided tour trigger | First unknown event in the first session that contains any unknowns. Once per user (not per-session), dismissable, never repeats after dismissal. |
| Q55 | Migration sunset cadence | Two-minor-version sunset: v1.0 marks v0.4 deprecated (load works, save warns); v1.2 removes v0.4 load. Each subsequent minor sunsets the oldest still-supported format. |
| Q56 | Telemetry deletion SLA | Moot for v1.0 — no server-side telemetry exists to delete (per Q61). SLA preserved as forward-looking commitment if v1.x reintroduces network telemetry: 72 hours from user-initiated deletion request, batched twice daily. |
| Q57 | Pooled vs per-class Mahalanobis under PCA-64 | Closed in v0.10 per Q69. Mahalanobis ships full-dim pooled. |
| Q58 | Audio API selection | WASAPI default on Windows; MME fallback if WASAPI initialization fails (with a one-time UI notification). ASIO not pursued in v1.0. Linux: PipeWire preferred, ALSA fallback (Phase 1.5). macOS: CoreAudio (Phase 2). |
| Q59 | Project license | GPL v3-or-later. Aligns with hobby/community OSS posture; compatible with the broader audio-ecosystem dependency set (Rubber Band, PyQt); "or-later" preserves Apache 2.0 dependency compatibility. |
| Q60 | License-review scope | Per dependency, week-1 review produces a one-page memo: (a) code license, (b) artifact/model-weight license, (c) commercial-use restrictions on weights, (d) fine-tuning rights, (e) attribution requirements, (f) redistribution requirements, (g) pre-training data license propagation — if the model was pre-trained on a dataset with usage restrictions (e.g., AudioSet's non-commercial-research framing), do those restrictions propagate to the released model weights for downstream GPL v3 redistribution? PANNs and BEATs memos must answer (g) explicitly. Criterion is GPL v3 compatibility. |
| Q61 | Telemetry posture | Local diagnostic file only in v1.x (`~/.voxkit/diagnostics/<session-id>.jsonl`). No network beacons. Manual user-submitted upload to GitHub issue if needed. Diagnostic file rotation: 30-day TTL on disk; user-configurable or fully disable-able. "Opt-in network telemetry" removed from the v1.x roadmap. |
| Q62 | Architectural portability | Phase 1 implementation keeps platform-specific code behind `Recorder` (audio I/O) and a sleep-handler abstraction. No platform-specific code elsewhere. Linux smoke CI runs from week 1 on platform-independent code. Linux build ships in Phase 1.5 as bandwidth permits; Mac in Phase 2. Enforced via `import-linter` per Q77. |
| Q63 | Eval-dataset access for OSS contributors | Three-tier dataset plan: (a) canonical (full AVP + 15-subject OOD, project-hosted on Zenodo IF redistribution rights confirmed; otherwise download-script), (b) minimum-reproducible (10–20 subjects per fold, always project-hosted, permissively licensed), (c) synthetic (procedurally generated, always in repo). CI on synthetic; PR validation on minimum-reproducible; release validation on canonical. |
| Q64 | Release-gate empirical justification | Each numeric release-gate threshold (25% missed-unknown, 5% false-unknown, 0.92/0.88 onset F-measure, 15/25 ms onset MAE) supported by a one-page memo before v1.0 lock: user-impact study findings (n ≥ 5 musicians) and/or published-literature ceilings. Numbers provisional until memos land. (Hobby pace: a "I tested with 2 friends" memo is acceptable in lieu of formal study.) |
| Q65 | Calibration-weight target empirical justification | Per Q42: default weight selected such that LR coefficient drift on deliberately noisy synthetic calibration data is < 2× the drift on clean calibration. The "8–12% per class" target from earlier versions is removed; it was an unjustified anchor. |
| Q66 | Class taxonomy parameterization | Internal architecture parameterized via `TaxonomyConfig` (default = 4 classes). User-facing surface unchanged in v1.0 (4 trained + 1 runtime unknown). v1.1 throat-bass becomes a config change rather than a migration. UI taxonomy editing is a v1.1 candidate. |
| Q67 | Audio thread sample-format and threading specifics | Audio callback receives float32 buffers, mono after device-side downmix, native byte order. Callback's only operations are `ring.try_push(buf)` and atomic increment of `dropped_buffers` on push failure. No allocation, no logging, no system calls. Worker thread MMCSS "Pro Audio" on Windows (`AvSetMmThreadCharacteristicsW`); `SCHED_FIFO` priority 80 on Linux (Phase 1.5, with `setcap` install docs). Drop policy: session-scope counter; > 0.1% over rolling 30-second window triggers end-of-session warning modal; diagnostic file always records the count. Latency budget: 10 ms per buffer at typical 5–10 ms buffer size; < 3 ms buffers budget relaxes to 3× buffer duration; > 30 ms buffers tightens to 1.5×. **GIL contract:** default path is Python callback with NumPy-buffer copy and atomic counter (GIL held < 50 µs/call typical at 5–10 ms buffers on reference CPU); hardened path is sounddevice `RawInputStream` with CFFI-level callback when default path measurably drops buffers under load. v1.0 ships the Python-callback path; CFFI is the documented escalation path. The active path is reflected in the `Recorder.AUDIO_CALLBACK_PATH` class attribute and a docstring at the top of `voxkit/audio/recorder.py`. |
| Q68 | Covariance storage format | Cholesky factor of (Ledoit-Wolf-shrunk) covariance, stored as lower-triangular `D_full × (D_full + 1) / 2` floats. Distance via `solve_triangular(L, x − µ)`; no explicit inverse stored. ~50% storage savings, better numerical stability. |
| Q69 | PCA-64 per-class Mahalanobis sweep | Cut from v1.0 (closes Q57). Mahalanobis ships full-dim pooled. |
| Q70 | Two-tier onset metric | Detection F-measure (Q53) AND alignment quality (median absolute timing error on true positives ≤ 15 ms AVP / ≤ 25 ms OOD). Both must pass. Numeric values provisional pending Q64 latency-budget memo. |
| Q71 | Self-test overfit guard | After every `fit_with_calibration`, the guard computes LOSO macro-F1 on AVP held-out folds (excluding subjects whose data was used in calibration). If macro-F1 has dropped > 1 point relative to the no-calibration baseline at the same `calibration_weight`, the calibration is rejected: previous calibration restored, recovery dialog shown per Q81. Guard runs on every `fit_with_calibration` call. |
| Q72 | CPU performance target | End-to-end inference on a 32-bar / 120 BPM session (≈ 16 s of audio at 16 kHz, ~64 onsets) completes in ≤ 8 wall-clock seconds (≤ 0.5× audio duration) on a 2018 mid-tier laptop CPU (reference: Intel i5-8250U, 4 cores, no GPU). Substrate decision (Q33) reports this number; if neither substrate hits it, ONNX optimization passes (graph optimization, INT8 quantization) come before substrate change. |
| Q73 | Recording-session inference UX | During recording: live waveform + click-bleed quality indicator (numeric percentage + colored bar per Q79) + recording duration + bar count if click track is set. No live classification. On stop: modal progress dialog with three named phases — "Detecting onsets" → "Extracting embeddings" → "Classifying events" — each with a percentage. Cancel button preserves audio and returns to recording mode. Total wall-clock target per Q72. After completion: user lands in editor with all events placed and labeled. |
| Q74 | Substrate tiebreaker reformulation | See Q33: overlapping bootstrap CIs replace the fixed 5-point margin. |
| Q75 | Temperature-scaling held-out source | During LOSO eval, T fit on a leave-out subject inside the training fold (separate from the LOSO-held-out subject). At user calibration time on a shipped model, T re-fit on a held-out 20% slice of AVP not used in LR training or Mahalanobis covariance estimation. T never fit on user calibration data. |
| Q76 | Inference pipeline threading | Onset detection, embedding extraction, and classification run on a dedicated `InferenceWorker` thread. The main thread runs the Qt event loop only. Cancellation via thread-safe `threading.Event`; worst-case cancel latency is one embedding-extraction call (~50 ms reference). On cancel, the recorded audio buffer is preserved in the Session; partial events are discarded. |
| Q77 | Platform-isolation enforcement | `import-linter` configuration in `.importlinter` committed to repo with rules: (a) `voxkit.audio.recorder` is the only module permitted to import platform audio libraries; (b) `voxkit.core`, `voxkit.classifier`, `voxkit.eval`, `voxkit.ui` may not import any platform-specific module by name; (c) `voxkit.core` may not import `voxkit.ui`. CI runs `lint-imports` on every PR; violations block merge. |
| Q78 | Format version field | `manifest.json` adds `voxkit_format_version: str` at top level. Migration is dispatched via a registered table of `(from_version, to_version) -> migrator`. Bundles missing the field are treated as `"0.4"`. |
| Q79 | Click-bleed quality indicator metric | Post-subtraction click residual ratio in dB: `20 * log10(rms(cleaned_audio[click_aligned_windows]) / rms(click_calibration[click_aligned_windows]))`. Mapped to color: green ≥ 20 dB, yellow 10–20 dB, red < 10 dB. Below 10 dB triggers the bleed banner. Numeric value shown alongside the bar. |
| Q80 | Click-bleed handler tracer bullet | Week 1 task: implement the simplest LMS / Wiener-based IR estimation against a 2-second click-only calibration recording on the maintainer's own setup with deliberately leaky open-back headphones. Acceptance: > 20 dB null on the click-only segment after 2 seconds of adaptation. If the simplest approach fails, escalate to NLMS / RLS before continuing with downstream components. |
| Q81 | CalibrationRejected dialog text | "VoxKit's accuracy check found that the most recent calibration didn't improve classification on the held-out test set; the previous calibration has been restored. This usually means the calibration recording was very different from your typical use (very few samples, unusual background noise, or a different microphone), and the model would have generalized worse with it. You can try again with more or quieter samples, or continue using the previous calibration." Diagnostic file records the macro-F1 delta. |
| Q82 | SPDX header enforcement | Pre-commit hook plus CI job runs `reuse` on every PR; missing or wrong SPDX header on a tracked source file blocks merge. `.reuse/dep5` lists exempted generated files. |
| Q83 | CONTRIBUTING.md | Skeleton committed week 1 covering environment setup, synthetic-tier CI invocation, dataset acquisition (the project-hosted minimum-reproducible subset per Q63), license expectations including SPDX header on every new source file, PR checklist (tests pass, `lint-imports` clean, no SPDX-header drift), and where to ask questions (GitHub Discussions, not Issues). |
| Q84 | Phase 1.5 Linux framing | "Ships as bandwidth permits, with the architectural constraint of Q62 / Q77 ensuring it remains tractable when that time comes." Architectural lock-in is the deliverable; the calendar slot is honestly conditional. |
| Q85 | Synthetic dataset role | Synthetic tier validates pipeline runs, imports, Cholesky round-trip, MIDI parseability, and eval-harness JSON shape. Does NOT validate model quality. Quality validation begins at the minimum-reproducible tier (PR validation) and is final at the canonical tier (release validation). README and eval-harness `--help` both repeat this caveat. |


---

## 3. Scope

### 3.1 In-scope (v1.0)

Recording with metronome + count-in + headphone attestation + clipping guard + Bluetooth-device exclusion at picker time + device-disconnect and OS-sleep handlers; FIR-based click-bleed setup with mid-session re-estimation; onset detection with sensitivity slider, noise-floor gating, and click-window guard; PANNs CNN14 (or BEATs) frozen embedding + temperature-scaled logistic-regression head + composite Mahalanobis-distance-based unknown-class gate; mandatory per-user calibration with live preview (after ≥1 sample per class), self-test overfit guard, and CalibrationRejected recovery; quantization with grid + strength; four-or-five-lane piano-roll editor with drag-to-retime, drag-to-reclassify, undo/redo, taxonomy disclosure text, and recording-session progress UX; multi-stem playback engine with mute/solo and variable-speed (pitch-preserved); MIDI export with configurable drum map (default 4 trained classes; opt-in unknown export); project file with v0.4 → v0.11 migration registry.

### 3.2 Out of scope (v1.0)

Realtime classification; tempo inference; non-percussion vocal sound classification; user-facing class taxonomy editing UI; Mac and Linux platform-native audio (Phase 1.5 / Phase 2); Bluetooth audio devices (Phase 2 prototype per §10 item 17); ASIO driver support; opt-in network telemetry; throat-bass / bass-music as a 6th trained class; per-user `softmax_threshold` / `distance_thresholds` adjustability UI; embedding-space nearest-neighbor browser as debugging aid.

### 3.3 Explicit non-goals

VoxKit is not trying to beat the AVP 2022 CNN paper. A small CNN on log-mel patches (or fine-tuned final PANNs layers) is plausibly v1.1 or v2.0 work; v1.0 ships the frozen embedding + linear head architecture.

---

## 4. Architecture

### 4.1 Top-level shape

```
[UI: Qt6 (PyQt6 or PySide6)]
        │
        ▼
[Orchestrator: Python; threads owned by Recorder, ResamplerWorker, InferenceWorker]
        │
        ▼
[Pipeline: Recorder → ResamplerWorker → ClickBleedHandler → OnsetDetector
           → EmbeddingExtractor → Classifier → TempoGridEngine → MIDIExporter]
        │
        ▼
[ProjectStore: zip-bundle .vxk files with v0.4 → v0.11 migration]
        │
        ▼
[Disk + local-only diagnostics file]
```

Single audio callback for record + click playback (recorder); single audio callback for playback + drum stems (playback engine). No separate streams that could drift and cause flam.

### 4.2 Stack

| Layer | Choice | Why |
|---|---|---|
| Project license | GPL v3-or-later (Q59) | Aligns with hobby/community OSS posture; compatible with the broader audio-ecosystem dependency set (Rubber Band, PyQt); "or-later" preserves Apache 2.0 dependency compatibility. |
| Language | Python 3.11+ | Audio + ML libs first-class. |
| UI | Qt6 via PyQt6 (GPL v3) or PySide6 (LGPL v3) — implementer's choice | Both clean under GPL v3 project license. Either works. Windows-native look, good custom-paint perf. |
| Audio I/O | `sounddevice` with WASAPI default, MME fallback on Windows; ALSA / PipeWire on Linux (Phase 1.5); CoreAudio on macOS (Phase 2) — abstracted behind `Recorder` per Q62, enforced via `import-linter` per Q77. Audio callback only does ring-buffer push per Q67 (Q67 amended adds Q76 GIL contract: default Python-callback path; CFFI-callback path documented as escalation). | Bluetooth excluded at picker time (Q24). WASAPI gives shared/exclusive mode flexibility. ASIO not pursued in v1.0. |
| Inference threading | `InferenceWorker` thread executing onset + embedding + classification; main thread is Qt event loop only (Q76). | Q73's cancellable progress dialog requires this. |
| Resampling | `scipy.signal.resample_poly` on a worker thread with pre-allocated filter state and SPSC ring-buffer feed (Q51). MMCSS "Pro Audio" on Windows; `SCHED_FIFO` priority 80 on Linux. | Off-thread eliminates audio-callback risk on irrational ratios like 44.1 → 16. |
| Onset detection | `librosa.onset` + custom click-aware preprocessing (FIR-subtracted, periodically re-estimated from active + passive silent windows) | Two-tier release gate per Q70 (detection F-measure + alignment MAE). |
| Embedding model (runtime) | PANNs CNN14 ONNX OR BEATs ONNX, decided in Phase 1 week 2 on AVP smoke-tier with overlapping-CI tiebreaker per Q33/Q74 | ~150 MB total runtime (~80 MB ONNX + ONNX Runtime). MIT-licensed code; weight licensing verified per Q60. |
| Embedding model (dev/training) | PyTorch + PANNs CNN14 checkpoint | Used in eval harness and to produce the ONNX export at build time. Not shipped. |
| OOD detection | Mahalanobis on full-dim embeddings (always, never PCA-projected) per Q34/Q43, distance to nearest centroid per Q34; pooled covariance with Ledoit-Wolf shrinkage stored as Cholesky factor (Q68); per-class 95th-percentile thresholds | PCA discards exactly the directions OOD lives in; OOD detection must run pre-projection. |
| Classifier head | scikit-learn `LogisticRegression(penalty='l2', C=1.0, multi_class='multinomial', max_iter=1000)` over either full-dim or PCA-64 embedding (Phase 1 week 2 decision per Q43). Output: temperature-scaled softmax with T fit on disjoint held-out fold per Q75. | Linear head over frozen embedding is already linearly separable for most audio classification tasks. |
| Class taxonomy | `TaxonomyConfig`-driven (Q66), default = 4 trained classes (kick, snare, closed_hat, open_hat) + runtime "unknown". v1.0 ships default; v1.1 throat-bass adds a class via config | Internal flexibility, external stability. |
| Time-stretch | `signalsmith-stretch` (MIT) primary; Rubber Band (GPL v2-or-later) acceptable fallback under GPL v3-or-later project licensing | Q46. Primary on technical grounds (lighter dep, code-base homogeneity). |
| MIDI | `pretty_midi` | FL Studio reads its output cleanly. Verified. |
| Telemetry | Local diagnostic file only (`~/.voxkit/diagnostics/<session-id>.jsonl`); no network in v1.x (Q61) | OSS posture independent of license. |
| Inference performance | Target: ≤ 0.5× wall-clock on 2018 mid-tier laptop CPU for a 32-bar / 120 BPM session (Q72) | Concrete target gates substrate decision. |
| Packaging | PyInstaller for Windows; AppImage + Flatpak for Linux (Phase 1.5); macOS code signing + notarization (Phase 2) | Installer ~150 MB. SmartScreen warning expected on unsigned builds; documented. |
| Platform isolation | Enforced via `import-linter` in CI per Q77; not convention | Q62 lock-in only survives if it's a check, not a convention. |
| Testing | `pytest` + golden audio fixtures + LOSO eval + tiered eval harness (smoke/full per §7.10) | See §7. |

### 4.3 Why offline

Realtime classification has three structural problems for this product: (a) the latency budget for click-aligned recording (sub-15 ms onset alignment per Q70's MAE tier) is incompatible with the wall-clock cost of embedding extraction at typical buffer sizes; (b) the user is going to want to scrub, retime, and reclassify after the fact, so a realtime classification commits work that gets thrown away; (c) the classifier output is batch-dependent (RMS percentile bounds for velocity, calibration uplift, post-temperature-scaled probabilities) — quantities that aren't well-defined in a realtime stream.

Offline analysis runs once after recording stops, with a progress dialog (Q73), and produces all events at once with consistent global statistics. The calibration UI uses a near-realtime preview classifier on individual sample utterances (refit time ~150–250 ms), so the realtime affordance is preserved where it matters: during the user's active feedback loop on calibration samples.

### 4.4 Why Windows-only in Phase 1 (and Linux in Phase 1.5)

Phase 1 Windows-first reflects (a) audio-driver maturity for the consumer-grade USB/built-in mic class targeted, (b) MIDI ecosystem (FL Studio is the priority test DAW, Windows-native), and (c) team-velocity-on-a-single-platform considerations. **Q62's architectural portability constraint is the v1.0 acknowledgment that Phase 1.5 — Linux ship — is non-negotiable for an OSS project's contributor base** (per Q84, "ships as bandwidth permits"; the architectural lock-in via Q62 + Q77 ensures the build is a wiring exercise rather than a refactor). macOS Phase 2 because CoreAudio is a separate learning curve and the Mac musician audience is well-served by existing commercial tools in this category. Linux is prioritized over Mac because OSS contributor distribution skews Linux.

Phase 1.5 is an *architecturally enabled, calendar-flexible* phase. The contents of the Linux build (ALSA / PipeWire backend, `SCHED_FIFO` install docs, AppImage / Flatpak packaging, Linux device test matrix) are unchanged in shape; only the calendar wording is softened.


---

## 5. Component specifications (high-level)

### 5.1 Recorder

The Recorder captures audio while playing a click in headphones. Bar 1 starts at `t=0` of the returned buffer, post-count-in.

- **API surface (Q58):** WASAPI is the default capture API on Windows. If WASAPI initialization fails, the recorder falls back to MME with a one-time UI notification ("Using compatibility audio mode for this device — latency may be slightly higher"). On Linux (Phase 1.5), PipeWire is preferred with ALSA fallback. On macOS (Phase 2), CoreAudio.
- **Audio callback contract (Q67, Q76 GIL guidance):** The audio callback's only job is to copy device buffers into a lock-free SPSC ring buffer (capacity: 2 seconds of device-rate audio) and atomically increment a `dropped_buffers` counter on push failure. Float32 buffers, mono after device-side downmix (sounddevice handles channel mix-down at `Stream` open time), native byte order. No allocation, no logging, no system calls. **Default path (v1.0):** Python callback that operates only on a pre-allocated NumPy buffer copy plus an atomic counter; NumPy's array operations release the GIL for the copy itself. Total GIL-held time at typical 5–10 ms buffers is < 50 µs on the reference CPU. **Hardened path (escalation if measured drops require):** sounddevice `RawInputStream` with a CFFI-level callback that does not acquire the GIL at all. The active path is reflected in the `Recorder.AUDIO_CALLBACK_PATH` class attribute and a docstring at the top of `voxkit/audio/recorder.py`.
- **Resampler thread (Q51):** A dedicated worker thread reads from the ring buffer, applies `scipy.signal.resample_poly` with pre-allocated filter state (allocated at device-picker time), and pushes 16 kHz buffers to the downstream pipeline. Worker thread registered with Windows MMCSS as "Pro Audio" task class (`AvSetMmThreadCharacteristicsW`). On Linux (Phase 1.5), `SCHED_FIFO` priority 80; install docs include `setcap` instructions.
- **Latency budget (Q67):** 10 ms per buffer at typical 5–10 ms buffer size. For very small buffers (< 3 ms), budget relaxes to 3× buffer duration. For large buffers (> 30 ms), tightens to 1.5× buffer duration. Alarm fires if budget exceeded for > 100 consecutive buffers.
- **Drop policy (Q67):** `dropped_buffers` counter exposed to the session-tracking layer. If rolling 30-second drop rate exceeds 0.1%, end-of-session modal warns the user. Diagnostic file always records the count.
- **Click playback:** One sounddevice callback simultaneously fills input buffer and reads from a pre-rendered click buffer to send to output. Single callback prevents inter-stream drift. Click rendered ahead of time at session sample rate from a 1 kHz tone burst, 30 ms long, with -3 dBFS peak. Count-in clicks written into the click buffer; the corresponding samples in the input buffer are discarded before return.
- **Clipping guard:** If any input sample exceeds 0.95 in magnitude, log a warning event; the UI surfaces this after recording completes.
- **Device picker (Q24):** Bluetooth detection on Windows uses the WASAPI device-class enumeration. Excluded devices appear grayed-out with hover text: "Bluetooth audio adds 100–300 ms of round-trip latency that VoxKit cannot compensate for. Please use a wired headphone connection." A `--allow-bluetooth` CLI flag enables the dev/debug path.
- **Disconnect handler (Q37):** On `AUDCLNT_E_DEVICE_INVALIDATED` HRESULT, the handler captures `last_good_sample_index`, flushes pending audio to the session buffer, raises `DeviceDisconnected(device_id, last_good_sample_index)`. The recording session is preserved.
- **Sleep handler (Q38):** On Windows, register a hidden message-only window via `pywin32` to receive `WM_POWERBROADCAST`. On `PBT_APMSUSPEND`, invoke the registered callback. On wake, the orchestrator re-enumerates devices and validates that the previously-active device is still present.
- **Cross-platform isolation (Q62, Q77):** `Recorder` is the single platform-abstraction point; enforced via `import-linter`.

### 5.2 Onset detector

Three-stage pipeline:

1. **Click subtraction.** `recording -= conv(click_pulse_train, h)` where `h` is the per-device bleed impulse response estimated at setup (§5.2.1). Subtraction is sample-exact at the click-position level.
2. **Onset detection.** `librosa.onset.onset_detect` on the click-subtracted signal, with vocal-percussion-tuned defaults: `units='samples'`, `pre_max=10ms`, `post_max=10ms`, `pre_avg=80ms`, `post_avg=80ms`, `delta` exposed as the user-facing sensitivity slider.
3. **Click-window guard.** Suppress any onset within ±15 ms of a known click position. A high firing rate here is a signal that the FIR estimate is stale.

Auxiliary: a noise-floor gate computed over the first 200 ms of recording. Onsets below 6 dB above noise floor are dropped.

**Two-tier release gate (Q70):**

- Detection F-measure ≥ 0.92 on AVP at IOU = 50 ms.
- Detection F-measure ≥ 0.88 on OOD at the same IOU.
- Median absolute timing error (MAE) on true positives ≤ 15 ms AVP / ≤ 25 ms OOD.

Both tiers must pass. A percussion application has two failure modes invisible to one another under a single metric — missed onsets and consistently late onsets. The MAE tier catches the latter.

### 5.2.1 Headphone-bleed setup step (FIR-based, with mid-session re-estimation)

**Initial setup:**

1. User puts on headphones at intended monitoring level.
2. Tool plays a 4-bar click while recording silence.
3. Tool finds the bleed-path impulse response `h` by averaging click-pulse-aligned windows from the recording. Length: 32–128 taps at session rate (default 64). Optionally apply a short Hann taper to suppress edge artifacts.
4. Coarse alignment: deconvolution search window ±2000 samples (~45 ms at 44.1 kHz) to cover round-trip latency on real interfaces.
5. Subtraction quality check: measure post-subtraction click residual energy. Quality indicator (Q79): post-subtraction click residual ratio in dB; UI mapping: green ≥ 20 dB, yellow 10–20 dB, red < 10 dB.
6. **Bleed gate with override (Q23).** If residual exceeds noise floor + 6 dB, default action is to refuse to proceed. User may override with explicit acknowledgment. Override flag persists in session metadata.
7. `h` persisted per device.

**Mid-session re-estimation (Q25, Q35, Q47, Q48, Q49):**

8. During recording, the system maintains a rolling buffer of click-aligned windows over the most recent 16 bars, plus a separate "silent windows" rolling buffer. A click-aligned window qualifies as **actively silent** (Q47) if its post-click RMS (excluding ±15 ms around the click position) is below `noise_floor + 6 dB`. A window is **passively silent** (Q48) if a run of ≥ 4 consecutive active-silent windows precedes it (~200 ms of consecutive silence at the click cadence). Passive sliding window is over 8 click-aligned windows of 50 ms each.
9. Re-estimation is triggered when: (a) every 32 bars of cumulative recording since the last estimate, OR (b) the click-window guard fires on > 2% of click positions in the most recent 8 bars.
10. Re-estimation uses silent-window averaging only (Q35). If a trigger fires but fewer than 8 silent windows are available in the recent 16 bars, the user is prompted for a 4-bar silent re-capture.
11. Compute happens on a worker thread; the IR is hot-swapped atomically when ready (RWLock-style guard; swap is a single pointer assignment after the new IR's residual has been validated against the rolling buffer via `compare_estimates`).
12. If `compare_estimates` returns "old is better," the new IR is logged but not activated; the trigger is treated as a transient.
13. If the re-estimated IR fails the bleed-gate threshold, the user is prompted to take a quick re-setup.
14. **`bleed_ir_history` (Q39, Q49):** FIFO cap at 20 entries, two protected slots (most recent setup-origin entry + most recent active-silent re-estimation entry) exempt from eviction.

### 5.3 Embedding extractor

For each onset:

1. Extract a window from the recording: 200 ms starting at the onset (clamped to buffer end).
2. **Pad to model input length (1 s at 16 kHz) using surrounding recording context** (Q21) — extend the window backward and forward into the recording, keeping the original onset's relative position constant. If the onset is within 800 ms of the recording boundary, fall back to zero-padding for the missing portion only.
3. **No pre-emphasis filter.** PANNs was not trained on pre-emphasized input; applying it shifts the spectrum out of the model's training distribution.
4. Resample full audio to 16 kHz **once** with `librosa.resample`, then slice — much cheaper than per-onset resampling.
5. Run through the frozen PANNs CNN14 model via ONNX Runtime in production, PyTorch in eval. Take the 2048-dim embedding (768-dim for BEATs) from the final pooling layer.
6. The embedding is the feature vector for classification.

The extractor returns full-dim embeddings; PCA projection (if Q43 ships) happens in the Classifier, not here. This split keeps the full-dim embedding available for Mahalanobis.

**Auxiliary features (not for classification):**

- **RMS** of the 200 ms window — used for MIDI velocity mapping. Mapping uses **5th/95th percentile** of session-wide RMS values:
  ```
  rms_lo = np.percentile(all_event_rms, 5)
  rms_hi = np.percentile(all_event_rms, 95)
  velocity = clip(round(127 * log10(rms / rms_lo) / log10(rms_hi / rms_lo)), 1, 127)
  ```
  Robust against single-event outliers. Edge case: fewer than 20 events → fall back to (10th, 90th) for 10–19 events, and to (min × 1.5, max × 0.67) for < 10 events.

**Cache:** embeddings persisted to project file. Cache invalidated if `session.embedding_model_id` doesn't match the loaded model's `model_id`.

### 5.4 Classifier (composite unknown gate)

**Primary (ships):** `sklearn.linear_model.LogisticRegression(penalty='l2', C=1.0, multi_class='multinomial', max_iter=1000, class_weight=None)` over either the full-dim embedding (2048 PANNs / 768 BEATs) or PCA-64-projected embedding depending on Q43 outcome. 4 trained classes per `TaxonomyConfig` (Q66). Output: temperature-scaled softmax with T fit on disjoint held-out fold per Q75.

**Composite unknown gate (Q34):**

```python
def predict_one(embedding: np.ndarray, calibrated: CalibratedModel) -> tuple[ClassId, float]:
    # 1. LR head runs in whatever space LR was trained in
    embedding_for_lr = (
        calibrated.pca @ embedding if calibrated.pca is not None else embedding
    )
    logits = calibrated.lr.decision_function(embedding_for_lr[np.newaxis])[0]
    probs = softmax(logits / calibrated.T)
    top_class_idx = int(np.argmax(probs))
    top_score = float(probs[top_class_idx])

    # 2. Mahalanobis ALWAYS on full-dim, distance to NEAREST centroid,
    #    via Cholesky factor (Q68)
    distances = np.array([
        mahalanobis_sq_via_cholesky(embedding, c, calibrated.pooled_cov_cholesky_full_dim)
        for c in calibrated.class_centroids_full_dim
    ])
    nearest_class_idx = int(np.argmin(distances))
    nearest_distance = float(np.sqrt(distances[nearest_class_idx]))

    if top_class_idx != nearest_class_idx:
        calibrated.telemetry.log_softmax_mahalanobis_disagreement(...)

    softmax_unknown = top_score < calibrated.softmax_threshold
    distance_unknown = nearest_distance > calibrated.distance_thresholds[nearest_class_idx]

    if softmax_unknown or distance_unknown:
        return (calibrated.taxonomy.unknown_class_id, top_score)
    return (calibrated.taxonomy.classes[top_class_idx], top_score)
```

**Why nearest centroid, not top-class centroid:** If an embedding sits roughly between two centroids, softmax picks one, but the Mahalanobis distance to the *other* centroid may be smaller — so the top-class formulation could miss the case where the embedding is actually nearer to a different in-distribution class. Distance to the nearest centroid is the right "is this near anything we know about" question. Each class's threshold is the radius of its 95th-percentile ball; we ask whether the embedding sits inside any class's ball.

**Why Mahalanobis on full-dim, even with PCA-64:** PCA fit on AVP keeps the directions of high AVP variance. By construction it discards directions of low AVP variance. **OOD samples often differ from AVP precisely along the discarded directions.** Running Mahalanobis in the PCA-64 space silently disables the OOD detector for any OOD that's "perpendicular to AVP" — a known failure mode in the OOD-detection literature. The model bundle stores both the (optional) PCA matrix and the full-dim Mahalanobis machinery (Cholesky factor of pooled covariance, centroids, per-class thresholds). Storage footprint of the full-dim Cholesky is `D_full × (D_full+1)/2` floats ≈ 8 MB for PANNs, ≈ 1.2 MB for BEATs.

**Mahalanobis covariance source under weighted calibration (Q52):** When `fit_with_calibration` runs with `calibration_weight = 50` (or the eval-selected value), centroids and the LR fit incorporate the weighted calibration data. **Pooled covariance and per-class distance thresholds, however, are computed on AVP only, unweighted.** With 3–12 calibration points per class, weighting them 50× makes them dominate the covariance estimate. The covariance is supposed to characterize the *spread* of in-distribution embeddings; that quantity should come from the larger, less-noisy AVP set. The mean (centroid) is a different beast — robust statistic, benefits from incorporating user-domain data.

**Why drop `class_weight='balanced'` in user refit (Q26):** The earlier compound weighting (`class_weight='balanced'` × `calibration_weight=5.0`) silently overrode user intent. With the user recording 10 kicks and 5 of every other class during calibration, a snare calibration sample got 6.25× weight while a kick sample got 3.125× — the opposite of what most users would expect. With `class_weight=None`, calibration samples within a single class get equal weight, and the calibration weight does the only lifting. `class_weight='balanced'` is still applied to the **base AVP fit at build time** (where balanced classes genuinely matter and there is no user signal to preserve).

**Why temperature scaling (Q27):** Logistic regression's `predict_proba` outputs are not well-calibrated probabilities. Temperature scaling fits a single scalar T by NLL minimization: `softmax(logits / T)`. This produces a proper probability distribution (per-class scores sum to 1.0) at negligible inference cost. Per-class Platt sigmoids via `CalibratedClassifierCV` were dropped: per-class scores did not sum to 1.0, and the eval wallclock cost (5× the LR fit) was non-trivial. Temperature is refit on a 20% held-out fold of AVP not used in LR training or Mahalanobis covariance estimation per Q75; **never fit on user calibration data.**

**Calibration retrain:**
- Linear head retrained from scratch on AVP + calibration samples (calibration samples weighted per Q42 default; provisional 50× pending Q65 sensitivity study).
- Retrain time: ~150–250 ms on a CPU at 4 classes; ~200–350 ms with temperature refit. Runs on a worker thread; queue subsequent refit requests; coalesce if multiple arrive while one is in-flight.
- **Live class prediction during calibration:** enabled only once at least one sample exists for every class. Before that, UI shows: "Record at least one of each sound to enable live preview."

**Self-test overfit guard (Q71):** After fitting, the guard computes LOSO macro-F1 on AVP held-out folds (excluding subjects whose data was used in calibration). If macro-F1 has dropped > 1 point relative to the no-calibration baseline at the same `calibration_weight`, calibration is rejected: previous calibration restored, recovery dialog shown per Q81. Diagnostic file always records the macro-F1 delta.

**Distribution-shift warning threshold (Q45):** AVP-derived (median score on AVP held-out folds × 0.7). Stored in model bundle. If user's first-100-events median score drops below this, orchestrator surfaces a non-blocking recalibration prompt per Q44.

**Output per onset:** `(class_id, score)` where `score` is the calibrated probability of the predicted class. For `unknown` returns, `score` is the highest of the 4 trained-class probabilities — interpretable as "the model's best guess was X with this confidence, but that wasn't enough."

**Acknowledged failure modes:** Overlapping/layered sounds → single-class assignment, flagged in UI. Whisper-quiet hats below noise gate → missed onsets. Voices stylistically far from AVP's 28 amateur subjects → degraded accuracy that calibration partially corrects. Open vs closed hat is a harder discrimination than a 3-class merge made it look; per-class F1 in LOSO will likely show open_hat as the weakest class. Fallback: an advanced-settings toggle "treat open hat as closed hat" merges them if shipping eval shows open_hat per-class F1 < 0.7.

### 5.5 Tempo and grid

Pure functions over BPM, time signature, and event timing:

```python
def beats_per_bar(ts: TimeSignature) -> int: ...

def grid_positions(bpm: float, ts: TimeSignature, bars: int, grid: str) -> list[float]:
    """Returns list of times in seconds for each grid line."""

def quantize(onset_time: float, grid_positions: list[float], strength: float) -> float:
    """strength=0 → no change; strength=1 → snap to nearest grid line.
    Linear interpolation in between."""

def velocity_from_rms(rms: float, rms_lo: float, rms_hi: float) -> int:
    """Logarithmic mapping; clipped to 1..127."""

def compute_velocity_bounds(rms_values: np.ndarray) -> tuple[float, float]:
    """Returns (5th percentile, 95th percentile)."""
```

All pure. No state. Triplet grids: `1/8T` = 1/12 of a bar in 4/4; `1/16T` = 1/24. Grid string parser handles "1/N" and "1/NT" syntax.

### 5.6 Calibration manager

Captures calibration samples, manages the calibration set, triggers refit, provides live preview during capture, exposes a cancellable handle to in-flight refits per Q28. Calibration weighting decision driven by Q42/Q65 sensitivity study. Self-test overfit guard per Q71.

- **Live preview gating:** `can_predict_live()` returns False until every class has ≥ 1 sample.
- **Refit on each add:** ~150–250 ms (4 classes). Runs on worker thread; queue subsequent refit requests; coalesce if multiple arrive while one is in-flight.
- **`commit()` semantics (Q28):** returns a `CommitHandle` with `cancel()`, `wait()`, `status`, `error`. Newer commits cancel in-flight predecessors. Last-write-wins. Worker checks the cancellation flag at safe points (after AVP load, before temperature refit, before save). On cancel, transitions to `cancelled` and exits.
- **Minimum gate (Q30):** `MIN_SAMPLES_PER_CLASS = 3`. Commit refused below this. UI shows "more samples = better results" hint between 3 and 10 samples per class.
- **Drop-off telemetry (Q30):** `record_abandon_event` writes `(class, count)` pairs to the local diagnostic file when the user closes calibration without committing. **No data leaves the user's machine** in v1.x.
- **Status reporting:** counts per class; leave-one-out CV accuracy estimate (post-temperature) over the calibration set itself; per-class consistency (intra-class cosine similarity).
- **Calibration set persistence:** persists in user profile, separate from any single project.

### 5.7 Telemetry

The telemetry sink writes to a local file in the user's profile directory (`~/.voxkit/diagnostics/<session-id>.jsonl`). No network beacons. No remote server. Diagnostic file rotation: 30-day TTL on disk; user-configurable or fully disable-able.

Schema:

```json
{
  "ts": "2025-...",
  "event": "softmax_mahalanobis_disagreement | calibration_overfit_guard_triggered | bleed_re_estimation | resampler_overrun | dropped_buffer | distribution_shift_warning | calibration_abandoned | ...",
  "details": {...}
}
```

If a user wants to share diagnostics for a bug report, the documented flow is: open the diagnostic file, review it, attach to a GitHub issue. There is no in-app "share telemetry" button in v1.0.

The 72-hour deletion SLA (Q56) is preserved as a forward-looking commitment but is moot for v1.0 — there is nothing on a server to delete.

If a future minor reintroduces network telemetry (Q61), it must satisfy: (i) explicit per-session opt-in with data preview, (ii) build-time disable flag, (iii) operates correctly under no-outbound-network configuration.

### 5.8 Editor (UI)

Four-or-five-lane piano roll synced with waveform display; drag to retime, drag across lanes to reclassify, undo/redo. Lane count driven by `TaxonomyConfig` (Q66); default = 5 lanes (4 trained + unknown).

- **Custom-painted via QPainter.** Five horizontal lanes for kick/snare/closed_hat/open_hat/unknown. Each event is a draggable rectangle.
- **Waveform displayed beneath the lanes,** time-axis-aligned. Click anywhere to seek the playback engine.
- **Snap toggle:** when on, dragging snaps to current grid; when off, free-drag.
- **Reclassify-on-drag:** when an event is dragged across a lane, the editor emits `eventReclassified(int, Event)` with the new class. The calibration manager can listen and prompt the user to add as a calibration sample. Opt-in, not automatic.
- **Undo/redo:** command-pattern stack capped at 100 entries.
- **Confidence-based visual hint:** events with score < 0.7 are rendered with a yellow border. The unknown lane is visually distinct: neutral grey, dashed lane separator above and below, and a small "?" glyph in the lane label.
- **`score` field consumed throughout:** event tooltips show "Confidence: 87%"; for unknown events, "Best guess: snare (32%) — below confidence threshold."
- **Taxonomy disclosure (Q31):**
  - Calibration screen header: "VoxKit recognizes four sounds: kick, snare, closed hi-hat, and open hi-hat. Anything that doesn't clearly match one of those will be tagged 'unknown' so you can decide what to do with it."
  - Editor first-open banner: "VoxKit detected 5 lanes: 4 trained sounds plus 'unknown' for events that didn't clearly match. Drag from unknown into one of the four trained lanes to reclassify."
  - Reclassify-on-drag tooltip (on first drag of each session): "Drag corrections train VoxKit on your sounds when you re-calibrate."
- **Bleed-resetup banner (Q29):** for sessions loaded with `bleed_ir_origin == "migrated_pending_recapture"`, a non-modal banner appears: "This session uses an estimated bleed setting from an older VoxKit. [Re-run setup] [Dismiss]." Banner reappears on next open until re-setup is done.
- **Mid-session bleed prompts (from §5.2.1):** non-blocking toasts when re-estimation events occur. Two variants: (a) silent windows available but new IR failed the bleed-gate: "Bleed has changed (likely headphone shift). [Re-setup] [Continue with current]." (b) trigger fired but no silent windows available: "Bleed has drifted but VoxKit needs a quiet moment to recheck. [Take 4 seconds of silence] [Skip and keep current bleed setting]."
- **Distribution-shift warning toast (Q44):** "Confidence is consistently low on your events. Re-calibrating with more samples may help. [Open calibration] [Dismiss]." Fires once per session if the warning condition is met.
- **Device-disconnect modal (Q37):** blocking modal — "Audio device disconnected. Recording paused at bar X. Reconnect the device or select a different one." [Reconnect/Select] [Save and exit].
- **Post-sleep toast (Q38):** non-blocking — "Device validated after sleep. Bleed setup may have changed (volume, headphone position). [Re-run bleed setup] [Continue]."
- **First-run guided tour (Q54):** trigger fires on the **first unknown event** in the first session that contains any unknowns. Once per user (not per-session), dismissable, never repeats after dismissal. Implementation via Qt `QPropertyAnimation`.
- **PCA-Mahalanobis recalibration banner (persistent):** on session load if `mahalanobis_full_dim is None` after a v0.8 → v0.9 PCA-session migration: "VoxKit improved out-of-distribution detection in this version. Re-run a quick calibration to enable it. Until then, unknown detection uses the previous (less accurate) method." Banner is **not dismissable** until calibration runs. If the user ignores it, the OOD detector silently falls back to softmax-only gating (logged in diagnostics) but the banner remains.
- **Recording-session UX (Q73):**
  - During recording: live waveform; click-bleed quality indicator (numeric percentage + colored bar per Q79); recording duration; bar count if click track is set. No live classification.
  - On stop: modal progress dialog with three named phases — "Detecting onsets" → "Extracting embeddings" → "Classifying events" — each with a percentage. Total wall-clock target ≤ 0.5× audio duration on reference hardware (Q72). For a 16-second session, ≤ 8 seconds.
  - Cancel button preserves audio and returns to recording mode. After completion: user lands in editor with all events placed and labeled.
- **Inference pipeline threading (Q76):** the recording-session progress dialog is backed by a dedicated `InferenceWorker` thread. The main thread runs only the Qt event loop. Phase progress reported via Qt signals; the cancel button sets a `threading.Event` flag checked between phases and at the top of each per-onset iteration inside embedding extraction. Worst-case cancel latency is one embedding-extraction call (~50 ms reference). On cancel, the recorded audio buffer is preserved in the Session; partial events are discarded.

### 5.9 MIDI exporter

Reads class set and MIDI mapping from `TaxonomyConfig` (Q66). Default mappings:

- kick → 36 (GM Acoustic Bass Drum)
- snare → 38 (GM Acoustic Snare)
- closed_hat → 42 (GM Closed Hi-Hat)
- open_hat → 46 (GM Open Hi-Hat)

Velocity formula per §5.3 (logarithmic 5th/95th-percentile RMS mapping).

**Unknown export (opt-in):** by default, unknown events are not exported. The export dialog reports a count: "12 events tagged 'unknown' will not be exported." A toggle exports them to a configurable GM percussion note (default 56 / Cowbell). Dialog has a dropdown of GM percussion notes (35–81) for the unknown-event note selection.

`pretty_midi`'s tempo + time-sig handling is correct for FL Studio import; verified. Single drum track; events on channel 9 (MIDI channel 10, GM percussion). Note duration: fixed 50 ms. Velocity comes from the Event (already computed by the velocity-from-RMS function).

### 5.10 Project file

Schema (zip-bundle .vxk):

- `manifest.json` — top-level metadata including `voxkit_format_version` (Q78), session metadata (bpm, time sig, bars, sample_rate, recording_sample_rate, recording_audio_api, embedding_model_id), `bleed_gate_overridden`, `dropped_buffer_count`, calibration params reference, `softmax_threshold`, `calibration_weight`, taxonomy config reference. SPDX header (`SPDX-License-Id: GPL-3.0-or-later`).
- `audio.wav` — recording, 32-bit float, mono.
- `bleed_ir.npy` — current active FIR taps if present.
- `bleed_ir_history.npz` — historical IRs keyed by `measured_at` ISO string, with two protected slots (Q49).
- `mahalanobis_full_dim.npz` — `class_centroids_full_dim`, `pooled_cov_cholesky_full_dim` (lower triangular, Cholesky factor per Q68), `distance_thresholds`. Always present in v0.9+ sessions, regardless of `pca_matrix_present`.
- `pca_matrix.npz` — present only when `pca_matrix_present == True`. Used only for the LR head input projection (Q43), not for Mahalanobis.
- `events.json` — events without embeddings (uses `score`, not `confidence`).
- `embeddings.npz` — embeddings keyed by event index, float16 (optional, regenerable).
- `taxonomy_config.json` — `TaxonomyConfig` (Q66). v1.0 ships with default 4-class config.
- `temperature_calibration.json` — temperature scalar T (Q27).

**Migration registry (Q78):**

`manifest.json` adds a top-level `voxkit_format_version` string field. Migration code dispatches on this field via a registered table:

```python
MIGRATIONS: dict[tuple[str, str], Migrator] = {
    ("0.4",  "0.5"):  migrate_0_4_to_0_5,
    ("0.5",  "0.6"):  migrate_0_5_to_0_6,
    ("0.6",  "0.7"):  migrate_0_6_to_0_7,
    ("0.7",  "0.8"):  migrate_0_7_to_0_8,
    ("0.8",  "0.9"):  migrate_0_8_to_0_9,
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

Per-step migration semantics:

- **v0.4 → v0.5:** convert `bleed_gain: float` to a length-1 `BleedIR`. Banner: "This session uses an estimated bleed setting from an older VoxKit. Re-run setup for best results."
- **v0.5 → v0.6:** convert single `bleed_ir` to length-1 `bleed_ir_history`; set `origin="setup"`; set `output_calibration = None`.
- **v0.6 → v0.7:** drop `PlattCalibration` (per-class `{a, b}`) and set `output_calibration = None`; add `unknown_threshold = 0.45`. Banner: "VoxKit's confidence model has been improved. Re-calibrate to refresh confidence scores for this session."
- **v0.7 → v0.8 / v0.8 → v0.9:** add `mahalanobis_full_dim` machinery; if v0.8 session was created with PCA-64, the Mahalanobis machinery in the file is in PCA space and is **discarded** on load (v0.9 will re-fit Mahalanobis on full-dim at the next calibration). Persistent non-dismissable banner per §5.8.
- **v0.9 → v0.10:** convert `pooled_inv_covariance_full_dim` to `pooled_cov_cholesky_full_dim` (compute `inv(pooled_inv_covariance)`, then Cholesky factor — well-conditioned after Ledoit-Wolf shrinkage). Add default `TaxonomyConfig`. No user-visible event.
- **v0.10 → v0.11:** stamp `voxkit_format_version: "0.11"` on save; populate the field on legacy bundles missing it. `migrate_0_10_to_0_11_stamp_version` is a no-op on data.

Bundles missing the version field are treated as `"0.4"` and walked through all migrations in sequence.

**Sunset cadence (Q55):** Two-minor-version sunset. v1.0 introduces deprecation warning for v0.4; v1.2 removes v0.4 load. Each subsequent minor version sunsets the oldest still-supported format. Caps the migration matrix at ~3 hops, not n-hops-forever.


---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Click-bleed handler IR estimation fails on novel headphone/mic setups | Medium | High (cascade: bleed misclassified as percussion, all downstream metrics degrade) | Q80 week-1 tracer bullet on maintainer's setup with deliberately leaky open-back headphones. Acceptance threshold: > 20 dB null after 2 seconds. Escalation path documented (NLMS / RLS) if simplest approach fails. |
| Audio callback GIL contention causes buffer drops despite MMCSS | Medium without fix | Medium | Q67 amended: default path keeps GIL-held time < 50 µs/call via NumPy buffer copy; hardened CFFI-callback path documented as escalation if measured drops require it. |
| Inference pipeline blocks main thread during recording-stop processing | High without fix | Medium (Q73 cancel button non-functional, UI freeze) | Q76: dedicated `InferenceWorker` thread; main thread runs Qt event loop only; cancellation via `threading.Event` checked between phases and per-onset. |
| Platform-specific code leaks beyond `Recorder` over time | High without fix | Medium (Phase 1.5 Linux becomes a refactor, not a wiring exercise) | Q77: `import-linter` enforces isolation in CI; violations block merge. |
| First-time contributors cannot run any tests because dataset access is unclear | Medium | Medium (contribution velocity collapses) | Q83 CONTRIBUTING.md committed week 1 documents the synthetic-tier-only quickstart path. Q63 three-tier dataset plan. |
| Source files drift out of SPDX compliance over time | Medium without fix | Low–Medium | Q82: pre-commit hook + CI `reuse` check blocks merge. |
| Pre-training data license (AudioSet) propagation to redistributable model weights unaddressed | Medium | Medium (potential redistribution legal grey area) | Q60 amended: memo template seventh field surfaces the question for explicit disposition. |
| Classifier accuracy poor on uncalibrated users | Medium | High | Mandatory calibration as blocking gate; PANNs embedding raised the uncalibrated baseline. |
| Onset detector misses quiet hats / fires on breath | Medium | Medium | Vocal-percussion-tuned defaults; sensitivity slider; release gate on AVP F-measure (Q53) and alignment MAE (Q70). |
| Click bleed leaves colored residual after subtraction | Low | Low | FIR-based subtraction (§5.2.1); mid-session re-estimation handles drift over long takes; silent-window-only re-estimation prevents user-performance contamination. |
| Open vs closed hat confusion in classifier | Medium | Low–Medium | Per-class F1 reported in LOSO; user can drag to reclassify. Fallback "merge hats" toggle if open_hat F1 < 0.7. |
| Inference runtime size / antivirus | Low | Low | ONNX Runtime keeps installer at ~150 MB. PyTorch antivirus heuristic surface gone. |
| AVP-overfitting (LOSO numbers don't reflect real users) | Low–Medium | Medium | §7.9 OOD validation set (≥15 subjects). Bootstrap CIs on the gap. N=15 distinguishes a 10-point gap with ~80% power. |
| PANNs domain mismatch (AudioSet vs vocal percussion) | Medium | Medium–High | OOD set is the primary detector. BEATs benchmark in §7.3 quantifies whether a different embedding would help. If OOD-vs-AVP gap > 15 macro-F1 points, fine-tuning last 2 PANNs layers becomes a v1.1 priority. |
| Sounds outside the 4-class taxonomy mapped silently | Low | Low | Composite gate per Q34: distance to nearest centroid on full-dim embeddings. User-facing taxonomy disclosure (Q31). Fifth class (other/unknown) is first-class via the runtime gate. |
| Audio-callback resampling glitches under load | Low–Medium without fix | Medium | Q51: resampling moved off the audio callback to a worker thread with pre-allocated filter state and a lock-free SPSC ring buffer. |
| Release-gate criterion ambiguous; build could "pass" while shipping a 75% miss rate | Medium without fix | High | Q50: rewritten as two explicit gates (missed-unknown ≤ 25%, false-unknown ≤ 5%). |
| Substrate decision swayed by 3-subject pilot noise | Medium without fix | Medium | Q33 amended: pilot raised to 5 subjects and used as tiebreaker only. Q74: overlapping bootstrap CIs replace fixed 5-point margin. |
| Calibration weight rationale unverified | Low | Low–Medium | Q42/Q65: eval measures empirical effective influence per weight; default re-anchored on noise-sensitivity study. |
| Time-stretch licensing cost / friction at v1.0 | Very Low | Low | Q46 amended: MIT-licensed `signalsmith-stretch` is the default; Rubber Band only if quality A/B test rejects the MIT option. |
| Onset detection regression undetected | Low | High | Q53 + Q70: explicit F-measure thresholds (0.92 AVP / 0.88 OOD at IOU = 50 ms) AND alignment MAE thresholds (≤ 15 ms AVP / ≤ 25 ms OOD) gate the build. |
| Mahalanobis covariance dominated by 50× weighted calibration noise | Medium without fix | Medium | Q52: pooled covariance and distance thresholds computed on AVP only. Centroids alone use weighted calibration. |
| Incompatible-license dependency creep | Low (with Q60 vigilance) | Medium | Q60: per-dependency license memo verifies GPL v3 compatibility. |
| Telemetry-induced contributor pushback | Low | Medium | Q61: local-file diagnostics only; no network telemetry in roadmap. |
| Win32-lock-in delays Linux/Mac | Medium without fix | Medium | Q62: architectural portability in Phase 1; Linux CI from week 1. Q77: enforced via `import-linter`. |
| OSS contributors cannot reproduce evals | High without fix | High (kills contribution velocity) | Q63: three-tier dataset plan; CI on synthetic, PR validation on minimum-reproducible. |
| Release-gate numbers indefensible at audit | Medium | Medium | Q64: per-threshold one-page memos before v1.0 lock. |
| Class-taxonomy hardcoding blocks v1.1 throat-bass | High | Medium | Q66: parameterize internally now; user-facing surface unchanged. |
| Audio thread sample-format ambiguity | Low | Medium | Q67: float32 mono native byte order; explicit contract documented. |
| Onset alignment regression invisible to F-measure | Medium | Medium | Q70: alignment MAE tier added to release gate. |
| Calibration silently overfits without user notification | Low (with fix) | Medium–High | Q71: self-test overfit guard specified; rejected calibration triggers recovery dialog (Q81). |
| Temperature scaling overfits to LR training data | Medium without fix | Medium | Q75: T fit on disjoint held-out fold. |
| CPU performance unacceptable for the recording-studio loop | Medium | High (unusable product) | Q72: concrete wall-clock target on reference hardware; substrate decision must report this number. |
| Misleading accuracy expectations from Sinyor 2005's 95.55% | Medium | Medium | Spec explicitly states this number is not transferable to AVP. Numbers are validated, not promised, before any release. |
| AVP class imbalance after hat split | Medium | Low | Class weights at training base fit (`balanced`); per-class F1 reported in LOSO. |
| Bundled checkpoint AudioSet license | Low | Low | Q60 amended: memo addresses pre-training data license propagation. Attribution in About dialog. |
| Time-stretch artifacts at extreme speeds | Medium | Medium | `signalsmith-stretch` (Rubber Band fallback). Test at 50/75/150/200%. |
| Inter-stem flam during playback | Low | High | Single audio callback for all stems. |
| FL Studio reads tempo or time-sig incorrectly | Low | Medium | Release-gate test, owned by maintainer. |
| Windows packaging / SmartScreen warning | Low | Low | Documented. |
| User performs in 3/4 but selects 4/4 | Medium | Low | Metronome is the realtime confirmation. |
| User overrides bleed gate and gets bad results | Low | Low | Override flag in session metadata; first-line of any "results are wrong" support response is to check the flag. |
| Eval gates aspirational, not run | Low | High | Tiered eval (§7.10): smoke pass on every PR enforces the gate exists and runs; nightly full pass keeps numbers fresh. |
| Bluetooth user blocked at device picker complains | Low–Medium | Low | Explicit error text explains the latency reason; v1.1/v1.2 may add explicit Bluetooth support if Phase 2 measurement prototype demonstrates < 50 ms median latency. |
| Bleed path drift mid-session | Low | Low–Medium | Periodic re-estimation (§5.2.1); user-prompted if drift exceeds bleed-gate threshold. |
| First-run drop-off from calibration friction | Medium | Medium | Lowered minimum to 3/class; drop-off metric instrumented for v1.1. Not fully mitigated; tracked. |
| Mac users blocked entirely (Phase 1) | Medium | Low | Documented as Phase 1 limitation. Phase 2 deliverable. |
| Audio device disconnection mid-session | Low | Medium | Typed `DeviceDisconnected` exception (Q37), recording paused, modal in UI. Session preserved up to disconnect point. |
| OS sleep mid-session | Low–Medium | Medium | Wake handler treats wake as device-disconnect-recovery (Q38). Re-validates device, offers bleed re-setup, does not auto-resume recording. |
| Unknown threshold mistuned for a given user | Medium | Low | Eval sweep includes threshold values; default 0.45 chosen from sweep. Per-user threshold is a v1.1 candidate. Distribution-shift telemetry warning (Q44) catches the worst case. |


---

## 7. Test strategy

### 7.1 Unit tests

Every pure function (feature extractor, quantizer, MIDI writer, time-sig math, velocity-from-rms with percentile bounds, beats_per_bar, grid_positions, quantize) gets unit tests.

### 7.2 Golden-file pipeline tests

Small set of canonical recordings (hand-labeled, both 4/4 and 3/4) checked into the repo. Pipeline runs against them; events compared to ground truth with ±15 ms timing tolerance, exact class match. Synthetic dataset (Q63 tier (c)) used as the absolute floor for CI on every PR per Q85.

### 7.3 Classifier evaluation

**Primary metric:** Leave-One-Subject-Out cross-validation on AVP-LVT v4. For each of the 28 subjects, train (the linear head — embedding stays frozen) on the other 27 subjects' embeddings, evaluate on the held-out subject. Report:

- Macro-F1 (primary).
- Per-class precision / recall / F1, including separate open_hat / closed_hat numbers.
- 5×5 confusion matrix (4 trained classes + unknown).
- Per-subject accuracy distribution (mean + std across the 28 LOSO folds).

**Why LOSO:** random splits leak the same speaker's voice into train and test, inflating reported accuracy by 10–20 points. LOSO is the only protocol that matches the production reality where every user is out-of-distribution on first contact.

**Stratified k-fold** is run as a secondary sanity check (catches obvious bugs) but is *not* the headline number.

**Sweep axes:**
- Embedding window length ∈ {150, 200, 300, 500} ms.
- Padding strategy ∈ {context, zero}.
- Logistic regression `C` ∈ {0.1, 1, 10}.
- Calibration sample weight ∈ {1, 5, 25, 50, 125, 625} (geometric with 50× probe per Q42).
- Mahalanobis distance percentile ∈ {90, 92.5, 95, 97.5, 99} on full-dim only (PCA-projected Mahalanobis dropped per Q34).
- Softmax confidence threshold ∈ {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70} for the unknown gate.

Total: 4 × 2 × 3 × 6 × 5 × 9 = 6480 post-hoc evaluations per LOSO fold; only the LR fit is per-config (the embedding cache is shared across post-hoc evaluations). Wallclock estimate: smoke tier ~16 h / 3 h on 8-core; full tier ~85 h single-machine, ~11 h on 8-core.

**Operating-point selection (Q50, Q70):** the sweep selects the `(softmax_threshold, distance_percentile)` pair that minimizes missed-unknown-rate on a representative OOD-like fold *subject to* the false-unknown-rate ceiling (≤ 5%) on AVP held-out. If no operating point satisfies both, the build fails the gate and the model needs work (more training data, different substrate, etc.) — not a threshold knob fix.

**Substrate evaluation (Q33, Q74):**
- Week 1: collect 5-subject OOD pilot.
- Week 2: smoke-tier LOSO on AVP for both PANNs and BEATs.
- Decision: substrate wins outright if 95% bootstrap CI of AVP-LOSO macro-F1 (1000 resamples) does not overlap with the other substrate's. Otherwise pilot OOD (5-fold cross-val on pilot subjects) is the tiebreaker.

**Benchmark comparison (one-shot, not swept):** AdaBoost + CART on the v0.3 hand-crafted feature set, SVM-RBF on the same, linear head on raw mel-spectrogram (no PANNs), BEATs frozen embedding + linear head. If any of these beats PANNs + linear head by > 2 macro-F1 points on LOSO, halt and investigate (likely a bug in PANNs integration).

### 7.4 Calibration uplift evaluation

For each held-out subject in LOSO, simulate calibration by sampling N (∈ {3, 5, 10, 20}) of their utterances per class as "calibration data," append them to the training set with elevated weight (per the sweep), retrain, and re-evaluate on the rest of that subject's utterances.

Report:
- Macro-F1 with vs without calibration.
- Uplift distribution across subjects.
- **Empirical effective-influence measurement (Q42):** at each weight, compute leave-one-out coefficient sensitivity averaged across calibration samples; report median per-class effective influence.
- **Calibration-weight sensitivity study (Q65):** for each weight, additionally fit LR to deliberately noisy synthetic calibration data: 3 calibration points per class with Gaussian additive noise on embeddings at three noise levels (σ ∈ {0.1, 0.5, 1.0} × per-feature std). Compute LR coefficient drift from the no-calibration baseline at each (weight, noise) pair. Default weight = largest weight at which `drift(noisy) / drift(clean) < 2.0` across all noise levels.

**Pass criterion for calibration design:** median uplift ≥ 8 macro-F1 points across subjects. Self-test overfit guard per Q71 runs on every calibration in the eval; rejected calibrations logged.

### 7.5 MIDI round-trip test

Generated `.mid` re-parsed by `mido`; verify every event matches expected (time within MIDI tick resolution, pitch, velocity). Tests exercise 4 drum-map entries (kick, snare, closed_hat, open_hat) plus opt-in unknown-class export to a configurable GM note. Reads class set from `TaxonomyConfig` per Q66. Plus the manual FL Studio test on every release.

### 7.6 Latency / flam test

Synthetic recording with known impulse positions; measure scheduled vs actual playback time of MIDI samples through the audio callback. Verify per-stem alignment within ±1 ms.

**Resampler-worker latency budget (Q51):** 10 ms per buffer; alarm if > 100 consecutive buffers exceed. Synthetic test feeds a known-rate device (48 kHz) and verifies the worker thread keeps up under simulated CPU contention (pinned at 80% load on the worker core).

**CPU performance benchmark (Q72):** end-to-end inference on the standard 32-bar / 120 BPM session timed on reference hardware (i5-8250U or equivalent CI runner). Required: ≤ 8 wall-clock seconds. CI emits the number on every run; PRs that increase it by > 10% require explicit reviewer sign-off.

### 7.7 Click-bleed test (FIR-aware)

Test rig: synthetic recording = `conv(click_pulse_train, h_synthetic) + noise`, where `h_synthetic` is a known FIR (e.g., delay + 4-tap lowpass).

Verify:
1. Per-tap recovery within 2 dB at SNRs of 0, 10, 20, 30 dB.
2. After subtraction: residual energy < 1% of original bleed energy at SNR ≥ 20 dB.
3. Onset detector finds **≤ 1 false positive per 100 click positions** on the click-subtracted signal at SNR ≥ 20 dB.
4. Alignment robustness: synthetic delay of 500 samples should be recovered.
5. Bleed-gain measurement from §5.2.1 recovers known scalar gain within ±2 dB at bleed gains -40, -30, -20, -10 dB. The setup step refuses the -10 dB case.

**Mid-session re-estimation tests:**

6. Synthetic recording with a known IR that changes mid-recording (`h_initial` for first half, `h_drifted` for second half). Verify: drift trigger fires within 8 bars of the IR change; re-estimated IR matches `h_drifted` to within 2 dB per-tap accuracy; session retains both IRs in `bleed_ir_history`.
7. **Active vs passive silent-window equivalence:** verify passive-tagged windows are equivalent to active-tagged in IR quality on a synthetic input.

### 7.8 Onset detection F-measure (release-gating)

**Two-tier gate per Q70:**
- Detection F-measure ≥ 0.92 AVP / ≥ 0.88 OOD at IOU = 50 ms (Q53).
- Median absolute timing error on true positives ≤ 15 ms AVP / ≤ 25 ms OOD (Q70).

CI emits both numbers nightly. Both tiers must pass for release. The 0.92 / 0.88 / 15 ms / 25 ms numbers are **provisional pending the Q64 justification memo** that benchmarks against published AVP/PANNs onset literature. PRs that drop AVP F-measure by > 0.005 require explicit reviewer sign-off.

### 7.9 OOD validation

A held-out test set of vocal-percussion recordings from **at least 15 subjects** who are not in AVP-LVT v4. Sourcing guidance:

- Recruit across at least three style categories: Western beatboxing, throat-bass / bass-music styles, and one general/eclectic group.
- Mix experience levels: at least 5 self-identified beginners or hobbyists, at least 5 intermediate-or-better.
- Mix recording conditions: a portion using consumer-grade USB mics, a portion using studio-grade interfaces.
- Same 4-class annotation protocol as AVP.

**Reporting:**
- Macro-F1 on the OOD set, both uncalibrated and after a 5-sample-per-class calibration.
- **Bootstrap 95% confidence interval on the OOD-vs-AVP-LOSO macro-F1 gap.** Resampling at the subject level, 1000 bootstrap iterations.
- Per-style-category breakdown (if N permits within category).
- Per-class F1 on the OOD set, with attention to whether the AVP-strong classes (kick, snare) hold up vs the AVP-weaker classes (open_hat).

**Power note:** N=15 detects a true gap of 10 macro-F1 points with ~80% power at α=0.05 under reasonable variance assumptions. N=5 had ~30% power for the same gap and was not informative.

**Pass criterion (Q50):** the build passes the OOD release gate if **both** of the following hold, evaluated at the operating-point `(softmax_threshold, distance_percentile)` pair selected by the §7.3 sweep:

1. **Missed-unknown-rate on full OOD ≤ 25%.** Of all events in the 15-subject OOD recordings, at most 25% are classified into one of the four trained classes.
2. **False-unknown-rate on AVP held-out ≤ 5%.** Of all events in the AVP held-out folds, at most 5% are wrongly routed to unknown.

If condition 1 is met but condition 2 fails, the gate is too aggressive. If condition 2 is met but condition 1 fails, the gate is too permissive. If neither can be satisfied simultaneously by any sweep operating point, the model itself is the problem — not a threshold issue.

The 25% / 5% thresholds are *provisional* until the Q64 user-impact memo lands. The methodology document is the deliverable; numeric values are outputs.

**OOD pilot (week 1, 5 subjects)** is for substrate decision tiebreaker only. Full OOD (15 subjects) is the release gate.

### 7.10 Tiered eval cadence

Operates on three dataset tiers (Q63):

- **CI tier:** synthetic dataset, in repo. Every PR. Smoke validation only. ~2 minutes. Per Q85, validates pipeline runs, imports, Cholesky round-trip, MIDI parseability, and eval-harness JSON shape — does NOT validate model quality. README and eval-harness `--help` both repeat this caveat; the synthetic-tier CI report names the tier in its output banner.
- **PR-validation tier:** minimum-reproducible (10–20 subjects per fold). Project-hosted. PRs that touch classifier, onset detector, embedding extractor, or eval harness. ~30 minutes.
- **Release tier:** canonical (full AVP + 15-subject OOD). Release candidates only. Gates v1.0 lock. Wallclock per §7.3.

Two cadences share code paths and differ only in input scope:

**Smoke tier (every PR, ~30 min wallclock):**
- 3 fixed AVP subjects (canonical IDs in `voxkit/eval/smoke_subjects.json`).
- LOSO restricted to these 3 subjects (3 folds, single window=200 ms, single C=1.0, single pad=context).
- Calibration uplift at single sample-count=5, single weight=5.0.
- Onset F-measure on the same 3 subjects.
- OOD: 3 fixed OOD subjects.
- BEATs benchmark: skipped.
- Hard fail on: smoke macro-F1 regression > 5 points vs main-branch baseline; onset F-measure < 0.80.

**Full tier (nightly + pre-release, ~11 hours on 8-core):**
- Full §7.3 sweep.
- Full §7.4 calibration uplift sweep.
- Full §7.7 click-bleed bench.
- Full §7.9 OOD validation (all 15 subjects, bootstrap CIs).
- BEATs benchmark.
- Hard fail on: full macro-F1 regression > 2 points vs prior nightly; OOD pass criteria from §7.9.

Both tiers write JSON results to a shared schema; the dashboard reads either. Smoke results gate PR merge; full results gate releases. Output JSON schema versioned (`schema_version`); schema includes a `tier` field.

### 7.11 CI checks

- **Import-graph linter (Q77):** `lint-imports` runs on every PR; violation blocks merge.
- **SPDX header check (Q82):** `reuse` lint runs on every PR; missing or wrong header blocks merge.
- **Audio callback no-allocation regression test:** `tracemalloc` against the audio thread on a 10-minute synthetic session, validating zero on-thread allocations after `open_stream`. Hardened with a second invocation with the CFFI-callback path enabled if it has been wired in.
- **GIL-held-time micro-benchmark:** median per-call GIL hold time < 100 µs at 5 ms buffer size (target < 50 µs but with headroom).
- **Migration table round-trip (Q78):** for each registered `(from, to)` migrator, a synthetic bundle in the `from` schema is migrated, re-loaded, re-saved; the resulting bundle equals what fresh-saving the same logical data would have produced. Catches half-migrations and unregistered legacy fields.
- **MMCSS registration verified on Windows:** thread priority class is "Pro Audio" after `open_stream`.
- **Linux CI smoke test (Phase 1.5):** confirms `Recorder.list_devices()` returns expected devices; `SCHED_FIFO` priority 80 obtainable with appropriate caps.
- **WASAPI → MME fallback test:** simulated WASAPI init failure triggers MME open + UI notification.
- **Cholesky round-trip test:** covariance → Cholesky → reconstruct → max element-wise difference < 1e-10. `mahalanobis_sq_via_cholesky` agrees with explicit-inverse computation on a synthetic dataset (within 1e-8).
- **Full-dim Mahalanobis on PCA-projected synthetic OOD regression test:** construct OOD samples that are zero in PCA-64-retained directions but non-zero in discarded directions; verify full-dim Mahalanobis flags them while PCA-64 Mahalanobis does not. If a future PR moves Mahalanobis back into PCA space, this test fails.
- **Self-test guard tests:** fires on intentionally-bad calibration; does not fire on good calibration.
- **Operating-point selection test:** sweep produces a `(softmax_threshold, distance_percentile)` pair satisfying both Q50 bounds; gate fails (loudly) if no such pair exists.
- **TaxonomyConfig round-trip test:** classifier with a non-default 5-class config (synthetic) trains, predicts, and persists correctly.


---

## 8. Multi-perspective review (consolidated)

The review follows nine perspectives accumulated across versions: **Lin** (DSP/audio engineering), **Priya** (applied ML / OOD detection), **Sam** (architecture/API), **Jordan** (UX/PM), **Marco** (vocal percussion domain), **Alex** (QA/methodology), **Dana** (security/legal), **Casey** (pragmatist), and **Riley** (OSS maintainer, added in v0.10 with the GPL v3 license decision). Voting is one vote per panelist per item; strong consensus = ≥ 6/9, weak = 4–5/9, rejected = ≤ 3/9.

Notes from the panel that have shaped the current spec:

**Lin (DSP):** Click subtraction was the right call (introduced v0.4) and should have been in v0.2. Onset detector now has a release gate that closes the biggest open hole. The FIR-bleed model (v0.5) is physically reasonable; FIR estimation via averaged impulse response is the textbook approach. Mid-session re-estimation (v0.6) closes the LTI-assumption gap; trigger thresholds are heuristic starting points to be revisited after Phase 1 telemetry. The Bluetooth refusal is correct given current scope. Moving the resampler off the audio callback (v0.9, Q51) is the right call and overdue. Threading specifics in Q67 (v0.10) are overdue — "high priority worker thread" without naming MMCSS misses the only knob that matters on Windows for sustained sub-10 ms scheduling. The two-tier onset metric (Q70, v0.10) is right — F-measure alone hides timing-drift regressions. v0.10 specifies MMCSS but is silent on Python's GIL behavior in the audio callback; v0.11's Q67 amendment adds the GIL contract explicitly. Click-bleed handler being unchanged from v0.9 across two architectural revisions is a hidden risk; week-1 tracer bullet (Q80) is overdue.

**Priya (ML):** Moving to PANNs + linear head (v0.4) is the correct industry-standard approach. Dropping pre-emphasis and using context-padding for the embedding window (v0.5) are free fixes. Splitting open/closed hat raises the difficulty slightly but matches user mental model. BEATs benchmark is worth knowing. The OOD scale-up to 15 (v0.6) is the single highest-value spec change. Bootstrap CIs on the gap statistic mean we'll know whether a "gap of 8 points" is real or noise. Temperature scaling is a free correctness fix that stops a category of "the UI lies to me" complaints. Dropping `class_weight='balanced'` in the refit removes a real but subtle bug. The full-dim Mahalanobis fix (v0.9, Q34/Q43 amended) is the most important change in the v0.9 panel — running Mahalanobis in PCA-64 space was the kind of silent failure mode that a properly-staffed v1.0 launch would discover the hard way. Distance to nearest centroid is straightforward best practice. AVP-only covariance under weighted calibration (Q52) is the right call. Self-test overfit guard (Q71, v0.10) finally becomes load-bearing instead of a referenced ghost. The CI-overlap tiebreaker (Q33/Q74) is a strict improvement over the fixed margin. Temperature-scaling held-out source (Q75) closes a methodology hole. The CalibrationRejected wording (Q81 in v0.11) corrects a dishonesty: blaming the user's environment when the actual cause is statistical drift on the held-out fold. Pre-training data license propagation (AudioSet → released weights) is the legal grey area worth surfacing in Q60 amended.

**Sam (Architecture):** PyTorch dependency is a real cost, but unavoidable for embedding inference. ONNX Runtime (v0.5) is a strict improvement. The `CommitHandle` API (v0.6, Q28) is the right shape. Migration prompt for v0.4 sessions (v0.6, Q29) is the conservative choice. Migration sunset cadence (Q55, v0.9) is operationally important. The v0.8 → v0.9 migration banner must be persistent until calibration runs, not dismissable forever — adopted in v0.10. Q66 (taxonomy parameterization, v0.10) is the most operationally important architectural change in v0.10. Win32-isolation in Q62 should be enforced via lint or import-graph checks, not just convention — adopted in v0.11 as Q77 with `import-linter` in CI. Q69 (cutting PCA-64 per-class sweep) is the right scope-discipline call. Q78 (format version field) is overdue; implicit migration on field presence/absence is fine for two versions and brittle by version five. Q76 (inference threading) is the right fix to a v0.10 oversight: Q73 promises a cancel button that doesn't work without an `InferenceWorker`.

**Jordan (UX/PM):** The 3/class minimum and drop-off metric (v0.6) give the team something to learn from for v1.1. Mac and fifth-class as v1.1 candidates are explicit, not buried. First-unknown trigger for the guided tour (v0.9, Q54) is right — the v0.8 ">3 events" trigger reads like over-engineering. Recording-session UX spec (v0.10, Q73) was a missing piece. Migration-banner contradiction resolution (v0.10) is a clean fix. Class-taxonomy parameterization being internal-only in v1.0 is the right scope discipline. Click-bleed quality indicator getting a real metric definition (Q79, v0.11) means users can interpret the colored bar; without this, the indicator is theater. CalibrationRejected dialog text correction (Q81, v0.11) is the kind of small honesty item that adds up to user trust.

**Marco (Domain):** 4-class taxonomy is correct (v0.5). OOD set explicitly addresses the "AVP is narrow" concern. Throat-bass-as-6th-class is the right v1.1 deferral; fast doubles still the v1.1 spike to flag. Alignment MAE tier (Q70, v0.10) most directly affects whether VoxKit feels good to drum into. Detection F-measure in the right range with consistently late onsets produces grooves that "feel off." Click-bleed tracer bullet (Q80, v0.11) matters more than it sounds — if click bleed is misclassified as percussion, every musician using VoxKit for the first time will see false hits and conclude it doesn't work, regardless of how good the trained classifier is.

**Alex (QA):** The smoke tier (v0.6) is what makes the release gates real. The 30-min budget is tight but achievable; if it slips past 45 min it should be re-scoped rather than relaxed. The release-gate rewrite (Q50, v0.9) is the highest-leverage doc fix — v0.8 wording would have been re-litigated at every release-gate review. Q63 dataset tiering (v0.10) is the change that most affects sustainability of the test process. Q77, Q82, and the migration table round-trip test (v0.11) are the quality-gate additions that matter. Synthetic-tier purpose disambiguation (Q85, v0.11) is small but prevents the single most predictable contributor mistake (assuming CI green = ready to merge a model change).

**Dana (Security/Legal):** AudioSet is CC-BY (research). PANNs / BEATs weights derived from AudioSet are MIT per the authors' release. Attribution in About dialog. `signalsmith-stretch` as default (v0.9) is straightforwardly better. WASAPI default is fine — no licensing implications. Q59 (GPL v3-or-later, v0.10) is a clean, defensible license choice for this project type. The "or-later" qualifier preserves Apache 2.0 dependency compatibility. Q60 (license-review scope) is appropriate due-diligence. Q61 (telemetry posture) is appropriate for OSS. Q60 amendment in v0.11 (pre-training data license propagation field) closes a real grey area for AudioSet-pre-trained weights. Q82 SPDX CI enforcement is appropriate due-diligence; an unmaintained SPDX state is worse than no SPDX claim.

**Casey (Pragmatist):** Concerned about scope expansion in Phase 1 (Q62 + Q63 + Q66) but agrees each item is much cheaper now than later. Linux build in Phase 1.5 should be planned concretely; for the hobby pace, the Phase 1.5 commitment can sensibly be "Linux when there's bandwidth," provided the architectural constraint (Q62) is honored — adopted as Q84 in v0.11. Concerned about the storage footprint of the full-dim covariance in the model bundle (16 MB for PANNs, 4 MB for BEATs — halved by Cholesky storage in v0.10). Acceptable for desktop. The twelve adopted items in v0.11 are individually small plus three real items (Q77 import-linter is half a day; Q76 inference worker is a few hours; Q80 click-bleed tracer bullet is genuine new work, on the order of two days). Total v0.11 cost is roughly one work-week.

**Riley (OSS maintainer, added v0.10):** With GPL v3 settled, the highest-leverage change becomes the dataset access plan (Q63). Without three-tier datasets, contributors cannot run the eval, which means the contribution loop is broken. Q62 (architectural portability) is a close second. The license decision itself (Q59) is fine; GPL v3 has good precedent in the audio-OSS world (Audacity, Ardour). CONTRIBUTING.md in week 1 (Q83) is the highest-leverage single OSS-hygiene item not yet captured. Phase 1.5 Linux re-framing (Q84, v0.11) is honest and avoids creating a contributor expectation that gets repeatedly broken by hobby pace. Import-graph enforcement (Q77) is what makes the architectural lock-in real for downstream contributors.

**Rejected items (carried-forward and current):**

- *Adding macOS to Phase 1 in addition to Linux* — OSS-distribution argument for Mac specifically is weaker (most Mac musicians use commercial tools), Phase 1 already loaded. Mac stays Phase 2.
- *Bluetooth full reversal* — kept the hard exclusion at picker time. BT latency makes click-aligned recording structurally unreliable. Recorded as Phase 2 measurement prototype.
- *Cut mid-session re-estimation entirely* — rejected; the LTI assumption fails on session timescales.
- *Replace LR head with fine-tuned final layers* — rejected for v1.0; possible v1.1 path if OOD-vs-AVP gap > 15 macro-F1 points.
- *Per-user `unknown_threshold` UI in v1.0* — rejected; v1.1 candidate.
- *Network telemetry in v1.0* — rejected per Q61; not on the v1.x roadmap.
- *Throat-bass as a 6th trained class in v1.0* — rejected; becomes a `TaxonomyConfig` change in v1.1 per Q66.
- *Joint sweep of L2 regularization C with calibration weight* — rejected on cost-vs-benefit; revisit if calibration-weight eval shows surprise behavior at extreme weights.
- *Cleaner `TrainedClass` / `RuntimeClass` type split* — rejected; code-style only, migration churn is real and the gain is aesthetic.
- *Re-specifying `bleed_ir_history` "two protected slots" inline in v0.11* — rejected; the v0.9 specification is correct and stable.
- *Cutting canonical dataset hosting from v1.0* — rejected; removing the hosted canonical tier turns "first-time contributor reproducing the release-gate eval" from a one-command flow into a multi-step friction flow that disproportionately bounces newcomers.
- *Specifying AVP class-imbalance handling explicitly in spec* — rejected; AVP is approximately balanced after subject-level pooling.


---

## 9. Phased delivery plan

**Phase 0 — Foundations**

- Repo, packaging skeleton, project model, file I/O.
- Recording with metronome, count-in, headphone attestation, clipping guard.
- Waveform display.
- *Parallel:* download AVP-LVT v4, set up data loading + augmentation pipeline.

**Phase 1 — Analysis pipeline**

*Week 1:*
- **Project license file** (`COPYING`, `LICENSE`) added to repo: GPL v3-or-later text. SPDX identifiers added to source files (`SPDX-License-Id: GPL-3.0-or-later`). README updated with license badge.
- **License review for PANNs, BEATs, signalsmith-stretch, Rubber Band** per Q60 amended: one-page memo per dependency covering code license, weight license, commercial-use restrictions, fine-tuning rights, attribution, redistribution, **pre-training data license propagation** (the seventh field). Memos archived in `docs/licenses/`. Criterion: GPL v3 compatibility.
- **A/B audible quality test of `signalsmith-stretch`** on a 10-clip set (Q46). If `signalsmith-stretch` passes, ship. If it fails, integrate Rubber Band as fallback.
- **Dataset access plan** per Q63: confirm AVP and OOD redistribution rights; if confirmed, host canonical datasets on Zenodo or equivalent. Build the minimum-reproducible subset (10–20 subjects per fold). Build the synthetic dataset generator and commit to repo.
- **5-subject OOD pilot** recruitment and recording (Q33 amended).
- **Release-gate justification methodology document** (Q64) drafted. (Hobby scope-down: a "test with 2–3 friends" protocol is acceptable in lieu of formal study.)
- **OOD power sensitivity analysis** (Q36) — half-day analysis, gates full OOD recruitment.
- **CONTRIBUTING.md skeleton committed** (Q83). Some sections may be stubs filled out by week 4.
- **`.importlinter` configuration committed** with Q77 rules; CI job `lint-imports` wired into PR check set.
- **`reuse` configuration and CI job for SPDX header enforcement** (Q82); `.reuse/dep5` listing exempted generated files.
- **`voxkit_format_version` field added to `manifest.json`** (Q78); migration dispatch table populated with all registered migrators.
- **Click-bleed handler tracer bullet** (Q80): simplest LMS / Wiener-based IR estimation, 2-second adaptation against click-only calibration, acceptance > 20 dB null on a deliberately leaky headphone setup. If it fails, escalate to NLMS / RLS before continuing with downstream components.
- **Inference-worker scaffolding** (Q76): `InferenceWorker` thread class with `start()`, `cancel()`, and Qt-signal hooks; not wired to UI yet, but contract published.
- Component 1 (Project & Session): v0.4–v0.10 → v0.11 migration registry.
- Component 2 (Recorder): WASAPI default + resampler worker thread + lock-free ring buffer + MMCSS registration + audio-callback contract per Q67 amended + drop-policy UX + Bluetooth filter + disconnect + sleep + sample-rate handling.
- **Linux CI smoke test** for platform-independent code (Q62) — not a full Linux audio test, just a confirmation that imports, file I/O, and ML pipeline pieces run.
- Tiered eval harness operational from week 1; CI on synthetic, PR validation on minimum-reproducible.
- Test fixtures for silent-window re-estimation (active + passive).

*Week 2:*
- **Q33 substrate decision** using AVP-LOSO with overlapping-CI tiebreaker (Q74); 5-subject pilot OOD invoked only if CIs overlap.
- **Q43 PCA-64 decision** in parallel; LR-only sweep (Q57 / Q69 closed).
- **Q42/Q65 calibration weighting decision** via empirical effective-influence measurement plus sensitivity study against noisy calibration.
- **CPU performance benchmark on each substrate (Q72):** report wall-clock for the 32-bar / 120 BPM session on reference hardware; if neither substrate hits ≤ 0.5×, ONNX optimization passes added before substrate decision.
- Component 3 (Bleed): silent-window-only mid-session re-estimation + passive detection.
- Component 6 (Classifier): temperature scaling per Q75 + composite gate with full-dim distance-to-nearest Mahalanobis using Cholesky factor (Q68) + AVP-only covariance + self-test overfit guard (Q71) + TaxonomyConfig-driven class set (Q66).
- Component 6: GIL-contract documentation block in `voxkit/audio/recorder.py` (Q67 amended) recording the active callback path.
- Component 11: `InferenceWorker` wired to the recording-session progress dialog with cancel semantics per Q76.
- Component 11: CalibrationRejected dialog wording updated per Q81; diagnostic-file logging of the macro-F1 delta wired up.
- Component 1: `voxkit_format_version` write path on save; migration table entry tested via the round-trip test.
- OOD subject recruitment for full OOD (kicks off mid-week 2 once Q33 is locked).
- **User-impact study recruiting** for Q64 release-gate justification memo (n ≥ 5 musicians; relaxable for hobby pace).

*Week 3–8:*
- Components 4, 5, 7 implementation.
- Component 11 (Editor UI): 5-lane piano roll, taxonomy disclosure, bleed banners, first-run guided tour, persistent migration banner, recording-session progress UX.
- Component 9 (MIDI exporter): unknown-class export toggle; reads from `TaxonomyConfig`.
- AVP F-measure release gate (§7.8) operational with detection + alignment tiers (Q70).
- LOSO eval with full sweep including explicit (missed-unknown, false-unknown) operating-point selection (Q50).
- Migration matrix tests v0.4–v0.10 → v0.11.
- First FL Studio import test.
- Full OOD recordings (15 subjects) completed; release gate evaluated per Q50.
- User-impact study findings folded into release-gate memos (Q64); if any threshold needs revision, this happens before v1.0 lock.

**Phase 1 critical-path implications:**
- Project license file in repo + SPDX headers from week 1.
- `import-linter` and `reuse` CI jobs from week 1; both block merge.
- `manifest.json` version field and migration table from week 1.
- Click-bleed tracer bullet from week 1; escalation path defined if simplest IR estimation fails the > 20 dB acceptance threshold.
- `InferenceWorker` scaffolding from week 1, wired to UI in week 2.
- Component 2 (Recorder) must land before any OOD recordings.
- License-review memos for PANNs, BEATs, signalsmith-stretch, Rubber Band — week 1.
- Q36 OOD power memo — week 1, gates full OOD recruitment.
- Dataset access infrastructure — week 1.
- 5-subject pilot OOD recording — week 1, gates Q33 if AVP CIs overlap.
- Q33 substrate decision — week 2.
- CPU performance benchmark on each substrate — week 2 (Q72 input to Q33).
- Q42/Q65 calibration-weight decision — week 2.
- Q43 PCA-64 (LR-only) — week 2.
- Component 6 (Classifier) — Cholesky-backed full-dim Mahalanobis + self-test guard + temperature held-out + TaxonomyConfig.
- Component 11 (Editor) — 5-lane piano roll + first-unknown tour + persistent migration banner + recording-session progress UX.
- Component 12 (Eval harness) — operating-point selection, alignment-MAE tier, CPU perf benchmark from week 2.
- Migration matrix test (v0.4–v0.10 → v0.11) — every PR from week 1.

**Phase 1.5 — Linux build (as bandwidth permits, per Q84):**
- Wire ALSA / PipeWire backend into `Recorder` (Q62); rest of the codebase already Linux-clean from week-1 CI.
- `SCHED_FIFO` + `setcap` install documentation.
- Linux audio device test matrix: 5 representative devices (USB class-compliant interface, built-in laptop mic, USB microphone, Focusrite/PreSonus-class interface, headset).
- Linux build distribution: AppImage + Flatpak.

**Phase 2 — macOS build:** CoreAudio backend; macOS code signing and notarization workflow.

**Phase 3 — Playback hardening (within Phase 1 scope):**
- Multi-stem mixer with mute/solo.
- Loop region.
- Variable-speed playback (debounced `signalsmith-stretch` re-render).
- 3/4 + 4/4 verified end-to-end.

**Phase 4 — Hardening (within Phase 1 scope):**
- Held-out user accuracy eval.
- Windows packaging + SmartScreen documentation.
- 2–3 friend alpha tests.
- v1.0 release.

---

## 10. Remaining open questions

1. Calibration sample count default. Empirical, eval (§7.4) determines.
2. Whether to expose embedding-space nearest-neighbor browser as a debugging aid in the editor. Minor UX question.
3. Whether OOD-vs-AVP gap warrants fine-tuning the last 2 PANNs layers in v1.1 (or v1.0 if gap is large). Decided after Phase 1 numbers are in.
4. Whether to expose the FIR tap count as a user-facing setting. Default 64 should work for ~all setups; defer to user feedback.
5. Mid-session re-estimation triggers (32 bars / 2% guard fires) are heuristic. Revisit after Phase 1 telemetry.
6. Mac platform support. Phase 2 deliverable; decision contingent on Phase 1 demand signal from in-app feedback.
7. Per-user `softmax_threshold` and `distance_thresholds` adjustability — v1.1 candidate.
8. Throat-bass / bass-music as 6th and 7th trained classes — v1.1 candidate. Now a `TaxonomyConfig` change rather than a migration per Q66.
9. Opt-in network telemetry — removed from v1.x roadmap per Q61. Reintroduction requires constraints in Q61.
10. (Closed) PCA-64 prototype — closed in v0.8; LR-only bake-off in v0.10.
11. OOD-tuned threshold (vs AVP-tuned) — v1.1 candidate. The Mahalanobis distance threshold being AVP-anchored partially addresses this.
12. Whether passive silence detection should relax the active-detection RMS margin. Decision after Phase 1 telemetry.
13. Whether Mahalanobis distance threshold should adapt per-user. v1.1 candidate.
14. User-confirmed unknown feedback loop. Phase 2 prototype.
15. Audio device clock-drift compensation (v0.9 prototype task, Phase 2). Long sessions (>10 min) may exhibit drift between the audio device's sample clock and the system clock used for click-track scheduling. Prototype deliverable: Phase 2. If median drift exceeds 1 ms over 32 bars, design a compensation scheme. Decision: ship in v1.1 if compensation is needed; otherwise drop.
16. (Closed) Per-class Mahalanobis covariance under PCA-64 — closed in v0.10 per Q69.
17. Bluetooth audio device support. Phase 2 prototype: measure end-to-end latency on representative modern Bluetooth devices (LL-AAC, aptX LL, LC3 codecs). If median end-to-end latency < 50 ms with low jitter (P95 < 100 ms), design "Bluetooth allowed but flagged" mode for v1.1 or v1.2.
18. v1.0 user-facing class taxonomy editing. Q66 parameterizes the architecture; v1.0 ships default 4-class config without UI for editing. v1.1 candidate.
19. Distribution mechanism for canonical eval datasets if AVP / OOD redistribution rights are not confirmed in week 1. If not, the download-script path needs a contributor-experience review.
20. Qt binding choice: PyQt6 vs PySide6. Both clean under GPL v3. Defer to implementer; decide once on project start to avoid mixed imports.
21. Lock-free SPSC ring buffer named implementation reference. v0.11 keeps the contract-only specification (Q67 + Q76); commit to a named implementation if the audio-thread no-allocation tracemalloc test or dropped-buffer-rate metric flags a problem under real-world load.
22. Per-class F1 reporting alongside macro-F1 in release-gate output. Cheap; deferred because macro-F1 + per-class operating-point selection in §7.3 already lets a closed/open-hat regression be detected. Revisit if vocal-percussion-specific confusability shows up in v1.0 telemetry.
23. Re-run inference from the editor without re-recording. Real UX value but adds Component 11 surface area; deserves a deliberate v1.1 design pass rather than a v1.0 retrofit.


---

## 11. Component specifications (build-ready)

### Component diagram (data flow, left to right)

```
[Recorder] ──▶ raw_audio_buffer
   │           (Bluetooth filtered; DeviceDisconnected and OS-sleep handled;
   │            WASAPI default with MME fallback; audio callback ONLY copies
   │            to lock-free SPSC ring buffer per Q67 contract: float32 mono
   │            native byte order; GIL held < 50 µs/call; CFFI path as
   │            escalation; voxkit.audio.recorder is the ONLY module permitted
   │            to import sounddevice — enforced by import-linter in CI per Q77)
   ▼
[ResamplerWorker]  (DEDICATED THREAD, MMCSS "Pro Audio" on Windows,
   │               SCHED_FIFO priority 80 on Linux (Phase 1.5))
   │               (scipy.signal.resample_poly with pre-allocated filter state;
   │                per-buffer latency budget per Q67; 10 ms worker budget)
   ▼
[ClickBleedHandler] ──▶ cleaned_audio
   │  ▲    (FIR subtraction; mid-session re-estimation fed by ACTIVE +
   │  │     PASSIVE silent-window detection (Q47/Q48); bleed_ir_history
   │  │     retains TWO protected slots (Q49); Q79 quality indicator metric:
   │  │     post-subtraction click residual ratio in dB)
   │  └─── re-estimation trigger (every 32 bars OR click-guard >2% in last 8 bars)
   ▼
[OnsetDetector] ──▶ onset_samples[]
   │      (release-gated at F ≥ 0.92 AVP / 0.88 OOD per Q53;
   │       AND alignment MAE ≤ 15 ms AVP / ≤ 25 ms OOD per Q70)    ┐
   ▼                                                                 │
[EmbeddingExtractor] ──▶ embeddings[N×D_full]                       │ Q76: ALL THREE
   │  (D_full = 2048 PANNs / 768 BEATs; substrate locked in          ├─ run on the dedicated
   │   Phase 1 week 2; embeddings kept FULL-DIM throughout;          │  InferenceWorker thread.
   │   PCA projection (if Q43 ships) applied only to LR-head branch) │  Cancel via threading.Event
   ▼                                                                 │  checked between phases
[Classifier] ──▶ (class_id, score)[]                                │  and per-onset; main thread
   │       ◀── [CalibrationManager] (returns CommitHandle;           │  runs Qt event loop only.
   │            self-test overfit guard per Q71; Q81 CalibrationRejected) ┘
   │
   │   Composite gate (Q34):
   │     LR_input = PCA @ emb if pca_present else emb
   │     probs = softmax(LR(LR_input) / T)              [T from Q75 disjoint fold]
   │     softmax_unknown = max(probs) < softmax_threshold
   │     dists = mahalanobis_sq_via_cholesky(emb, c, L_pooled)
   │             for c in centroids_full_dim             [FULL-DIM, Cholesky per Q68]
   │     nearest = argmin(dists)
   │     distance_unknown = sqrt(dists[nearest]) > distance_thresholds[nearest]
   │     return 'unknown' if softmax_unknown OR distance_unknown
   │            else taxonomy.classes[argmax(probs)]     [TaxonomyConfig per Q66]
   ▼
[TempoGridEngine] ──▶ quantized_events[]
   ▼
[MIDIExporter] ──▶ .mid file
   │       (taxonomy.classes drives note mapping; unknowns excluded by default;
   │        opt-in exports to configurable GM note; ExportSummary reports
   │        events_skipped_unknown count)
   ▼
[Editor UI] ◀── [PlaybackEngine]
   │   (5 lanes by default, configured by TaxonomyConfig; unknown lane:
   │    neutral grey, dashed separator, "?" glyph; "Best guess: snare (32%) —
   │    below confidence threshold" tooltip; FIRST-RUN tour on FIRST UNKNOWN
   │    per Q54; bleed banners; sleep/disconnect modals; PCA-Mahalanobis
   │    recalibration banner persistent until calibration runs; recording-session
   │    progress modal per Q73 backed by InferenceWorker per Q76)
   ▼
[ProjectFile (voxkit_format_version stamped on save;
   │           migration via registered (from,to) → migrator table per Q78)]
```

`EvalHarness` (Component 12) sits outside the runtime path. Three dataset tiers per Q63.

---

### Component 1: Project & Session

**Purpose:** Single source of truth for a session.

**Public API:**

```python
@dataclass
class TimeSignature:
    numerator: int
    denominator: int

ClassId = Literal["kick", "snare", "closed_hat", "open_hat", "unknown"]
TrainedClassId = Literal["kick", "snare", "closed_hat", "open_hat"]

@dataclass
class TaxonomyConfig:
    """Q66: single source of truth for class set."""
    classes: tuple[TrainedClassId, ...]  # default: ("kick", "snare", "closed_hat", "open_hat")
    midi_mapping: dict[TrainedClassId, int]  # default: GM drum map
    unknown_class_id: ClassId = "unknown"

    @classmethod
    def default_v1_0(cls) -> "TaxonomyConfig":
        return cls(
            classes=("kick", "snare", "closed_hat", "open_hat"),
            midi_mapping={"kick": 36, "snare": 38, "closed_hat": 42, "open_hat": 46},
        )

@dataclass
class Event:
    onset_sample: int
    quantized_time: float
    class_id: ClassId
    score: float        # calibrated probability (renamed from confidence in v0.6)
    velocity: int
    embedding: Optional[np.ndarray]  # shape (D_full,), float16

@dataclass
class BleedIR:
    taps: np.ndarray       # float32, length 32–128
    sample_rate: int
    device_id: str
    measured_at: datetime
    origin: Literal["setup", "midsession_reestimate_active",
                    "midsession_reestimate_passive", "migrated_pending_recapture"]
    residual_db: float     # Q79: post-subtraction click residual ratio in dB

@dataclass
class TemperatureCalibration:
    """Q27: temperature scalar for softmax calibration."""
    T: float               # fit on disjoint held-out fold per Q75

@dataclass
class MahalanobisFullDim:
    """Q68: Cholesky factor instead of inverse covariance."""
    class_centroids: np.ndarray       # (n_classes, D_full)
    pooled_cov_cholesky: np.ndarray   # (D_full, D_full), lower triangular
    distance_thresholds: np.ndarray   # (n_classes,) per-class 95th percentile on AVP

@dataclass
class ProjectManifest:
    """Q78: explicit format version field."""
    voxkit_format_version: str        # e.g., "0.11"

@dataclass
class Session:
    bpm: float
    time_signature: TimeSignature
    bars: int
    sample_rate: int                  # inference rate (16 kHz)
    recording_sample_rate: int
    recording_audio_api: Literal["WASAPI", "MME", "ALSA", "PipeWire", "CoreAudio"]
    audio: np.ndarray
    bleed_ir: Optional[BleedIR]       # current active IR
    bleed_ir_history: list[BleedIR]   # all IRs estimated this session (20 max, 2 protected slots)
    bleed_gate_overridden: bool       # Q23
    embedding_model_id: str           # e.g. "panns_cnn14_16k_v1"
    output_calibration: Optional[TemperatureCalibration]  # Q27
    mahalanobis_full_dim: Optional[MahalanobisFullDim]    # Q68
    pca_matrix_present: bool          # affects LR head only (Q43)
    softmax_threshold: float          # default 0.45 (Q34)
    calibration_weight: float         # default per Q42/Q65 sensitivity study
    taxonomy: TaxonomyConfig          # Q66
    events: list[Event]
    quantize_grid: str
    quantize_strength: float
    dropped_buffer_count: int         # Q67 telemetry

# Migration dispatch table (Q78)
Migrator = Callable[[dict], dict]
MIGRATIONS: dict[tuple[str, str], Migrator] = {
    ("0.4",  "0.5"):  migrate_0_4_to_0_5,
    ("0.5",  "0.6"):  migrate_0_5_to_0_6,
    ("0.6",  "0.7"):  migrate_0_6_to_0_7,
    ("0.7",  "0.8"):  migrate_0_7_to_0_8,
    ("0.8",  "0.9"):  migrate_0_8_to_0_9,
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

`migrate_0_10_to_0_11_stamp_version` is a no-op on data, only setting `voxkit_format_version: "0.11"` and (if missing) populating the field on legacy bundles. The Cholesky conversion remains in `migrate_0_9_to_0_10_cholesky`.

**File format (zip-bundle .vxk):**
- `manifest.json` — metadata including `voxkit_format_version`, `recording_audio_api`, `dropped_buffer_count`, taxonomy config reference. SPDX header (`SPDX-License-Id: GPL-3.0-or-later`).
- `audio.wav` — recording, 32-bit float, mono.
- `bleed_ir.npy` — current active FIR taps if present.
- `bleed_ir_history.npz` — historical IRs keyed by `measured_at` ISO string, with two protected slots.
- `mahalanobis_full_dim.npz` — `class_centroids_full_dim`, `pooled_cov_cholesky_full_dim` (lower triangular, Cholesky per Q68), `distance_thresholds`. Always present in v0.9+ sessions.
- `pca_matrix.npz` — present only when `pca_matrix_present == True`. Used for LR head only (Q43).
- `events.json` — events without embeddings (uses `score`).
- `embeddings.npz` — embeddings keyed by event index, float16 (optional, regenerable).
- `taxonomy_config.json` — `TaxonomyConfig` (Q66). v1.0 ships with default 4-class config.
- `temperature_calibration.json` — temperature scalar T (Q27).

**Dependencies:** numpy, dataclasses-json, zipfile.

**Tests:**
- Round-trip save/load with and without `bleed_ir`, with and without `output_calibration`.
- Migration table round-trip (§7.11): for each registered `(from, to)`, a synthetic bundle in the `from` schema is migrated, re-loaded, re-saved, and compared structurally to a fresh save of the same logical data.
- Legacy bundle (no `voxkit_format_version`) loads as `"0.4"` and walks all migrations.
- v0.6 → v0.7 migration: synthetic v0.6 session with PlattCalibration loads cleanly; `output_calibration = None` post-load; banner state set.
- v0.8 → v0.9 migration: if `pca_matrix_present == True`, `mahalanobis_full_dim` set to `None` on load; persistent banner state set.
- v0.9 → v0.10 migration: Cholesky round-trip (covariance → Cholesky → reconstruct → max element-wise difference < 1e-10).
- v0.10 → v0.11 migration: `voxkit_format_version` field stamped on save.
- `bleed_ir_history` cap: 25 appends → length 20; both protected slots retained; oldest unprotected entries evicted.
- TaxonomyConfig round-trip: non-default 5-class config trains, predicts, and persists correctly.

**Failure modes:**
- Version mismatch newer than v0.11 → fail with explicit "session created in newer VoxKit" message.
- Corrupted zip → audio-only recovery mode.


---

### Component 2: Recording subsystem

**Purpose:** Audio capture and click playback.

**Public API:**

```python
class DeviceDisconnected(Exception):
    """Raised when the active input stream loses its device.
    Recording state is paused; orchestrator surfaces a modal."""
    device_id: str
    last_good_sample_index: int

class OSSleepEvent(Exception):
    """Raised on platform sleep notification while recording is active.
    Recording state is paused; on wake, device is re-validated."""
    sleep_at: datetime

@dataclass
class AudioDevice:
    id: str
    name: str
    is_bluetooth: bool    # detected via OS device-class enumeration
    sample_rates: list[int]

@dataclass
class DeviceInfo:
    id: str
    name: str
    is_bluetooth: bool
    sample_rates: list[int]
    transport_known: bool

class Recorder:
    INFERENCE_SAMPLE_RATE: int = 16_000
    SUPPORTED_DEVICE_RATES: tuple[int, ...] = (16_000, 32_000, 44_100, 48_000, 88_200, 96_000)
    PREFERRED_API_WINDOWS: str = "WASAPI"    # Q58
    FALLBACK_API_WINDOWS: str = "MME"
    PREFERRED_API_LINUX: str = "PipeWire"   # Q62, Phase 1.5
    FALLBACK_API_LINUX: str = "ALSA"
    PREFERRED_API_MACOS: str = "CoreAudio"  # Q62, Phase 2

    AUDIO_CALLBACK_PATH: Literal["python_default", "cffi_hardened"] = "python_default"
    # Q67 amended: default Python callback with NumPy buffer copy and atomic counter,
    # GIL held < 50 µs/call. CFFI RawInputStream callback documented as hardened
    # escalation path. v1.0 ships "python_default". The active path is documented
    # in the docstring at the top of voxkit/audio/recorder.py.

    def list_devices(self, exclude_bluetooth: bool = True) -> list[DeviceInfo]:
        """Enumerates available input devices.
        Bluetooth devices excluded by default per Q24.
        Set exclude_bluetooth=False only in dev/debug builds."""

    def open_stream(self, device_id: str) -> AudioStream:
        """Tries preferred API for current platform first; falls back to
        MME (Windows) or ALSA (Linux) with a one-time UI notification if
        preferred API init fails. Pre-allocates resampler filter state.
        Spawns the resampler worker thread with platform-appropriate priority
        (MMCSS Pro Audio on Windows, SCHED_FIFO on Linux)."""

    def get_click_pulse(self) -> np.ndarray: ...
    def get_click_positions(self, req: RecordingRequest) -> list[int]: ...
    def install_sleep_handler(self, callback: Callable[[OSSleepEvent], None]) -> None: ...
    def handle_disconnect(self) -> DeviceDisconnected: ...
    def get_dropped_buffer_count(self) -> int: ...   # Q67
```

**Audio callback contract (Q67 amended with Q76 GIL guidance):**

```python
# Default path (v1.0):
def _python_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    # GIL-held time per call: < 50 µs typical at 5–10 ms buffers on reference CPU.
    # NumPy releases the GIL during the memcpy in __setitem__.
    # Contract: indata is float32, mono after device-side downmix, native byte order.
    # No allocation, no logging, no system calls on the hot path.
    if not ring.try_push(indata):
        dropped_buffer_count.fetch_add(1)  # atomic increment
```

**Threading model (Q51, Q67):**

```
Audio callback thread (real-time):
    on_buffer(device_buffer):
        ring_buffer.try_push(device_buffer)   # lock-free, non-blocking
        # That's it. No allocations, no resampling, no logging on the hot path.

Resampler worker thread (high priority, not real-time):
    # Windows: MMCSS "Pro Audio" via AvSetMmThreadCharacteristicsW
    # Linux (Phase 1.5): SCHED_FIFO priority 80, requires CAP_SYS_NICE
    while running:
        device_buffer = ring_buffer.pop_blocking(timeout=10ms)
        elapsed_start = time.perf_counter()
        resampled = resample_poly_with_state(device_buffer, pre_allocated_state)
        downstream_queue.push(resampled)
        elapsed_ms = (time.perf_counter() - elapsed_start) * 1000.0
        budget_ms = compute_budget_ms(device_buffer_duration_ms)  # Q67
        if elapsed_ms > budget_ms:
            consecutive_overruns += 1
            if consecutive_overruns > 100:
                emit_alarm("Resampler worker over budget")
        else:
            consecutive_overruns = 0
```

The ring buffer is a single-producer (audio callback) / single-consumer (worker) lock-free queue. Capacity: 2 seconds of device-rate audio. If the worker stalls and the ring buffer fills, the audio callback drops the buffer (with a counter; user-facing surface is a warning if drops exceed 0.1% over a session).

**Drop-policy UX (Q67):** `dropped_buffer_count` exposed via `get_dropped_buffer_count()`. Session tracker computes rolling 30-second drop rate; if > 0.1%, end-of-session modal: "VoxKit dropped X% of audio during this recording due to system load. The recording may have gaps. Consider closing other applications and re-recording, or proceed to editing."

**Implementation notes:**
- Disconnect detection: `AUDCLNT_E_DEVICE_INVALIDATED` HRESULT during the WASAPI callback is the canonical disconnect signal. The handler captures `last_good_sample_index` from the session buffer write pointer at the moment of the exception.
- Sleep handling on Windows: register a hidden message-only window via `pywin32` to receive `WM_POWERBROADCAST`. On `PBT_APMSUSPEND`, invoke the registered callback. On `PBT_APMRESUMEAUTOMATIC` / `PBT_APMRESUMESUSPEND`, the orchestrator re-enumerates devices and validates that the previously-active device is still present at the same `device_id`.
- Bluetooth detection on Windows uses the WASAPI device-class enumeration (devices with class `KSCATEGORY_AUDIO` and a Bluetooth transport flag are excluded). A `--allow-bluetooth` CLI flag enables the dev/debug path.

**Dependencies:** sounddevice, numpy, pywin32 (Windows), comtypes (Windows).

**Tests:**
- Unit: device-list filter excludes a synthetic Bluetooth device, includes a synthetic USB device.
- Audio callback latency under load: with the worker artificially delayed (mock 50% over budget), verify the audio callback still completes within its real-time budget and ring-buffer overflow is detected and counted.
- Filter pre-allocation: verify no allocations occur on the audio callback thread during a 10-minute synthetic session (use Python's `tracemalloc` against the audio thread; expected zero on-thread allocations after `open_stream`).
- GIL-held-time micro-benchmark: median per-call GIL hold time < 100 µs at 5 ms buffer size.
- WASAPI → MME fallback: simulated WASAPI init failure triggers MME open + UI notification.
- MMCSS registration verified on Windows: thread priority class is "Pro Audio" after `open_stream`.
- Synthetic disconnect during recording — verify `DeviceDisconnected` raised, session buffer contains audio up to `last_good_sample_index`.
- Synthetic sleep/wake — verify `OSSleepEvent` raised, post-wake device re-validation succeeds when the same device is present.
- Drop policy: synthetic session with worker artificially delayed (mock 50% over budget) confirms `dropped_buffer_count` increments correctly and the end-of-session modal fires.
- Linux smoke test (Phase 1.5): `SCHED_FIFO` priority 80 obtainable with appropriate caps; documented in install docs.

**Failure modes:**
- WASAPI enumeration fails to expose the transport flag on some old drivers → device shows as "transport unknown" and is excluded conservatively. Documented.
- User has only Bluetooth devices available → picker shows "No supported audio devices found" with link to device-support documentation.
- Sleep handler registration fails on locked-down Windows policies → log warning, continue without sleep handling. Documented; affects some enterprise environments.


---

### Component 3: Click bleed handler

**Purpose:** Estimate the bleed-path impulse response, subtract bleed from recordings, and re-estimate mid-session as the bleed path drifts (Q25, Q35, Q47, Q48, Q49, Q79, Q80).

**Public API:**

```python
@dataclass
class BleedEstimate:
    ir: np.ndarray
    residual_db: float      # Q79: post-subtraction click residual ratio in dB
    alignment_offset: int

@dataclass
class ReestimationConfig:
    bars_per_check: int = 32              # elapsed-bar trigger
    guard_fire_threshold: float = 0.02   # 2% of click positions in window
    guard_window_bars: int = 8           # rolling window for guard rate
    rolling_buffer_bars: int = 16        # how much click-aligned audio to retain
    silent_window_min_count: int = 8     # min silent windows needed before re-estimating
    silent_window_rms_margin_db: float = 6.0  # active-silent threshold (Q47)
    passive_run_length: int = 4          # consecutive active-silent windows for passive (Q48)
    passive_window_count: int = 8        # sliding window size (Q48)
    passive_window_ms: float = 50.0      # each window length in ms (Q48)

class ClickBleedHandler:
    def estimate_ir(
        self,
        silent_recording: np.ndarray,
        click_pulse: np.ndarray,
        click_positions: list[int],
        sample_rate: int,
        ir_length: int = 64,
        search_window_samples: int = 2000,
    ) -> Optional[BleedEstimate]:
        """Initial setup estimation. Averages click-pulse-aligned windows.
        Acceptance criterion: residual_db ≥ 20 dB null (Q79). Returns None
        if the bleed gate fails (residual too high) and user hasn't overridden."""

    def subtract(
        self,
        recording: np.ndarray,
        click_pulse: np.ndarray,
        click_positions: list[int],
        ir: np.ndarray,
    ) -> np.ndarray:
        """Apply FIR subtraction. Returns cleaned audio."""

    def push_rolling_window(
        self,
        recording_segment: np.ndarray,
        click_positions_in_segment: list[int],
        bars_in_segment: int,
    ) -> None:
        """Append to the rolling buffer used for re-estimation."""

    def tag_active_silent_windows(
        self,
        recording_segment: np.ndarray,
        click_positions: list[int],
        noise_floor: float,
        config: ReestimationConfig,
    ) -> list[bool]:
        """Returns per-window silence flag per Q47.
        A window is actively silent if post-click RMS (excluding ±15 ms
        around click position) is below noise_floor + config.silent_window_rms_margin_db."""

    def tag_passive_silent_windows(
        self,
        active_silent_flags: list[bool],
        config: ReestimationConfig,
    ) -> list[bool]:
        """Returns per-window passive silence flag per Q48.
        Passive sliding window is over config.passive_window_count windows of
        config.passive_window_ms ms each. A window is passively silent if
        a run of ≥ config.passive_run_length consecutive active-silent windows
        precedes it (~200 ms of consecutive silence at the click cadence)."""

    def maybe_reestimate(
        self,
        click_pulse: np.ndarray,
        sample_rate: int,
        guard_fire_rate_recent: float,
        bars_since_last_estimate: int,
        config: ReestimationConfig = ReestimationConfig(),
    ) -> Optional[BleedEstimate]:
        """Returns a new BleedEstimate if a trigger fires (elapsed-bar OR guard-rate),
        else None. Uses silent-window averaging only from the rolling buffer.
        Returns None if fewer than config.silent_window_min_count silent windows
        are available — in that case, the orchestrator prompts the user for a
        4-bar silent re-capture rather than silently re-estimating."""

    def compare_estimates(
        self,
        rolling_buffer: np.ndarray,
        click_positions: list[int],
        click_pulse: np.ndarray,
        ir_old: np.ndarray,
        ir_new: np.ndarray,
    ) -> dict:
        """Compare residuals of two IRs against the rolling buffer.
        Returns {'old_residual_db': float, 'new_residual_db': float,
                 'recommended': 'old' | 'new'}. Used to validate that
        re-estimation actually improved the fit."""

    def get_quality_attenuation_db(
        self,
        cleaned_audio: np.ndarray,
        calibration_audio: np.ndarray,
        click_positions: list[int],
    ) -> float:
        """Q79: compute the click-bleed quality indicator in dB.
        Returns 20 * log10(rms(cleaned[click_windows]) / rms(calibration[click_windows])).
        More positive is better attenuation. Green ≥ 20 dB, yellow 10–20 dB, red < 10 dB.
        Below 10 dB triggers the bleed banner in the UI."""
```

**Implementation notes:**
- The rolling buffer stores the most recent ~16 bars of recorded audio plus the corresponding click positions. New segments are appended; old segments are evicted FIFO.
- `maybe_reestimate` is called by the runtime orchestrator on a timer (cheap; just checks counters). When a trigger fires, the actual estimation runs on a worker thread.
- Hot-swap atomicity: the active IR is held behind an `RWLock`-style guard; the swap is a single pointer assignment after the new IR's residual has been validated against the rolling buffer via `compare_estimates`.
- If `compare_estimates` returns `recommended='old'` (i.e., the new IR is actually worse), the new IR is logged but not activated; the trigger is treated as a transient.
- `bleed_ir_history` FIFO eviction per Q39/Q49: two protected slots (most recent setup-origin entry + most recent active-silent re-estimation entry) are never evicted.

**Dependencies:** numpy, scipy.signal, threading.

**Tests:**
- All initial-setup FIR estimation tests (per-tap recovery within 2 dB at SNRs of 0, 10, 20, 30 dB).
- Residual energy < 1% of original bleed energy at SNR ≥ 20 dB after subtraction.
- Alignment robustness: synthetic delay of 500 samples should be recovered.
- `tag_active_silent_windows`: correctly flags a 200 ms region below RMS threshold as active-silent; does not flag a region above threshold.
- `tag_passive_silent_windows`: correctly identifies passive-silent regions when preceded by ≥ 4 consecutive active-silent windows.
- `push_rolling_window` + `maybe_reestimate` integration with synthetic drift (IR changes mid-recording). Expect re-estimation to fire within 8 bars and recover the new IR within 2 dB per-tap.
- `compare_estimates` correctly identifies the better IR on a held-out validation segment.
- Hot-swap thread-safety test — high-frequency `subtract` calls during a `maybe_reestimate` worker run should never see a partially-updated IR.
- Active vs passive silent-window equivalence: verify passive-tagged windows are equivalent to active-tagged in IR quality on a synthetic input.
- `get_quality_attenuation_db` returns 0 dB for identical audio, positive dB for cleaned audio with lower RMS in click windows.
- `bleed_ir_history` cap: 25 appends → length 20; both protected slots retained; oldest unprotected entries evicted first. Protected slots are never evicted.
- Tracer-bullet integration test (Q80, week 1 acceptance): on synthetic "leaky open-back headphones" simulation, 2-second adaptation achieves > 20 dB null on click-only follow-up segment.

**Failure modes:**
- User performs *during* a re-estimation window → estimation contaminated. Mitigation: active-silent window detection (Q47) rejects windows with post-click RMS above noise_floor + 6 dB.
- Re-estimation worker thread starves under heavy concurrent load → trigger fires but estimation lags. Mitigation: log the lag; if lag exceeds 2 bars, surface in dev-mode diagnostic file.
- Trigger fires but insufficient silent windows → user prompted for a 4-bar silent re-capture rather than proceeding with contaminated data.


---

### Component 4: Onset detector

**Purpose:** Detect vocal percussion onsets in the click-subtracted audio; expose click-guard firing rate to Component 3.

**Public API:**

```python
@dataclass
class OnsetDetectorConfig:
    pre_max_ms: float = 10.0
    post_max_ms: float = 10.0
    pre_avg_ms: float = 80.0
    post_avg_ms: float = 80.0
    click_guard_window_ms: float = 30.0  # suppress onsets within ±15 ms of click
    noise_gate_db: float = 6.0           # drop onsets below 6 dB above noise floor
    noise_floor_estimation_ms: float = 200.0  # first N ms assumed silent

class OnsetDetector:
    def detect(
        self,
        cleaned_audio: np.ndarray,
        click_positions: list[int],
        sample_rate: int,
        sensitivity: float = 0.07,  # `delta` in librosa.onset.onset_detect
        config: OnsetDetectorConfig = OnsetDetectorConfig(),
    ) -> list[int]:
        """Returns list of onset sample positions.
        Internally: (1) compute noise floor from first config.noise_floor_estimation_ms ms;
        (2) run librosa.onset.onset_detect; (3) apply click-window guard;
        (4) apply noise-floor gate."""

    def click_guard_fire_rate(
        self,
        click_positions: list[int],
        raw_onsets: list[int],
        window_bars: int,
        bpm: float,
        sample_rate: int,
    ) -> float:
        """Returns the fraction of click positions that had a raw onset within
        ±15 ms, over the most recent window_bars bars. Read by Component 3
        as the guard_fire_rate_recent input to maybe_reestimate."""
```

**Two-tier release gate (Q53, Q70):**
- F-measure ≥ 0.92 on AVP at IOU = 50 ms.
- F-measure ≥ 0.88 on OOD at the same IOU.
- Median absolute timing error on true positives ≤ 15 ms AVP / ≤ 25 ms OOD.

Below either threshold, the build does not pass the §7.8 release gate.

**Dependencies:** librosa, numpy.

**Tests:**
- Synthetic recording with known onset positions: verify detection within ±15 ms timing tolerance.
- Click-window guard: verify no onset is reported within ±15 ms of a click position on a synthetic click-heavy recording.
- Noise-floor gate: verify onsets below 6 dB above noise floor are dropped.
- `click_guard_fire_rate`: verify correct rate calculation on synthetic data with known click positions and raw onsets.

**Failure modes:** Quiet hats below noise gate → missed onsets (acknowledged; mitigated by sensitivity slider). Breath transients → spurious onsets (mitigated by noise gate and click-guard).

---

### Component 5: Embedding extractor

**Purpose:** Extract embeddings from the frozen PANNs CNN14 or BEATs model for each detected onset.

**Public API:**

```python
@dataclass
class EmbeddingExtractorConfig:
    window_ms: float = 200.0          # extraction window length (sweep winner from §7.3)
    padding: Literal["context", "zero"] = "context"  # Q21
    model_input_length_s: float = 1.0
    target_sample_rate: int = 16_000

class EmbeddingExtractor:
    def __init__(self, model_path: Path, model_id: str): ...

    def extract_all(
        self,
        audio_16khz: np.ndarray,
        onset_samples_16khz: list[int],
        config: EmbeddingExtractorConfig = EmbeddingExtractorConfig(),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (embeddings, rms_values).
        embeddings: (N, D_full) float32, where D_full = 2048 (PANNs) or 768 (BEATs).
        rms_values: (N,) float32.
        IMPORTANT: Returns FULL-DIM embeddings always. PCA projection (if Q43 ships)
        happens in the Classifier, not here. This keeps the full-dim embedding
        available for Mahalanobis (Q34/Q43)."""
```

**Implementation notes:**
- Resample full audio to 16 kHz **once** (`librosa.resample`), then slice per onset — much cheaper than per-onset resampling.
- Context padding (Q21): extend the 200 ms extraction window backward and forward into the recording using real audio. Zero-pad only when within 800 ms of the recording boundary.
- No pre-emphasis filter. PANNs was not trained on pre-emphasized input.
- ONNX Runtime used in production; PyTorch used in eval/dev. The dev/runtime split must be documented in the build runbook so the export step is reproducible.
- The model's `model_id` is stored in session metadata; changing models invalidates the embedding cache.

**Dependencies:** onnxruntime (production), torch (dev/eval), librosa, numpy.

**Tests:**
- Verify context-padded window correctly extends into surrounding audio without exceeding buffer boundaries.
- Zero-padding triggered correctly for onsets within 800 ms of recording boundaries.
- Embedding shape is (N, 2048) for PANNs and (N, 768) for BEATs.
- RMS values computed correctly (5th/95th percentile mapping tested in Component 5 unit tests).
- Cache invalidation: `embedding_model_id` mismatch correctly triggers cache miss.

---

### Component 6: Classifier

**Purpose:** Map a full-dim embedding to a (class, calibrated score) pair using composite gate; manage user calibration; enforce self-test overfit guard.

**Public API:**

```python
from scipy.linalg import cholesky, solve_triangular
from sklearn.covariance import LedoitWolf

@dataclass
class CalibratedModel:
    taxonomy: TaxonomyConfig                       # Q66
    pca: Optional[np.ndarray]                      # (D_full, 64), optional per Q43
    lr: LogisticRegression
    T: float                                       # temperature scalar per Q75
    class_centroids_full_dim: np.ndarray           # (n_classes, D_full)
    pooled_cov_cholesky_full_dim: np.ndarray       # (D_full, D_full), lower triangular (Q68)
    distance_thresholds: np.ndarray                # (n_classes,) per-class 95th percentile on AVP
    softmax_threshold: float                       # default 0.45 (Q34)
    telemetry: TelemetrySink                       # local file in v1.0 per Q61

class Classifier:
    CLASSES: tuple[ClassId, ...] = ("kick", "snare", "closed_hat", "open_hat", "unknown")
    TRAINED_CLASSES: tuple[ClassId, ...] = ("kick", "snare", "closed_hat", "open_hat")

    def __init__(
        self,
        base_model_path: Path,
        taxonomy: TaxonomyConfig = TaxonomyConfig.default_v1_0(),  # Q66
        softmax_threshold: float = 0.45,
    ): ...

    def predict(self, embeddings: np.ndarray) -> list[tuple[ClassId, float]]:
        """Composite gate per Q34:
            LR_input = PCA @ emb if pca_present else emb
            probs = softmax(LR(LR_input) / T)          # T from Q75 disjoint fold
            top_class, top_prob = argmax(probs), max(probs)

            # Mahalanobis ALWAYS on full-dim, distance to NEAREST centroid (Q34)
            dists = [mahalanobis_sq_via_cholesky(emb, c, L_pooled)
                     for c in centroids_full_dim]
            nearest = argmin(dists)
            nearest_dist = sqrt(dists[nearest])

            if top_prob < softmax_threshold OR
               nearest_dist > distance_thresholds[nearest]:
                return (taxonomy.unknown_class_id, top_prob)
            return (taxonomy.classes[top_class], top_prob)
        """

    def fit_with_calibration(
        self,
        avp_embeddings: np.ndarray,             # full-dim
        avp_labels: np.ndarray,
        calibration_embeddings: np.ndarray,     # full-dim
        calibration_labels: np.ndarray,
        calibration_weight: float,              # default from Q42/Q65 sensitivity study
    ) -> None:
        """Refits:
          - LR head (with optional PCA projection on inputs):
              uses AVP + weighted calibration data. class_weight=None per Q26.
          - Temperature scalar (Q75):
              fit on disjoint held-out 20% slice of AVP, never on user calibration data.
          - Mahalanobis centroids (full-dim):
              uses AVP + weighted calibration data per Q52.
          - Mahalanobis pooled covariance Cholesky factor (full-dim):
              uses AVP only, unweighted, per Q52, Q68.
          - Mahalanobis distance thresholds (per-class):
              computed on AVP only, unweighted, per Q52.

        After fitting, runs the self-test overfit guard per Q71. If guard
        fails, restores previous calibration and raises CalibrationRejected
        with diagnostic information for the recovery dialog (Q81)."""

    def get_distribution_shift_threshold(self) -> float:
        """Returns the AVP-derived threshold per Q45."""
```

**Implementation of Mahalanobis distance via Cholesky factor (Q68):**

```python
def mahalanobis_sq_via_cholesky(x: np.ndarray, mu: np.ndarray, L_lower: np.ndarray) -> float:
    """Mahalanobis squared distance via Cholesky factor.
    Avoids storing or inverting the full covariance matrix.
    ~50% storage savings; better numerical stability than explicit inverse."""
    diff = x - mu
    y = solve_triangular(L_lower, diff, lower=True)
    return float(y @ y)

def fit_mahalanobis_full_dim(
    avp_embeddings: np.ndarray,            # (N_avp, D_full)
    avp_labels: np.ndarray,
    calibration_embeddings: np.ndarray,    # (N_cal, D_full)
    calibration_labels: np.ndarray,
    calibration_weight: float,
    classes: list[str],
):
    # Centroids: AVP + weighted calibration (Q52)
    centroids = []
    for c in classes:
        avp_in_c = avp_embeddings[avp_labels == c]
        cal_in_c = calibration_embeddings[calibration_labels == c]
        n_avp = len(avp_in_c); n_cal = len(cal_in_c)
        denom = n_avp + calibration_weight * n_cal
        centroid = (avp_in_c.sum(axis=0) +
                    calibration_weight * cal_in_c.sum(axis=0)) / denom
        centroids.append(centroid)
    centroids = np.stack(centroids)

    # Pooled covariance: AVP only, unweighted (Q52)
    centered = np.vstack([
        avp_embeddings[avp_labels == c] - centroids[i]
        for i, c in enumerate(classes)
    ])
    cov = LedoitWolf().fit(centered).covariance_

    # Cholesky factor (Q68) — replaces explicit inverse storage
    L = cholesky(cov, lower=True)

    # Per-class distance thresholds: AVP only, unweighted (Q52)
    thresholds = []
    for i, c in enumerate(classes):
        avp_in_c = avp_embeddings[avp_labels == c]
        dists_sq = np.array([
            mahalanobis_sq_via_cholesky(e, centroids[i], L) for e in avp_in_c
        ])
        thresholds.append(float(np.percentile(np.sqrt(dists_sq), 95)))
    return centroids, L, np.array(thresholds)
```

**Self-test overfit guard (Q71):**

```python
def self_test_overfit_guard(
    classifier_after_calibration: Classifier,
    classifier_baseline: Classifier,
    avp_embeddings: np.ndarray,
    avp_labels: np.ndarray,
    avp_subjects: np.ndarray,
    calibration_subjects: set[str],
) -> tuple[bool, dict]:
    """Returns (passed, diagnostics).
    Drops AVP subjects who contributed calibration data, then runs
    LOSO macro-F1 on remaining subjects with both classifiers.
    Returns passed=False if calibrated F1 has regressed by > 1 point."""
    held_out_mask = ~np.isin(avp_subjects, list(calibration_subjects))
    f1_calibrated = loso_macro_f1(
        classifier_after_calibration,
        avp_embeddings[held_out_mask],
        avp_labels[held_out_mask],
        avp_subjects[held_out_mask],
    )
    f1_baseline = loso_macro_f1(
        classifier_baseline,
        avp_embeddings[held_out_mask],
        avp_labels[held_out_mask],
        avp_subjects[held_out_mask],
    )
    passed = (f1_baseline - f1_calibrated) <= 0.01   # 1 point threshold
    return passed, {
        "f1_calibrated": f1_calibrated,
        "f1_baseline": f1_baseline,
        "delta": f1_calibrated - f1_baseline,
    }
```

If `passed` is False, the caller restores the previous calibration and raises `CalibrationRejected` carrying the diagnostics dict. The Editor catches this exception and surfaces the recovery dialog (Q81). The diagnostic file always records the macro-F1 delta.

**Distribution-shift warning (Q44, Q45):**

```python
def check_distribution_shift(
    self,
    event_scores: list[float],
    min_events: int = 100,
) -> bool:
    """Returns True if distribution shift warning should be surfaced.
    Fires if median score over first min_events events drops below
    the AVP-derived threshold stored in the model bundle (Q45)."""
    if len(event_scores) < min_events:
        return False
    return float(np.median(event_scores[:min_events])) < self.distribution_shift_threshold
```

**Dependencies:** scikit-learn, numpy, scipy.linalg.

**Tests:**
- Cholesky round-trip: covariance → Cholesky → reconstruct → max element-wise difference < 1e-10.
- `mahalanobis_sq_via_cholesky` agrees with the explicit-inverse computation on a synthetic dataset (within 1e-8).
- Distance-to-nearest-centroid behaves correctly: synthetic embedding equidistant from two centroids has nearest-distance equal to either.
- Full-dim Mahalanobis on PCA-projected synthetic OOD regression test: construct OOD samples that are zero in PCA-64-retained directions but non-zero in discarded directions; verify full-dim Mahalanobis flags them while PCA-64 Mahalanobis does not.
- Calibration fit at weight=50 with deliberately noisy calibration data: verify Mahalanobis distance thresholds are unaffected by calibration noise (AVP-only per Q52); verify centroids do shift with calibration data.
- Self-test guard fires on intentionally-bad calibration: 12 calibration samples drawn from a different subject distribution; verify guard rejects and previous calibration is restored.
- Self-test guard does NOT fire on good calibration: 12 calibration samples drawn from the same distribution as AVP; verify guard passes and new calibration is committed.
- Operating-point selection from §7.3 sweep produces a `(softmax_threshold, distance_percentile)` pair satisfying both Q50 bounds; gate fails (loudly) if no such pair exists.
- TaxonomyConfig round-trip: classifier with a non-default 5-class config (synthetic) trains, predicts, and persists correctly.
- Temperature scaling: synthetic over-confident logits → T > 1; under-confident → T < 1; perfectly calibrated → T ≈ 1.
- Distribution-shift warning fires on synthetic data with median score < threshold over 100 events, doesn't fire above.


---

### Component 7: Calibration manager

**Purpose:** Orchestrate calibration sample collection, refit, and commit/cancel lifecycle.

**Public API:**

```python
@dataclass
class CommitHandle:
    """Q28: cancellable handle to an in-flight calibration refit."""
    status: Literal["pending", "running", "done", "cancelled", "error"]
    error: Optional[Exception] = None

    def cancel(self) -> None: ...
    def wait(self, timeout: float = None) -> None: ...

class CalibrationRejected(Exception):
    """Raised by fit_with_calibration when the self-test overfit guard (Q71) fails."""
    f1_calibrated: float
    f1_baseline: float
    delta: float  # negative means regression

class CalibrationManager:
    MIN_SAMPLES_PER_CLASS: int = 3

    def add_sample(self, class_id: TrainedClassId, embedding: np.ndarray) -> CommitHandle:
        """Adds a calibration sample; triggers an async refit."""

    def commit(self) -> CommitHandle:
        """Commits the current calibration set to the session.
        Returns a CommitHandle. Newer commits cancel in-flight predecessors.
        Raises CalibrationRejected if the self-test overfit guard fails (Q71)."""

    def can_predict_live(self) -> bool:
        """Returns True only once at least 1 sample exists for every class. (Q5 fix)"""

    def predict_live(self, embedding: np.ndarray) -> tuple[ClassId, float]:
        """Live preview prediction on individual sample utterances.
        Refit time: ~150–250 ms (4 classes) + temperature refit. Worker thread."""

    def status(self) -> dict:
        """Returns counts per class; leave-one-out CV accuracy estimate (post-temperature)
        over the calibration set itself; per-class consistency (intra-class cosine similarity)."""

    def record_abandon_event(self) -> None:
        """Writes (class, count) pairs to the local diagnostic file (Q61) when the
        user closes calibration without committing. Aggregated for v1.1 planning."""
```

**Implementation notes:**
- Refit runs on a worker thread; subsequent refit requests are queued; multiple requests arriving while one is in-flight are coalesced.
- The worker checks the cancellation flag at safe points (after AVP load, before temperature refit, before save). On cancel, transitions to `cancelled` and exits.
- Calibration set persistence: persists in user profile, separate from any single project, so calibration carries across projects.
- Local-only telemetry sink (Q61): integration with self-test overfit guard ensures `CalibrationRejected` events are logged locally.

**Dependencies:** scikit-learn, numpy, threading.

**Tests:**
- `can_predict_live()` returns False until every class has ≥ 1 sample.
- Commit refused below `MIN_SAMPLES_PER_CLASS = 3` per class.
- `CommitHandle.cancel()` correctly interrupts an in-flight refit.
- Multiple adds while one refit is in-flight: coalesced correctly; only one refit runs for the batch.
- `record_abandon_event` writes to local diagnostic file when user exits without committing.
- CalibrationRejected raised and previous calibration restored on guard failure.

---

### Component 8: Tempo & grid engine

**Purpose:** Compute tempo grid positions, quantize events, and map RMS to velocity.

**Public API:**

```python
def beats_per_bar(ts: TimeSignature) -> int:
    """Returns numerator. 4 for 4/4, 3 for 3/4."""

def grid_positions(bpm: float, ts: TimeSignature, bars: int, grid: str) -> list[float]:
    """Returns list of times in seconds for each grid line.
    grid: one of '1/8', '1/16', '1/32', '1/16T', '1/8T'.
    Triplet grids: 1/8T = 1/12 of a bar in 4/4; 1/16T = 1/24."""

def quantize(onset_time: float, grid_positions: list[float], strength: float) -> float:
    """strength=0 → no change; strength=1 → snap to nearest grid line.
    Linear interpolation in between. Always returns a value clamped to
    the range of grid_positions."""

def compute_velocity_bounds(rms_values: np.ndarray) -> tuple[float, float]:
    """Returns (5th percentile, 95th percentile) of RMS values.
    For < 20 events: falls back to (10th, 90th) for 10–19 events,
    and to (min × 1.5, max × 0.67) for < 10 events."""

def velocity_from_rms(rms: float, rms_lo: float, rms_hi: float) -> int:
    """Logarithmic mapping per §5.3; clipped to 1..127."""
```

All pure. No state. No exceptions for normal inputs (return clamped values).

**Dependencies:** numpy (for percentile computation).

**Tests:**
- `beats_per_bar`: 4 for 4/4, 3 for 3/4.
- `grid_positions`: correct positions for all grid types at several BPMs and bar counts.
- `quantize`: at strength=0, returns onset unchanged; at strength=1, returns nearest grid position; at strength=0.5, returns midpoint.
- `compute_velocity_bounds`: correct percentiles for various array sizes including edge cases below 20 events.
- `velocity_from_rms`: logarithmic mapping, clipped to 1..127, handles edge cases.

---

### Component 9: MIDI exporter

**Purpose:** Export quantized events to a Standard MIDI File.

**Public API:**

```python
@dataclass
class MIDIExportConfig:
    grid: str = "1/16"
    drum_map: str = "GM"
    include_unknown_events: bool = False               # Q7
    unknown_event_gm_note: int = 56                   # GM Cowbell, opt-in only

@dataclass
class ExportSummary:
    events_exported: int
    events_skipped_unknown: int                        # Q7
    midi_path: Path

class MIDIExporter:
    def export(
        self,
        session: Session,
        config: MIDIExportConfig,
        path: Path,
    ) -> ExportSummary: ...
```

**Implementation notes:**
- Reads class set and MIDI mapping from `TaxonomyConfig` (Q66). Default mappings: kick→36, snare→38, closed_hat→42, open_hat→46.
- Velocity formula per §5.3 (logarithmic 5th/95th-percentile RMS mapping).
- Unknown events: excluded by default. Export dialog reports `events_skipped_unknown` as a count. When enabled, unknowns are exported as `unknown_event_gm_note` (default 56 / Cowbell). Dialog has a dropdown of GM percussion notes (35–81).
- `pretty_midi`'s tempo + time-sig handling is correct for FL Studio import; verified. Single drum track; events on channel 9 (MIDI channel 10, GM percussion). Note duration: fixed 50 ms.
- Velocity comes from the Event (already computed by the velocity-from-RMS function).

**Dependencies:** pretty_midi, numpy.

**Tests:**
- Generated `.mid` re-parsed by `mido`; verify every event matches expected (time within MIDI tick resolution, pitch, velocity).
- Tests exercise 4 drum-map entries (kick, snare, closed_hat, open_hat) plus opt-in unknown-class export to a configurable GM note.
- Default export with unknowns present: verify they're skipped, summary reports correct count.
- Opt-in export: verify unknowns export to selected GM note.
- Round-trip with §7.5: unknown-class event survives session save/load and exports correctly.
- Reads class set from `TaxonomyConfig` per Q66.

---

### Component 10: Playback engine

**Purpose:** Synchronized multi-stem playback of recorded audio and drum samples.

**Unchanged structure from v0.5.** The drum-sample bundle now has 4 stems (kick, snare, closed_hat, open_hat) instead of 3. A single audio callback handles all stems simultaneously to prevent inter-stem flam.

**Dependencies:** sounddevice, numpy.

---

### Component 11: Editor UI

**Purpose:** Piano-roll editor synchronized with waveform display; recording-session UX; taxonomy disclosure; migration banners.

**Key behaviors (all explicit, no references to earlier versions):**

**Piano roll:**
- Custom-painted via QPainter. Five horizontal lanes: kick, snare, closed_hat, open_hat, unknown (lane count from `TaxonomyConfig` per Q66; default = 5).
- Unknown lane visually distinct: neutral grey color (vs. the four trained classes' accent colors), dashed lane separator above and below, and a small "?" glyph in the lane label.
- Each event is a draggable rectangle; horizontal position = quantized_time; height fills the lane.
- Waveform displayed beneath the lanes, time-axis-aligned. Click anywhere to seek the playback engine.
- Snap toggle: when on, dragging snaps to current grid; when off, free-drag.
- Undo/redo: command-pattern stack capped at 100 entries.
- Confidence-based visual hint: events with score < 0.7 are rendered with a yellow border.
- Event tooltips: "Confidence: 87%" for trained-class events; "Best guess: snare (32%) — below confidence threshold" for unknown events.

**Reclassify-on-drag:**
- When an event is dragged across a lane, the editor emits `eventReclassified(int, Event)` with the new class.
- The calibration manager can listen and prompt the user to add as a calibration sample. Opt-in ("Add this as a calibration example for kick? [Yes] [No]"), not automatic.

**Taxonomy disclosure (Q31):**
- Calibration screen header (always visible): "VoxKit recognizes four sounds: kick, snare, closed hi-hat, and open hi-hat. Anything that doesn't clearly match one of those will be tagged 'unknown' so you can decide what to do with it."
- Editor first-open (dismissable banner): "VoxKit detected 5 lanes: 4 trained sounds plus 'unknown' for events that didn't clearly match. Drag from unknown into one of the four trained lanes to reclassify."
- Reclassify-on-drag tooltip (on first drag of each session): "Drag corrections train VoxKit on your sounds when you re-calibrate."

**Bleed-resetup banner (Q29):** for sessions loaded with `bleed_ir_origin == "migrated_pending_recapture"`, a non-modal banner appears at the top of the editor: "This session uses an estimated bleed setting from an older VoxKit. [Re-run setup] [Dismiss]." Dismissal is per-session; the banner reappears on next open until re-setup is done.

**Mid-session bleed prompts (from §5.2.1):** Two variants:
- (a) When silent windows were available and new IR failed the bleed-gate: "Bleed has changed (likely headphone shift). [Re-setup] [Continue with current]."
- (b) When trigger fired but no silent windows were available: "Bleed has drifted but VoxKit needs a quiet moment to recheck. [Take 4 seconds of silence] [Skip and keep current bleed setting]."

**Distribution-shift warning toast (Q44):** "Confidence is consistently low on your events. Re-calibrating with more samples may help. [Open calibration] [Dismiss]." Fires once per session if the warning condition is met.

**Device-disconnect modal (Q37):** blocking modal — "Audio device disconnected. Recording paused at bar X. Reconnect the device or select a different one." [Reconnect/Select] [Save and exit].

**Post-sleep toast (Q38):** non-blocking — "Device validated after sleep. Bleed setup may have changed (volume, headphone position). [Re-run bleed setup] [Continue]."

**First-run guided tour (Q54):** trigger fires on the **first unknown event** in the first session that contains any unknowns. Once per user (not per-session), dismissable, never repeats after dismissal. Animation, caption, implementation via Qt `QPropertyAnimation`.

**PCA-Mahalanobis recalibration banner:** persistent banner on session load if `mahalanobis_full_dim is None` after a v0.8 → v0.9 PCA-session migration: "VoxKit improved out-of-distribution detection in this version. Re-run a quick calibration to enable it. Until then, unknown detection uses the previous (less accurate) method." Banner is **not dismissable** until calibration runs. If the user ignores it, the OOD detector silently falls back to softmax-only gating (logged in diagnostics) but the banner remains. No "Remind me later" link.

**Recording-session UX (Q73):**
- During recording: live waveform; click-bleed quality indicator (numeric dB value + colored bar per Q79: green ≥ 20 dB, yellow 10–20 dB, red < 10 dB); recording duration; bar count if click track is set. No live classification.
- On stop: modal progress dialog with three named phases — "Detecting onsets" → "Extracting embeddings" → "Classifying events" — each with a percentage. Total wall-clock target ≤ 0.5× audio duration on reference hardware (Q72). For a 16-second session, ≤ 8 seconds.
- Cancel button preserves audio and returns to recording mode.
- After completion: user lands in editor with all events placed and labeled.

**Inference pipeline threading (Q76):** the recording-session progress dialog is backed by a dedicated `InferenceWorker` thread. The main thread runs only the Qt event loop. Phase progress reported via Qt signals; the cancel button sets a `threading.Event` flag checked between phases and at the top of each per-onset iteration inside embedding extraction. Worst-case cancel latency is one embedding-extraction call (~50 ms reference). On cancel, the recorded audio buffer is preserved in the Session; partial events are discarded.

**CalibrationRejected dialog (Q81):** "VoxKit's accuracy check found that the most recent calibration didn't improve classification on the held-out test set; the previous calibration has been restored. This usually means the calibration recording was very different from your typical use (very few samples, unusual background noise, or a different microphone), and the model would have generalized worse with it. You can try again with more or quieter samples, or continue using the previous calibration."

**Dependencies:** PyQt6 or PySide6 (implementer's choice), numpy.

**Tests:**
- 5 lanes render correctly; unknown lane visually distinct.
- Drag-from-unknown into trained lane works and updates `class_id` correctly.
- Updated disclosure text appears on first open of calibration and editor.
- New mid-session bleed prompt variants (both (a) and (b)) appear under the right conditions; buttons function.
- Distribution-shift warning toast fires when the rolling-median condition is met; doesn't fire otherwise; once per session only.
- Device-disconnect modal appears on synthetic disconnect; both buttons function.
- Post-sleep toast appears on synthetic wake; both buttons function.
- Unknown-event tooltip shows the "best guess" interpretation.
- First-run guided tour fires on first unknown event in first session containing unknowns; does not fire on second session.
- PCA-Mahalanobis recalibration banner appears on session load when `mahalanobis_full_dim is None`; is not dismissable until calibration runs.
- Recording-session progress dialog shows three phases; cancel button correctly signals InferenceWorker and preserves audio buffer.
- CalibrationRejected dialog shown with correct text per Q81 when self-test guard rejects calibration.
- Lane count driven by `TaxonomyConfig` per Q66.


---

### Component 12: Eval harness (dev-only, smoke + full tiers)

**Purpose:** Run the tiered evaluation pipeline against the three dataset tiers (Q63); enforce release gates (Q50, Q53, Q70, Q72); produce reproducible structured output.

**Public API (CLI):**

```bash
# Substrate + PCA bake-off (Phase 1 week 2 — Q33, Q43; Q57 closed per Q69)
python -m voxkit.eval.substrate_and_pca_bakeoff \
    --avp-root /path/to/avp \
    --pilot-ood-root /path/to/pilot_ood \
    --pilot-subjects 5 \
    --tiebreaker bootstrap-ci \          # Q74 — replaces fixed margin
    --bootstrap-resamples 1000 \
    --output results/substrate_pca_bakeoff.json
    # PCA-64 per-class Mahalanobis sweep removed (Q69)

# Threshold + Mahalanobis sweep (full-dim Mahalanobis, Cholesky-backed)
python -m voxkit.eval.threshold_and_mahalanobis_sweep \
    --avp-root /path/to/avp \
    --ood-root /path/to/ood \
    --emit-operating-point     # selects (softmax_threshold, distance_percentile)
                               # pair satisfying Q50 bounds; fails loud if none exists
    --output results/threshold_sweep.json

# Calibration weighting sweep with effective-influence + noise-sensitivity study
python -m voxkit.eval.calibration_uplift \
    --avp-root /path/to/avp \
    --weights 1,5,25,50,125,625 \                   # Q42
    --sample-counts 3,5,10,20 \
    --measure-effective-influence \                  # Q42
    --noise-sensitivity-sigmas 0.1,0.5,1.0 \        # Q65
    --self-test-overfit-guard \                      # Q71
    --output results/calibration_uplift.json

# Onset F-measure + alignment-MAE release-gate evaluator (two-tier per Q70)
python -m voxkit.eval.onset_release_gate \
    --avp-root /path/to/avp \
    --ood-root /path/to/ood \
    --avp-f-threshold 0.92 \
    --ood-f-threshold 0.88 \
    --iou-ms 50 \
    --avp-mae-threshold-ms 15 \                     # Q70
    --ood-mae-threshold-ms 25 \                     # Q70
    --output results/onset_release_gate.json
    # Returns nonzero exit code if any threshold is missed.

# CPU performance benchmark (Q72)
python -m voxkit.eval.cpu_perf \
    --substrate panns|beats \
    --session-bars 32 --session-bpm 120 \
    --reference-target-multiple 0.5 \               # ≤ 0.5× wall-clock
    --output results/cpu_perf.json

# Smoke-tier eval (3 fixed subjects, every PR)
python -m voxkit.eval.smoke \
    --avp-root /path/to/avp \
    --ood-root /path/to/ood \
    --subjects-json voxkit/eval/smoke_subjects.json \
    --fail-on-regression-points 5 \                 # vs main-branch baseline
    --fail-on-onset-f-below 0.80 \
    --tier smoke \                                   # Q85: tier field in output banner
    --output results/smoke.json

# Full-tier eval (all 28 AVP subjects, all 15 OOD subjects, nightly)
python -m voxkit.eval.full \
    --avp-root /path/to/avp \
    --ood-root /path/to/ood \
    --fail-on-regression-points 2 \
    --tier full \
    --output results/full.json
```

**Output JSON schema:**

```json
{
    "schema_version": "0.11",
    "tier": "smoke | pr_validation | full",
    "substrate": "panns_cnn14_16k_v1 | beats_v1",
    "macro_f1_avp": 0.0,
    "macro_f1_ood": 0.0,
    "per_class_f1": {"kick": 0.0, "snare": 0.0, "closed_hat": 0.0, "open_hat": 0.0},
    "confusion_matrix_5x5": [],
    "onset_f_avp": 0.0,
    "onset_f_ood": 0.0,
    "onset_mae_avp_ms": 0.0,
    "onset_mae_ood_ms": 0.0,
    "missed_unknown_rate_ood": 0.0,
    "false_unknown_rate_avp": 0.0,
    "operating_point": {"softmax_threshold": 0.45, "distance_percentile": 95},
    "ood_avp_gap_ci_95": [0.0, 0.0],
    "cpu_perf_wall_s": 0.0,
    "calibration_uplift_median_f1": 0.0,
    "all_gates_passed": true
}
```

**Tiered dataset support (Q63):**
- **CI tier:** synthetic dataset, in repo. ~2 minutes. Validates pipeline runs, imports, Cholesky round-trip, MIDI parseability, and eval-harness JSON shape per Q85 — does NOT validate model quality. Output banner explicitly names the tier: "SYNTHETIC TIER — validates pipeline structure only, NOT model quality."
- **PR-validation tier:** minimum-reproducible (10–20 subjects per fold). Project-hosted. ~30 minutes.
- **Release tier:** canonical (full AVP + 15-subject OOD). Release candidates only.

**Synthetic dataset role caveat (Q85):** README and eval-harness `--help` both repeat: "The synthetic dataset tier validates that the pipeline runs end-to-end, imports work, Cholesky factors are correct, MIDI is parseable, and JSON output conforms to the schema. It does NOT validate model accuracy, calibration quality, or OOD performance. These require the minimum-reproducible or canonical dataset tiers."

**Dependencies:** scikit-learn, numpy, librosa, onnxruntime (or torch), pretty_midi, scipy.

**Tests:**
- `substrate_and_pca_bakeoff` end-to-end on a 3-subject dev fixture for both substrates emits a valid decision summary with correct bootstrap CI computation.
- `threshold_and_mahalanobis_sweep` emits operating point satisfying Q50 bounds on a synthetic fold; fails loudly if no such pair exists.
- `calibration_uplift` produces monotonic effective-influence vs weight curve; noise-sensitivity study produces drift vs weight at three noise levels.
- `onset_release_gate` returns nonzero exit code if any threshold is missed; returns zero if all pass.
- `cpu_perf` correctly times end-to-end inference on the standard session and compares to the 0.5× target.
- Smoke-tier eval on the canonical 3 smoke subjects emits correct tier field "smoke" in output banner.
- Full-tier eval produces bootstrap CIs over all 28 AVP LOSO folds.

---

## End of document

**Document graph for the build:**
- §2 (Q1–Q85) → all 85 decisions stated with full resolutions.
- §11 Components 1–12 → 12 buildable units, all APIs explicit with no references to earlier versions.
- §7 test strategy → Q50 release-gate criterion is the rewritten core; §7.11 CI checks enforce the gate per every PR.
- §8 multi-perspective review → nine reviewers; all major decisions traced to their originating review.
- §9 / Q62 → Architectural portability in Phase 1; Linux ship in Phase 1.5 as bandwidth permits.
- §9 / Q63 → Three-tier dataset access plan.
- §10 item 17 → Bluetooth support, Phase 2 prototype.

**Critical-path dependencies for Phase 1:**
- Project license file in repo + SPDX headers from week 1.
- Component 2 (Recorder) — Bluetooth filter + disconnect + sleep + sample-rate handling + WASAPI default + resampler worker thread + MMCSS registration. Must land before any OOD recordings.
- License-review memos for PANNs, BEATs, signalsmith-stretch, Rubber Band — week 1.
- Q36 OOD power memo — week 1, gates full OOD recruitment.
- Dataset access infrastructure — week 1.
- `import-linter` and `reuse` CI jobs from week 1; both block merge.
- `manifest.json` version field and migration table from week 1.
- Click-bleed tracer bullet from week 1; escalation path defined if simplest IR estimation fails the > 20 dB acceptance threshold.
- `InferenceWorker` scaffolding from week 1, wired to UI in week 2.
- 5-subject pilot OOD recording — week 1, gates Q33 if AVP CIs overlap.
- Q33 substrate decision — week 2.
- CPU performance benchmark on each substrate — week 2 (Q72 input to Q33).
- Q42/Q65 calibration-weight decision — week 2.
- Q43 PCA-64 (LR-only) — week 2.
- Component 6 (Classifier) — Cholesky-backed full-dim Mahalanobis + self-test guard + temperature held-out + TaxonomyConfig.
- Component 11 (Editor) — 5-lane piano roll + first-unknown tour + persistent migration banner + recording-session progress UX.
- Component 12 (Eval harness) — operating-point selection, alignment-MAE tier, CPU perf benchmark from week 2.
- Migration matrix test (v0.4–v0.10 → v0.11) — every PR from week 1.

End.