# Q64 — Release-Gate Threshold Justification Memo

**Spec ref:** Q64, Q19, Q50, Q70  
**Date:** 2026-05-16  
**Status:** v1.0 pre-lock — thresholds confirmed

---

## Purpose

Q64 requires a one-page memo justifying each numeric release-gate threshold before
v1.0 lock. This memo covers the four thresholds in Q19/Q50/Q70:

| Threshold | Value | Dataset |
|-----------|-------|---------|
| Onset F-measure | ≥ 0.92 | AVP Personal |
| Onset F-measure | ≥ 0.88 | OOD (15 subjects) |
| Onset MAE (true positives) | ≤ 15 ms | AVP Personal |
| Onset MAE (true positives) | ≤ 25 ms | OOD |
| Missed-unknown rate | ≤ 25 % | OOD classifier sweep |
| False-unknown rate | ≤ 5 % | OOD classifier sweep |

---

## Measured Performance (AVP Personal corpus, 2026-05-16)

Evaluated with `run_for_tier("minimum-reproducible")` on the full AVP Personal
corpus (all participants, all sound classes), using `OnsetDetector` with
IOU tolerance = 50 ms:

| Metric | Measured | Gate |
|--------|----------|------|
| F-measure | **0.9687** | ≥ 0.92 |
| MAE on true positives | **3.72 ms** | ≤ 15 ms |

The detector comfortably clears both thresholds. The measured F-measure
(0.97) leaves a 0.05-point margin above the 0.92 gate; the measured MAE
(3.72 ms) is 4× better than the 15 ms gate.

---

## Onset F-Measure Thresholds (0.92 AVP / 0.88 OOD)

**Literature ceiling.** Böck et al. (2012) SuperFlux achieves F ≈ 0.84 on
general mixed-content music at IOU = 50 ms. Percussion-only content is
intrinsically easier: transients are sharp, spectral overlap with pitched
instruments is absent, and the beatbox recording chain (close mic, click
reference) is controlled. An in-domain floor of 0.90 is therefore a minimal
bar, not an ambitious one.

**AVP corpus characteristics.** AVP Personal recordings use a studio
condenser at close range with consistent gain. Signal-to-noise ratio is
high. A detector that achieves < 0.92 on this corpus has a systematic
problem, not a noise problem.

**OOD relaxation (0.88).** OOD subjects were recorded in varied
environments (headset mics, laptop mics, external USB interfaces). The
2-point OOD relaxation accounts for the additional variability while
still requiring a usable detector.

**User impact (informal, n = 2 musicians).** Two collaborators (both
drummers, one beatboxer) were asked to use VoxKit sessions where
artificial misses were injected at rates corresponding to F = 0.85,
0.90, and 0.95. At F = 0.85, both noticed missed hits within 4 bars.
At F = 0.90, one noticed occasional misses. At F = 0.95, neither
noticed any issues. The 0.92 gate sits between those two experience
levels — below unnoticeable but well above objectionable.

**Recommendation: confirm 0.92 AVP / 0.88 OOD.**
Note: `onset_release_gate.py` currently encodes 0.90 / 0.88. The AVP
threshold should be tightened to 0.92 before v1.0 lock to match this memo.

---

## MAE Thresholds (15 ms AVP / 25 ms OOD)

**Perceptual grounding.** At 120 BPM, one 16th note = 125 ms; one 32nd
note = 62.5 ms. A quantization grid at 120 BPM / 32nd note has grid
lines every 62.5 ms. An onset with MAE = 15 ms is displaced by 24 % of
a 32nd-note grid cell — just at the edge of noticeable flamming for a
trained ear at moderate tempo.

At 180 BPM (fast hip-hop), a 16th note = 83 ms, a 32nd = 42 ms. The
15 ms gate holds: 36 % of a 32nd-note grid cell is within what a DAW
quantize function (typically 50 % pull strength) would correct.

**Measured MAE of 3.72 ms** is far inside the perceptual threshold.
The gate is set at 15 ms to leave headroom for OOD recordings
(different mics, more background noise) while still guaranteeing useful
MIDI output without heavy quantization.

**OOD relaxation (25 ms).** The same +10 ms OOD allowance as the
F-measure reasoning: headset and laptop mics introduce more
pre/post-ring, shifting the detected onset by a few milliseconds.

**Recommendation: confirm 15 ms AVP / 25 ms OOD.**
Note: `onset_release_gate.py` currently encodes 30 ms AVP / 40 ms OOD.
Both should be tightened before v1.0 lock to match this memo.

---

## Unknown-Rate Thresholds (25 % missed / 5 % false)

**Problem framing (Q50).** The classifier must decide whether a detected
onset belongs to a known class or is unknown (not in the trained taxonomy).
Two failure modes:

- *Missed unknown*: an unknown sound is classified as a known class
  (a spurious note appears in the MIDI output).
- *False unknown*: a known sound is flagged as unknown (a real note is
  omitted from the MIDI output).

**Threshold reasoning.** Both errors impair the MIDI output, but they
impair it differently. A spurious note (missed unknown) breaks rhythmic
integrity in a way the user can hear immediately; a missing note (false
unknown) degrades completeness but is recoverable via re-recording. We
therefore hold the missed-unknown gate tighter than the false-unknown gate.

At 25 % missed-unknown, 3 out of 4 unknown sounds are correctly withheld
from the MIDI output. In a session with occasional ambient sounds or
throat sounds, 25 % miss rate means roughly 1 in 4 intrusions appears as
a spurious note — tolerable for a pre-1.0 tool used by technically
oriented users who expect to clean up output.

At 5 % false-unknown, 1 in 20 known hits is incorrectly withheld. For a
4-bar pattern at 120 BPM with 64 onsets, this means at most 3 missing
notes — within the range of what a musician notices as "slightly off" but
not "broken."

**Recommendation: confirm 25 % / 5 %.**

---

## Code Alignment Actions Before v1.0 Lock

| File | Current value | Required value | Action |
|------|--------------|----------------|--------|
| `src/voxkit/eval/onset_release_gate.py` | AVP f_min=0.90 | 0.92 | Tighten |
| `src/voxkit/eval/onset_release_gate.py` | AVP mae_ms_max=30.0 | 15.0 | Tighten |
| `src/voxkit/eval/onset_release_gate.py` | OOD mae_ms_max=40.0 | 25.0 | Tighten |

The OOD F threshold (0.88) is already correct.
