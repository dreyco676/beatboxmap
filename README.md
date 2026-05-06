# VoxKit

VoxKit turns vocal-percussion recordings into MIDI drum tracks — offline, on your own machine, no cloud required.

You beatbox (or tap, or clap) into a microphone. VoxKit detects each hit, classifies it as a kick, snare, closed hi-hat, or open hi-hat, and writes a MIDI file that drops straight into any DAW on GM drum channel 10.

---

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Calibration](#calibration)
- [MIDI export](#midi-export)
- [Evaluation and datasets](#evaluation-and-datasets)
- [Development setup](#development-setup)
- [Packaging](#packaging)
- [Architecture](#architecture)
- [License](#license)

---

## How it works

```
microphone audio
      │
      ▼
 OnsetDetector          energy-flux onset detection, 5 ms resolution
      │                 click-guard suppression, 6 dB noise gate
      ▼
 EmbeddingExtractor     ONNX model (PANNs CNN14 or BEATs)
      │                 extracts a fixed-length embedding at each onset
      ▼
 LR head                logistic regression on PCA-reduced embeddings
 + Mahalanobis gate     rejects out-of-distribution events as "unknown"
      │
      ▼
 MIDI export            Type 1 MIDI, GM drum channel 10
                        kick=36  snare=38  closed_hat=42  open_hat=46
```

The classifier is **personalized** — it is fit to your own voice during a short calibration session before use. The pre-trained ONNX model is a general audio encoder; the logistic regression head and Mahalanobis OOD gate are fit fresh for each user.

---

## Requirements

- Python 3.11 or 3.12
- A microphone (for live recording)
- ONNX model files in `models/` (see [Architecture](#architecture))

---

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/anthropics/voxkit
cd voxkit
pip install -e ".[ui,audio-linux]"   # Linux
pip install -e ".[ui,audio-windows]" # Windows
```

### Core library only (no GUI, no audio capture)

```bash
pip install voxkit
```

### Optional extras

| Extra | Installs | Use when |
|---|---|---|
| `ui` | PySide6 | Running the desktop GUI |
| `audio-linux` | sounddevice | Live microphone recording on Linux |
| `audio-windows` | sounddevice | Live microphone recording on Windows |
| `dev` | pytest, import-linter, reuse, pre-commit, build | Contributing to VoxKit |
| `freeze` | PyInstaller | Building a standalone executable |

---

## Quick start

### Desktop GUI

```bash
voxkit
```

On first launch the calibration wizard opens automatically. Record a few examples of each drum sound, then start recording a performance.

```bash
voxkit --smoke-test   # headless sanity check (CI-safe)
voxkit --version
```

### Pipeline from Python

```python
import numpy as np
from voxkit.ui.inference_pipeline import run_pipeline
from voxkit.ui.model import Model

model = Model.load("path/to/saved_model.vkm")
audio = ...  # mono float32 at 16 kHz

result = run_pipeline(audio=audio, model=model)
for event in result.events:
    print(f"{event.t:.3f}s  {event.class_id}  score={event.score:.3f}")
```

Pass `cancel_flag` (a `threading.Event`) to interrupt long recordings mid-run. Pass `detect_onsets` to override the default `OnsetDetector` with your own callable.

---

## Calibration

Calibration records a few examples of each drum sound and fits a personalized classifier. This takes about two minutes.

### GUI flow

The calibration wizard walks you through:

1. Record 3–5 examples of each sound (kick, snare, closed hi-hat, open hi-hat).
2. Optionally preview the live classifier against new examples before committing.
3. Commit — the classifier is fit and saved.

### Python API

```python
from voxkit.ui.calibration_flow import CalibrationFlow
from voxkit.classifier.calibration_manager import CalibrationManager
from voxkit.classifier.embeddings import EmbeddingExtractor
from voxkit.classifier.classifier import Classifier
from voxkit.core.taxonomy import TaxonomyConfig

taxonomy = TaxonomyConfig.default_v1_0()
extractor = EmbeddingExtractor("models/panns_cnn14_16k.onnx", substrate_id="panns_cnn14")
classifier = Classifier.untrained(taxonomy, extractor.embedding_dim)
manager = CalibrationManager(classifier=classifier, taxonomy=taxonomy)

flow = CalibrationFlow(extractor=extractor, manager=manager, classifier=classifier)

for class_id in taxonomy.classes:
    audio = record_snippet()           # mono float32 at 16 kHz
    flow.add_sample(class_id, audio)

if flow.can_preview():
    class_id, score = flow.preview(new_audio)

handle = flow.commit()
```

Calibration samples are mixed into the LR head fit at an elevated weight alongside the AVP training data, so even a handful of personal examples meaningfully shifts the decision boundaries toward your voice.

### Distribution-shift warning

After processing 100 events, VoxKit computes the median softmax confidence score. If it falls below 70% of the expected level from calibration, the UI shows a toast notification suggesting recalibration. This catches gradual drift (fatigue, microphone position changes, different room acoustics).

---

## MIDI export

```python
from pathlib import Path
from voxkit.export.midi import export_midi
from voxkit.core.taxonomy import TaxonomyConfig

taxonomy = TaxonomyConfig.default_v1_0()

export_midi(
    events=result.events,
    out_path=Path("performance.mid"),
    bpm=120.0,
    taxonomy=taxonomy,
    include_unknowns=False,        # drop OOD events (default)
    ticks_per_beat=480,            # DAW-standard PPQ
    time_signature=(4, 4),        # optional meta event
)
```

GM note mapping: kick=36, snare=38, closed hi-hat=42, open hi-hat=46. Events classified as "unknown" (Mahalanobis OOD gate) are silently dropped unless `include_unknowns=True`.

Round-trip import:

```python
from voxkit.export.midi import import_midi
events = import_midi(Path("performance.mid"), taxonomy)
```

---

## Evaluation and datasets

### Dataset tiers

VoxKit uses a three-tier eval system that lets contributors run tests at whatever level of rigor their machine supports.

| Tier | Dataset needed | What it measures |
|---|---|---|
| `synthetic` | none | pipeline smoke-test, no quality assertions |
| `minimum-reproducible` | AVP Personal (free download) | F-measure and MAE on real onset-detection corpus |
| `canonical` | AVP Personal + release gate | F-measure, MAE, pass/fail against release thresholds |

```python
from voxkit.eval.harness import run_for_tier

result = run_for_tier("synthetic")           # always passes
result = run_for_tier("minimum-reproducible") # needs AVP dataset
result = run_for_tier("canonical")            # needs AVP dataset
```

### AVP dataset

Download the AVP (A Versatile Percussion) dataset from Zenodo (record IDs 5036529 / 5578744) and unzip it to:

```
data/avp/AVP_Dataset/
```

The dataset is third-party, licensed CC-BY-4.0, and is not committed to this repository. See `CONTRIBUTING.md §Datasets` for details.

### LOSO evaluation

Leave-One-Subject-Out evaluation measures how well the embedding substrates (PANNs and BEATs) discriminate the four drum classes across unseen speakers.

```bash
# PANNs CNN14 (fastest)
python scripts/run_avp_loso.py --substrate panns

# BEATs
python scripts/run_avp_loso.py --substrate beats

# Both substrates → bake-off decision
python scripts/run_avp_loso.py --substrate both

# Re-use cached embeddings (skip extraction)
python scripts/run_avp_loso.py --substrate panns --use-cache
```

Results are written to `data/avp_loso_<substrate>.json` with per-fold F1, mean, standard deviation, and 95% bootstrap confidence interval.

---

## Development setup

```bash
git clone https://github.com/anthropics/voxkit
cd voxkit
pip install -e ".[dev,ui,audio-linux]"
pre-commit install
```

### Running tests

```bash
pytest                        # full suite
pytest -m "not slow"          # skip long-running tests
pytest -k test_06             # single module
pytest --cov=voxkit           # with coverage
```

### REUSE compliance

```bash
reuse lint
```

### Import graph linting

```bash
lint-imports
```

The import graph is declared in `.importlinter`. The main boundary is that `voxkit.eval` (dev/eval tooling) must never be imported by the shipping UI or classifier code.

---

## Packaging

### Standalone executable (PyInstaller)

The `voxkit.spec` file builds a one-directory bundle that runs without a Python installation:

```bash
pip install -e ".[ui,audio-linux,freeze]"
pyinstaller voxkit.spec
# Output: dist/voxkit/voxkit  (Linux)
#         dist/voxkit/voxkit.exe  (Windows)
```

The bundle includes PySide6, onnxruntime, scikit-learn, scipy, and the `models/` directory. The `voxkit.eval` module is excluded.

### CI release workflow

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which builds:

- A platform-independent Python wheel and sdist
- A PyInstaller one-directory bundle for Windows x64 (`voxkit-windows-x64.zip`)
- A PyInstaller one-directory bundle for Linux x86_64 (`voxkit-linux-x86_64.tar.gz`)

All three artifacts are attached to the corresponding GitHub release.

---

## Architecture

### Component overview

```
voxkit/
├── audio/          Microphone recorder, resampler, audio budget / drop policy
├── classifier/     EmbeddingExtractor (ONNX), LR head, Mahalanobis OOD gate,
│                   calibration session, CalibrationManager, CalibrationFlow
├── core/           Session, Event, TaxonomyConfig, manifest, migrations
├── dsp/            OnsetDetector, onset evaluation, bleed metrics
├── eval/           Eval harness (3 tiers), LOSO, substrate bake-off,
│                   onset release gate, calibration uplift, CPU perf, migration check
├── export/         MIDI export / import
├── playback/       Playback engine
├── telemetry/      Local event sink
├── tempo/          Tempo grid
└── ui/             PySide6 app, MainWindow, editor, calibration wizard,
                    inference pipeline, inference worker, banners, dialogs
```

### Embedding substrates

Two ONNX models are supported. Place them in `models/`:

| Model | File | Embedding dim | Notes |
|---|---|---|---|
| PANNs CNN14 | `panns_cnn14_16k.onnx` | 2048 | Faster, recommended default |
| BEATs iter3+ | `beats_iter3plus_as2m.onnx` | 768 | Higher accuracy in some conditions |

The substrate bake-off script (`run_avp_loso.py --substrate both`) runs both and selects the winner by Wilcoxon signed-rank test with bootstrap tiebreaker.

### Onset detection

`OnsetDetector` uses non-overlapping 5 ms energy frames (hop = 80 samples at 16 kHz). The onset detection function is the positive first-difference of frame energies. A noise gate (6 dB above the noise floor estimated from the first 200 ms) drops faint transients. A click-guard removes onsets within ±15 ms of metronome clicks.

### Classifier

After embedding extraction, embeddings are projected to 256 PCA dimensions and passed to a logistic regression head fit with group-stratified cross-validation (one fold per recording participant). Calibration samples from the current user are mixed into the final LR fit at elevated weight without leaking into the cross-validation loop. A Mahalanobis distance gate operating on full-dimension embeddings rejects events that fall outside the training distribution.

### Session and MIDI

Each recording session produces a list of `Event(t, class_id, score)` objects. `export_midi` writes a Type 1 MIDI file on GM drum channel 10 at 480 PPQ. Unknown-class events are silently dropped unless `include_unknowns=True`.

---

## License

Copyright (C) VoxKit contributors.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See [LICENSES/GPL-3.0-or-later.txt](LICENSES/GPL-3.0-or-later.txt) for the full text.

The AVP dataset (`data/avp/`) is third-party and licensed under [CC-BY-4.0](LICENSES/CC-BY-4.0.txt): Blas Ishtar, George Tzanetakis, and AVP Dataset contributors.
