# VoxKit

VoxKit turns vocal-percussion into MIDI drum tracks — offline, on your own machine, no cloud required.

Beatbox, tap, or clap into a microphone. VoxKit finds every hit, figures out whether it's a kick, snare, closed hi-hat, or open hi-hat, and writes a MIDI file you can drop straight into any DAW.

---

## How it works

Recording goes through three stages:

**1. Onset detection** — VoxKit listens for transients in the audio. A small neural network (trained on the AVP Personal beatboxing corpus and running locally via ONNX) finds each hit with sub-millisecond timing precision. A noise gate suppresses background hiss and a click-guard stops the metronome from showing up as drum hits.

**2. Classification** — Each detected hit is passed through a neural audio encoder (BEATs iter3+, running locally via ONNX) that converts the sound into a compact representation. A lightweight classifier — trained on your own voice during calibration — maps that representation to one of the four drum classes. Hits that don't match any of your calibration samples are flagged as "unknown" rather than forced into a wrong category.

**3. MIDI export** — Classified hits are written to a standard MIDI file on GM drum channel 10 (kick=36, snare=38, closed hi-hat=42, open hi-hat=46) at 480 PPQ, ready for FL Studio, Ableton, Logic, or any other DAW.

Everything runs locally. No audio leaves your machine.

---

## Calibration

VoxKit's classifier is personalized — it learns *your* sounds, not a generic idea of what a kick or snare should sound like. Before your first recording session you run a short calibration:

1. The app asks you to record 3–5 examples of each sound.
2. Once you've recorded all four, you can optionally hit a **Preview** button and make a few sounds to check the classifier is reading you correctly.
3. Hit **Commit** and you're done. The whole thing takes about two minutes.

Your calibration is saved between sessions. If your voice or setup changes — different mic position, time of day, being sick — you can recalibrate in the same two minutes.

**Distribution shift warning:** After every 100 events in a session, VoxKit checks whether its confidence scores are still in the expected range. If they've dropped significantly it shows a notification suggesting a quick recalibration. This catches gradual drift you might not notice yourself.

---

## Installation

Download the latest release for your platform from the [Releases page](https://github.com/anthropics/voxkit/releases) and unzip it. No Python installation required.

- **Windows:** `voxkit-windows-x64.zip` → run `voxkit.exe`
- **Linux:** `voxkit-linux-x86_64.tar.gz` → run `./voxkit`

### Installing from source

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/anthropics/voxkit
cd voxkit
pip install -e ".[ui,audio-linux]"    # Linux
pip install -e ".[ui,audio-windows]"  # Windows
voxkit
```

---

## Supported sounds

| Sound | MIDI note |
|---|---|
| Kick | 36 |
| Snare | 38 |
| Closed hi-hat | 42 |
| Open hi-hat | 46 |

Sounds that don't match any of the above (throat bass, mouth pops, ambient noise) are classified as "unknown" and excluded from the MIDI output by default.

---

## Tips for best results

- **Use a consistent mic position.** The classifier learns the tonal fingerprint of your sounds through a specific mic and room. Moving the mic changes the sound enough to confuse it.
- **Calibrate in the same conditions you'll record in.** Same room, same mic gain, same distance.
- **Keep takes short.** VoxKit works best on focused 4–16 bar phrases. Long unbroken recordings give the onset detector more chances to drift.
- **Recalibrate if accuracy drops.** The distribution-shift warning will tell you when to, but you can also recalibrate any time from the menu.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, the dataset tier system, and the coding conventions.

```bash
pip install -e ".[dev,ui,audio-linux]"
pre-commit install
pytest
```

---

## License

Copyright (C) VoxKit contributors. Licensed under the [GNU General Public License v3 or later](LICENSES/GPL-3.0-or-later.txt).

The AVP dataset used for training evaluation is third-party, licensed [CC-BY-4.0](LICENSES/CC-BY-4.0.txt): Blas Ishtar, George Tzanetakis, and AVP Dataset contributors.
