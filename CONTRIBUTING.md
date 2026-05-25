<!--
SPDX-License-Identifier: GPL-3.0-or-later
Q83 skeleton (committed week 1 per VoxKit-spec-v0.11.md §9).
-->

# Contributing to VoxKit

Welcome. VoxKit is GPL v3-or-later, hobby-paced, and TDD-disciplined.
Read this once before your first PR.

## Contents

- [Environment setup](#environment-setup)
- [Running the tests](#running-the-tests)
- [Datasets](#datasets)
- [ONNX models](#onnx-models)
- [SPDX license headers](#spdx-license-headers)
- [PR checklist](#pr-checklist)
- [TDD discipline](#tdd-discipline)
- [Commit conventions](#commit-conventions)
- [CI overview](#ci-overview)
- [Where to ask questions](#where-to-ask-questions)

---

## Environment setup

```bash
git clone https://github.com/anthropics/voxkit.git
cd voxkit
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install the package in editable mode plus all dev tools.
pip install -e ".[dev,audio-linux]"  # Linux
pip install -e ".[dev,audio-windows]"  # Windows

# Install git hooks (SPDX check + import-linter run on every commit).
pre-commit install
```

**Python 3.11 or 3.12 required.** Earlier versions are not supported.

On Linux, sounddevice requires PortAudio:

```bash
sudo apt-get install libportaudio2   # Debian/Ubuntu / Linux Mint
sudo dnf install portaudio           # Fedora
```

**Optional: real-time thread priority (SCHED_FIFO) on Linux.**
VoxKit's resampler worker requests `SCHED_FIFO` priority 80 (Q67) for
lower-latency audio.  Without it the app falls back silently to normal
scheduling — usable but potentially droppy on a loaded CPU.

To enable it, grant the Python binary `cap_sys_nice`:

```bash
# Replace the path with your venv's python3 if using a virtual environment.
sudo setcap cap_sys_nice+ep $(which python3)

# Verify:
getcap $(which python3)
# Expected output: /usr/bin/python3 = cap_sys_nice+ep
```

> **Security note:** `cap_sys_nice` lets that binary change scheduling
> priorities for its own threads.  It does not grant root or allow
> modification of other processes.  Limit this to a venv python if you
> are security-conscious:
> ```bash
> sudo setcap cap_sys_nice+ep .venv/bin/python3
> ```

---

## Running the tests

VoxKit has three dataset tiers (spec §7.10, Q85):

```bash
# Synthetic tier — no external data; runs in CI on every PR.
pytest -m "not slow"

# PR-validation tier — needs the minimum-reproducible dataset (see below).
pytest -m "not slow" --dataset=minimum-reproducible

# Release tier — full canonical dataset. Slow; run before tagging a release.
pytest --dataset=canonical
```

> **The synthetic tier validates pipeline integrity, not model quality (Q85).**
> A green synthetic run does not mean the model is good — it means the code
> runs end-to-end. Model quality is gated at PR-tier and release-tier.

---

## Datasets

### Minimum-reproducible tier — AVP dataset

The PR-validation and release tiers use the **AVP-LVT v4 dataset**
(Acoustic Vocal Percussion Labelled Verse Takes, CC-BY-4.0).

1. Download `AVP_Dataset.zip` from Zenodo:
   - Record ID **5036529** (original release) or **5578744** (v4 update)
   - URL: `https://zenodo.org/record/5036529`

2. Place and unzip:

   ```bash
   mkdir -p data/avp
   cp AVP_Dataset.zip data/avp/
   cd data/avp && unzip AVP_Dataset.zip
   ```

   Expected layout after unzip:

   ```
   data/avp/AVP_Dataset/Personal/
     Participant_1/P1_Kick_Personal.wav
     Participant_1/P1_Kick_Personal.csv
     Participant_1/P1_Snare_Personal.wav
     ...
   ```

3. Run the LOSO evaluation to verify the dataset is readable:

   ```bash
   python scripts/run_avp_loso.py --substrate panns --use-cache
   ```

   Results are written to `data/avp_loso_panns.json`.

The AVP zip is excluded from git via `.gitignore`. Do not commit it.

---

## ONNX models

The embedding extractor requires an ONNX model file in `models/`.
These are not committed to git (too large; `.gitignore` excludes `models/*.onnx`).

### PANNs CNN14

```bash
python scripts/convert_panns_to_onnx.py
# Writes: models/panns_cnn14_16k.onnx
```

The script downloads the PANNs CNN14 checkpoint from the official source
and converts it. Run once; the result is cached at `models/`.

### BEATs (optional — substrate bake-off)

```bash
python scripts/convert_beats_to_onnx.py
# Writes: models/beats_iter3plus_as2m.onnx
```

Required only if you want to run `--substrate beats` or `--substrate both`.

---

## SPDX license headers

Every new `.py`, `.pyx`, `.toml`, `.yml`, and `.yaml` source file **must**
carry an SPDX header as its first line:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
```

For non-comment formats (TOML, YAML use `#`; HTML/Markdown use `<!-- -->`):

```yaml
# SPDX-License-Identifier: GPL-3.0-or-later
```

```markdown
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
```

CI runs `reuse lint` on every push (Q82). Missing or wrong headers block
merge. Generated artifacts that cannot carry headers are exempted in
`.reuse/dep5`.

### Third-party models and datasets

If your contribution depends on a new third-party model, dataset, or large
weight file, add a license-review memo under `docs/licenses/` using the
seven-field template (Q60). The seventh field — pre-training data license
propagation — is mandatory; do not omit it.

---

## PR checklist

Before opening a PR, verify each item locally:

- [ ] **Tests pass** — `pytest -m "not slow"` is fully green
- [ ] **Import-linter clean** — `lint-imports` reports 0 broken contracts (Q77)
- [ ] **SPDX headers clean** — `reuse lint` exits 0 (Q82)
- [ ] **New behaviour has a test first** — commit history shows red → green →
      refactor; the failing test landed before the implementation
- [ ] **Structural changes are separate** — Tidy First refactors are in their
      own commit (`tidy:` prefix), never mixed with behavioural changes
- [ ] **No SPDX drift** — every new or renamed source file has the correct
      header; no existing file lost its header

If your PR touches the eval harness or scoring code, also bump `EVAL_VERSION`
in `src/voxkit/eval/__init__.py` (Q41).

---

## TDD discipline

VoxKit follows Kent Beck's TDD methodology with Tidy First refactor discipline.
The test files in `tests/` **drive** implementation — they are not
after-the-fact verification. Read `tests/TDD_README.md` before adding code.

For each new behaviour:

1. **Red** — write the failing test first; run it and confirm it fails for
   the right reason (usually `ImportError`, then `AssertionError`).
2. **Green** — write the minimum implementation to make the test pass.
   Resist over-engineering.
3. **Refactor** — clean up while the tests stay green.

Structural changes (extract / rename / move / inline) land in their own
commit, prefixed `tidy:`, before the next behavioural test. Never mix
structural and behavioural changes in one commit.

---

## Commit conventions

```
feat: short description of new behaviour
fix:  short description of bug fix
tidy: short description of structural change (no behaviour change)
test: add or fix tests without touching production code
docs: documentation only
ci:   CI workflow changes
```

Examples:

```
tidy: extract _loso_macro_f1 to shared helper
feat: add RecordingPanelWidget device picker (Q24, Q73)
fix:  use average=None in per-class F1 to avoid binary-only error
```

Keep the first line under 72 characters. Reference spec questions (`Q77`,
`Q83`) where relevant; they're the canonical requirement IDs.

---

## CI overview

Every push and PR triggers two jobs (`.github/workflows/ci.yml`):

| Job | What it checks |
|-----|----------------|
| `static-checks` | `reuse lint` (SPDX, Q82) then `lint-imports` (Q77) |
| `tests-synthetic` | `pytest -m "not slow"` on Linux + Windows × Python 3.11/3.12 |

`static-checks` runs first; `tests-synthetic` is blocked until it passes.

PR-validation tier (minimum-reproducible dataset) and release tier (canonical
dataset) are triggered manually or on tag — they need the dataset cache that
is too large to include on every PR.

To reproduce CI locally:

```bash
# Static checks
reuse lint
lint-imports

# Synthetic-tier tests (matches CI matrix entry)
pytest -m "not slow"
```

---

## Where to ask questions

- **[Discussions](https://github.com/anthropics/voxkit/discussions)** — "how
  do I…", "should I…", proposing design changes, open-ended questions.
- **[Issues](https://github.com/anthropics/voxkit/issues)** — confirmed bugs
  and concrete tracked work items only.

Please use Discussions, not Issues, for questions. The Issues tracker is the
project's punch list, not a forum.
