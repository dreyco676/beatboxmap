<!--
SPDX-License-Identifier: GPL-3.0-or-later
Q83 skeleton (committed week 1 per VoxKit-spec-v0.11.md §9). Some
sections are intentionally stubs; flesh out as the implementation lands.
-->

# Contributing to VoxKit

Welcome. VoxKit is GPL v3-or-later, hobby-paced, and TDD-disciplined.
Read this once before your first PR.

## Environment setup

```bash
git clone https://github.com/anthropics/voxkit.git
cd voxkit
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install the package in editable mode plus dev tools.
pip install -e ".[dev,audio-linux]"  # or audio-windows on Windows
pre-commit install
```

Python 3.11 or 3.12. Earlier versions are not supported.

## Running the tests

VoxKit has three dataset tiers (spec §7.10):

```bash
# Synthetic tier — runs on every PR in CI. No external data.
pytest -m "not slow"

# PR-validation tier — minimum-reproducible dataset.
# Set $VOXKIT_DATASETS_DIR to point at the dataset cache.
pytest -m "not slow" --dataset=minimum-reproducible

# Release tier — full canonical dataset. Slow.
pytest --dataset=canonical
```

The synthetic tier validates **pipeline integrity**, not model quality
(Q85). A green synthetic run does NOT mean the model is good. PR-tier
and release-tier are where quality is gated.

## Obtaining datasets (PR validation and beyond)

> **TODO (week 2):** dataset hosting URL + checksum + ToS pointer per Q63.
> Until then, ask in [Discussions](https://github.com/anthropics/voxkit/discussions)
> if you want to run the PR tier locally.

## License expectations

- Project license: **GPL v3-or-later**.
- Every new `.py`, `.pyx`, and `.toml` source file MUST carry an SPDX
  header on the first line:

  ```python
  # SPDX-License-Identifier: GPL-3.0-or-later
  ```

  CI runs `reuse lint` on every PR (Q82). Missing or wrong headers
  block merge. Generated artifacts are exempted in `.reuse/dep5`.

- If your contribution depends on a new third-party model, dataset, or
  large weight file: add a license-review memo under `docs/licenses/`
  using the seven-field template (Q60 amended in v0.11). Pre-training
  data license propagation is the seventh field — don't skip it.

## PR checklist

- [ ] Tests pass: `pytest -m "not slow"`
- [ ] `lint-imports` clean (Q77)
- [ ] `reuse lint` clean (Q82)
- [ ] No new SPDX-header drift on existing files
- [ ] If the PR adds or changes behavior: a test came first (TDD),
      and the commit history shows it (red → green → refactor)
- [ ] Structural changes (Tidy First) committed separately from
      behavioral changes — never mixed in one commit

## TDD discipline

VoxKit follows Kent Beck's TDD methodology with Tidy First refactor
discipline. The test files in `tests/` are the **drivers** of
implementation, not after-the-fact verification. Read
`tests/TDD_README.md` before adding code.

For each new behavior:

1. **Red** — write the failing test first.
2. **Green** — write the minimum implementation to pass.
3. **Refactor** — clean up; tests stay green.

Structural changes (extract / rename / move / inline) land in their
own commit before the next behavioral test, prefixed `tidy:`.
Behavioral changes land separately, prefixed `feat:` or `fix:`.

## Where to ask questions

- **Discussions** for "how do I…" / "should I…" / proposing changes.
- **Issues** for confirmed bugs and concrete tracked work.

Please use Discussions, not Issues, for open-ended questions. The
Issues tracker is the project's punch list, not a forum.
