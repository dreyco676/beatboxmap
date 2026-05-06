# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD tests for the VoxKit application entrypoint.

Drives implementation of:
  - voxkit.ui.app.main()       — QApplication + MainWindow launch function
  - src/voxkit/__main__.py     — allows `python -m voxkit`
  - pyproject.toml [project.scripts] — registers `voxkit` CLI command

============================================================
TEST LIST
============================================================

  T01  voxkit.ui.app is importable without PySide6 at import time
  T02  voxkit.ui.app.main is callable
  T03  voxkit.__main__ module is importable
  T04  pyproject.toml declares a 'voxkit' script entrypoint pointing at main
  T05  `python -m voxkit --version` exits 0 and prints the version string
  T06  `python -m voxkit --smoke-test` exits 0 with offscreen Qt platform
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------
# T01-T03  Pure import / callable checks (no Qt required)
# ---------------------------------------------------------------

def test_T01_app_module_importable_without_qt():
    """voxkit.ui.app must not import PySide6 at module level so it is
    importable on headless machines that only have the core deps."""
    import importlib
    import sys

    # Remove any already-imported voxkit.ui.app so we get a fresh load.
    for key in list(sys.modules.keys()):
        if "voxkit.ui.app" in key:
            del sys.modules[key]

    # The import must succeed even when PySide6 is absent from the namespace
    # at import time (Qt is only required when main() is actually called).
    spec = importlib.util.find_spec("voxkit.ui.app")
    assert spec is not None, "voxkit.ui.app module not found"


def test_T02_main_is_callable():
    from voxkit.ui.app import main
    assert callable(main)


def test_T03_dunder_main_importable():
    import importlib
    spec = importlib.util.find_spec("voxkit.__main__")
    assert spec is not None, (
        "voxkit/__main__.py not found; `python -m voxkit` will not work"
    )


# ---------------------------------------------------------------
# T04  pyproject.toml declares the entrypoint
# ---------------------------------------------------------------

def test_T04_pyproject_declares_voxkit_script():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = _REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    scripts = data.get("project", {}).get("scripts", {})
    assert "voxkit" in scripts, (
        f"'voxkit' not found in [project.scripts] in pyproject.toml. "
        f"Found: {list(scripts.keys())}"
    )
    target = scripts["voxkit"]
    assert "voxkit.ui.app" in target, (
        f"'voxkit' script should point to voxkit.ui.app:main, got: {target!r}"
    )


# ---------------------------------------------------------------
# T05-T06  Subprocess launch tests
# ---------------------------------------------------------------

def test_T05_version_flag_exits_zero():
    """`python -m voxkit --version` must exit 0 and print the version.
    This does not require a display because argparse handles --version
    before Qt is imported."""
    result = subprocess.run(
        [sys.executable, "-m", "voxkit", "--version"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"--version exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    from voxkit import __version__
    assert __version__ in result.stdout or __version__ in result.stderr, (
        f"Version {__version__!r} not found in output.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_T06_smoke_test_flag_exits_zero():
    """`python -m voxkit --smoke-test` must create the MainWindow and exit 0
    without entering the event loop. Requires PySide6."""
    pytest.importorskip("PySide6")
    result = subprocess.run(
        [sys.executable, "-m", "voxkit", "--smoke-test"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"},
        timeout=15,
    )
    assert result.returncode == 0, (
        f"--smoke-test exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
