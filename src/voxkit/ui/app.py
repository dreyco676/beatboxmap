# SPDX-License-Identifier: GPL-3.0-or-later
"""VoxKit application entrypoint.

PySide6 is imported lazily inside main() so this module is importable on
headless machines that have only the core runtime deps installed.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    from voxkit import __version__
    p = argparse.ArgumentParser(
        prog="voxkit",
        description="VoxKit — vocal-percussion to MIDI",
    )
    p.add_argument(
        "--version", action="version", version=f"voxkit {__version__}",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create the main window and exit immediately (for CI / sanity checks).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.exit(
            "PySide6 is not installed.\n"
            "Install it with:  pip install 'voxkit[ui]'"
        )

    app = QApplication.instance() or QApplication(sys.argv[:1])

    from voxkit.ui.style import WINAMP_QSS
    app.setStyleSheet(WINAMP_QSS)

    from voxkit.ui.qt_widgets import MainWindow
    window = MainWindow()
    window.show()

    if args.smoke_test:
        app.processEvents()
        return 0

    return app.exec()
