# SPDX-License-Identifier: GPL-3.0-or-later
"""Allows `python -m voxkit` to launch the application."""

import sys
from voxkit.ui.app import main

sys.exit(main())
