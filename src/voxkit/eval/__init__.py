# SPDX-License-Identifier: GPL-3.0-or-later
"""voxkit.eval: dev-only evaluation harness. Never bundled with runtime."""

# Single source of truth for the eval scoring code's version. v0.12 T41
# requires this; the harness reads from here rather than hardcoding a
# value at write time.
EVAL_VERSION = "0.0.0"
