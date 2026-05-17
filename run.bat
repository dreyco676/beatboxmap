:: SPDX-FileCopyrightText: 2026 John Hogue
:: SPDX-License-Identifier: GPL-3.0-or-later
@echo off
REM Launch VoxKit using the conda voxkit environment (Python 3.12 + VS2022 runtime,
REM required for PySide6 6.11.1 -- the .venv-win Python 3.13 is built with MSVC
REM v.1929 which is missing functions that PySide6's Qt DLLs expect from VS2022).
start "" "%USERPROFILE%\anaconda3\envs\voxkit\pythonw.exe" -m voxkit %*
