# SPDX-License-Identifier: GPL-3.0-or-later
"""Winamp-inspired Qt stylesheet for VoxKit."""

from __future__ import annotations

# Winamp colour palette
_BG = "#232323"          # charcoal window background
_PANEL = "#2e2e2e"       # slightly lighter panel fill
_LCD_BG = "#001200"      # near-black green for "display" areas
_LCD_TEXT = "#00e000"    # bright LCD green
_BTN = "#464646"         # button face
_BTN_LIGHT = "#6e6e6e"  # button bevel highlight
_BTN_DARK = "#111111"   # button bevel shadow
_TEXT = "#d4d4d4"        # normal label text
_DIM = "#5a5a5a"         # disabled / decorative text
_SEL_BG = "#005500"      # selection / hover fill
_SEL_FG = "#00ff00"      # selection text
_BORDER = "#111111"      # hard border

WINAMP_QSS = f"""
/* ── base ───────────────────────────────────────────────────── */
QMainWindow, QDialog {{
    background-color: {_BG};
}}
QWidget {{
    background-color: {_BG};
    color: {_TEXT};
    font-family: "Tahoma", "Segoe UI", sans-serif;
    font-size: 11px;
}}

/* ── menu bar ────────────────────────────────────────────────── */
QMenuBar {{
    background-color: {_PANEL};
    color: {_TEXT};
    border-bottom: 1px solid {_BORDER};
    spacing: 2px;
    padding: 1px 2px;
}}
QMenuBar::item {{
    padding: 2px 8px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background-color: {_SEL_BG};
    color: {_SEL_FG};
}}
QMenu {{
    background-color: {_PANEL};
    color: {_TEXT};
    border: 1px solid {_BTN_LIGHT};
}}
QMenu::item:selected {{
    background-color: {_SEL_BG};
    color: {_SEL_FG};
}}

/* ── LCD "display" panels ────────────────────────────────────── */
QLabel#lcd {{
    background-color: {_LCD_BG};
    color: {_LCD_TEXT};
    font-family: "Courier New", "Consolas", monospace;
    font-size: 12px;
    padding: 3px 6px;
    border-top: 1px solid {_BORDER};
    border-left: 1px solid {_BORDER};
    border-bottom: 1px solid {_BTN_LIGHT};
    border-right: 1px solid {_BTN_LIGHT};
}}
QLabel#lcd_title {{
    background-color: {_LCD_BG};
    color: {_LCD_TEXT};
    font-family: "Courier New", "Consolas", monospace;
    font-size: 14px;
    font-weight: bold;
    padding: 4px 6px;
    border-top: 1px solid {_BORDER};
    border-left: 1px solid {_BORDER};
    border-bottom: 1px solid {_BTN_LIGHT};
    border-right: 1px solid {_BTN_LIGHT};
    letter-spacing: 2px;
}}

/* ── buttons ─────────────────────────────────────────────────── */
QPushButton {{
    background-color: {_BTN};
    color: {_TEXT};
    border-style: solid;
    border-width: 1px;
    border-top-color: {_BTN_LIGHT};
    border-left-color: {_BTN_LIGHT};
    border-bottom-color: {_BTN_DARK};
    border-right-color: {_BTN_DARK};
    padding: 2px 10px;
    min-height: 18px;
    font-size: 11px;
}}
QPushButton:hover:!pressed {{
    background-color: #555555;
}}
QPushButton:pressed {{
    background-color: #3a3a3a;
    border-top-color: {_BTN_DARK};
    border-left-color: {_BTN_DARK};
    border-bottom-color: {_BTN_LIGHT};
    border-right-color: {_BTN_LIGHT};
    padding-top: 3px;
    padding-left: 11px;
}}
QPushButton:disabled {{
    background-color: #383838;
    color: {_DIM};
    border-top-color: #4a4a4a;
    border-left-color: #4a4a4a;
    border-bottom-color: #222222;
    border-right-color: #222222;
}}

/* ── spin boxes (BPM, bars) — LCD look ──────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {_LCD_BG};
    color: {_LCD_TEXT};
    border-top: 1px solid {_BORDER};
    border-left: 1px solid {_BORDER};
    border-bottom: 1px solid {_BTN_LIGHT};
    border-right: 1px solid {_BTN_LIGHT};
    font-family: "Courier New", "Consolas", monospace;
    font-size: 12px;
    padding: 1px 4px;
    min-height: 18px;
    selection-background-color: {_SEL_BG};
    selection-color: {_SEL_FG};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {_BTN};
    border: 1px solid {_BTN_DARK};
    width: 14px;
}}

/* ── combo boxes ─────────────────────────────────────────────── */
QComboBox {{
    background-color: #1a1a1a;
    color: {_TEXT};
    border-top: 1px solid {_BORDER};
    border-left: 1px solid {_BORDER};
    border-bottom: 1px solid {_BTN_LIGHT};
    border-right: 1px solid {_BTN_LIGHT};
    padding: 1px 4px;
    min-height: 18px;
    selection-background-color: {_SEL_BG};
    selection-color: {_SEL_FG};
}}
QComboBox::drop-down {{
    border: none;
    background-color: {_BTN};
    border-left: 1px solid {_BTN_DARK};
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: #1a1a1a;
    color: {_TEXT};
    selection-background-color: {_SEL_BG};
    selection-color: {_SEL_FG};
    border: 1px solid {_BTN_LIGHT};
    outline: none;
}}

/* ── progress bar ────────────────────────────────────────────── */
QProgressBar {{
    background-color: {_LCD_BG};
    color: {_LCD_TEXT};
    border-top: 1px solid {_BORDER};
    border-left: 1px solid {_BORDER};
    border-bottom: 1px solid {_BTN_LIGHT};
    border-right: 1px solid {_BTN_LIGHT};
    text-align: center;
    font-family: "Courier New", monospace;
}}
QProgressBar::chunk {{
    background-color: #00aa00;
}}

/* ── divider lines ───────────────────────────────────────────── */
QFrame[frameShape="4"],  /* HLine */
QFrame[frameShape="5"]  /* VLine */ {{
    color: {_BORDER};
    background-color: {_BTN_LIGHT};
    max-height: 1px;
}}
"""
