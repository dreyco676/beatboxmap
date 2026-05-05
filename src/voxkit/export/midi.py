# SPDX-License-Identifier: GPL-3.0-or-later
"""MIDI export and import for VoxKit sessions (§11 Component 9, Q66).

export_midi   — write events to a MIDI Type 1 file (GM drum channel 10).
import_midi   — round-trip reconstruction of Events from a VoxKit MIDI file.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mido

from voxkit.core.session import Event
from voxkit.core.taxonomy import TaxonomyConfig

_DEFAULT_PPQ = 480
_DEFAULT_VELOCITY = 100
_NOTE_DURATION_SECS = 0.030  # 30 ms staccato duration for drum hits
_DRUM_CHANNEL = 9             # mido uses 0-indexed channels; GM channel 10 = index 9


def _seconds_to_ticks(t: float, bpm: float, ppq: int) -> int:
    """Convert a time in seconds to MIDI ticks."""
    return round(t * bpm / 60.0 * ppq)


def export_midi(
    events: list[Event],
    out_path: Path,
    bpm: float,
    taxonomy: TaxonomyConfig,
    *,
    include_unknowns: bool = False,
    unknown_midi_note: int | None = None,
    emit_velocity_from_score: bool = False,
    ticks_per_beat: int = _DEFAULT_PPQ,
    time_signature: tuple[int, int] | None = None,
) -> None:
    """Export events to a MIDI Type 1 file on GM drum channel 10.

    Parameters
    ----------
    events               : list of Events to export (in any order)
    out_path             : destination .mid file path
    bpm                  : session tempo in beats-per-minute
    taxonomy             : TaxonomyConfig that provides the midi_mapping
    include_unknowns     : if True, write unknown-class events using unknown_midi_note
    unknown_midi_note    : MIDI note for unknown-class events; required when include_unknowns=True
    emit_velocity_from_score : if True, map event score linearly to velocity (0.0→1, 1.0→127).
                          WARNING: classifier confidence score is NOT a velocity proxy and
                          does not represent percussion dynamics. Prefer the default fixed
                          velocity. This option may be removed in a future release.
    ticks_per_beat       : PPQ resolution (default 480, the DAW-standard value)
    time_signature       : (numerator, denominator) written as a MIDI meta-event; omitted if None
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    if include_unknowns and unknown_midi_note is None:
        raise ValueError(
            "unknown_midi_note must be supplied when include_unknowns=True"
        )
    if emit_velocity_from_score:
        warnings.warn(
            "emit_velocity_from_score=True: classifier confidence score is NOT a "
            "percussion velocity proxy and will produce unmusical dynamics. "
            "See export_midi docstring for details.",
            UserWarning,
            stacklevel=2,
        )

    unknown_id = taxonomy.unknown_class_id
    for e in events:
        if e.t < 0:
            raise ValueError(
                f"event has negative timestamp: t={e.t!r}; all timestamps must be >= 0"
            )
        if e.class_id != unknown_id and e.class_id not in taxonomy.classes:
            raise ValueError(
                f"class_id {e.class_id!r} not found in taxonomy; "
                f"known classes: {list(taxonomy.classes)}"
            )

    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    # Track 0: meta track — tempo (and optional time signature)
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    if time_signature is not None:
        num, den = time_signature
        meta_track.append(mido.MetaMessage(
            "time_signature", numerator=num, denominator=den, time=0,
        ))
    meta_track.append(mido.MetaMessage(
        "set_tempo", tempo=mido.bpm2tempo(bpm), time=0,
    ))

    # Track 1: drum events on channel 9 (GM channel 10)
    drum_track = mido.MidiTrack()
    mid.tracks.append(drum_track)

    note_dur_ticks = max(1, _seconds_to_ticks(_NOTE_DURATION_SECS, bpm, ticks_per_beat))

    # Collect (abs_tick, priority, message); priority 0=note_on, 1=note_off
    abs_messages: list[tuple[int, int, mido.Message]] = []
    for e in events:
        if e.class_id == unknown_id:
            if not include_unknowns:
                continue
            note = unknown_midi_note
        else:
            note = taxonomy.midi_mapping[e.class_id]

        if emit_velocity_from_score and e.score is not None:
            vel = int(round(max(1.0, min(127.0, e.score * 127.0))))
        else:
            vel = _DEFAULT_VELOCITY

        tick_on = _seconds_to_ticks(e.t, bpm, ticks_per_beat)
        tick_off = tick_on + note_dur_ticks
        abs_messages.append((tick_on, 0, mido.Message(
            "note_on", channel=_DRUM_CHANNEL, note=note, velocity=vel, time=0,
        )))
        abs_messages.append((tick_off, 1, mido.Message(
            "note_off", channel=_DRUM_CHANNEL, note=note, velocity=0, time=0,
        )))

    abs_messages.sort(key=lambda x: (x[0], x[1]))

    prev_tick = 0
    for abs_tick, _, msg in abs_messages:
        delta = abs_tick - prev_tick
        drum_track.append(msg.copy(time=delta))
        prev_tick = abs_tick

    mid.save(str(out_path))


def import_midi(path: Path, taxonomy: TaxonomyConfig) -> list[Event]:
    """Reconstruct Events from a MIDI file produced by export_midi.

    Parameters
    ----------
    path     : path to the .mid file
    taxonomy : same TaxonomyConfig used during export (for inverse note mapping)
    """
    inv_mapping = {note: cls for cls, note in taxonomy.midi_mapping.items()}

    mid = mido.MidiFile(str(path))

    # Parse tempo from the meta track (track 0)
    tempo = mido.bpm2tempo(120.0)
    if mid.tracks:
        for msg in mid.tracks[0]:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break

    bpm = mido.tempo2bpm(tempo)
    ppq = mid.ticks_per_beat

    events: list[Event] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note = msg.note
                if note in inv_mapping:
                    t = abs_tick / ppq * (60.0 / bpm)
                    events.append(Event(t=t, class_id=inv_mapping[note], score=1.0))

    events.sort(key=lambda e: e.t)
    return events
