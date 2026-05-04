# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 9: MIDI exporter.

Drives implementation of `voxkit.export.midi`.

Spec refs: §11 Component 9; v0.9 design (GM drum mapping for default
4 classes), Q66 (TaxonomyConfig drives note mapping; default mappings
unchanged: kick=36, snare=38, closed_hat=42, open_hat=46),
§7.5 (round-trip test), unknowns excluded by default.

============================================================
TEST LIST (implement strictly in order)
============================================================

Trivial cases
  T01  Exporting an empty event list produces a parseable MIDI file
  T02  Exporting a single kick event produces a file with exactly 1 note-on

GM drum mapping (Q66 defaults)
  T03  kick      → MIDI note 36
  T04  snare     → MIDI note 38
  T05  closed_hat → MIDI note 42
  T06  open_hat  → MIDI note 46
  T07  Mapping is taken from TaxonomyConfig.midi_mapping, not hardcoded

Note timing
  T08  Event at t=0.5s at 120 BPM lands at MIDI tick = ticks_per_beat
  T09  Two events at the same time produce simultaneous note-ons
  T10  Note-off follows note-on within a fixed short duration (e.g., 30ms)
  T11  Tempo meta-event present at tick 0 reflects session BPM

Unknown class handling (default behavior)
  T12  Events with class_id == "unknown" are excluded by default
  T13  include_unknowns=True keeps unknowns; they map to a configurable note
  T14  include_unknowns=True with no unknown mapping configured raises

Pre-conditions
  T15  Export rejects events with negative timestamps
  T16  Export rejects events with class_id not in taxonomy
  T17  Export rejects bpm <= 0

  -- TIDY FIRST before T18: extract `_seconds_to_ticks(t, bpm, ppq)`
     to a pure helper. Used by both export and (future) re-import.

MIDI round-trip (§7.5)
  T18  Round-trip preserves event count
  T19  Round-trip preserves class identities (via inverse mapping)
  T20  Round-trip preserves event ordering
  T21  Round-trip timing within ±1 tick of the original

Custom taxonomy (Q66)
  T22  A 5-class taxonomy with a custom note mapping exports correctly
  T23  Two custom taxonomies with the same classes but different note
       mappings produce different MIDI output

Velocity
  T24  Default note-on velocity is fixed (e.g., 100) when score is None
  T25  Velocity is derived from score when emit_velocity_from_score=True
       (linear: 0.0 → 1, 1.0 → 127)
  T26  Velocity is clamped to [1, 127]

File format
  T27  Output file is valid MIDI Type 1 (multi-track)
  T28  All percussion events go on MIDI channel 10 (general MIDI drums)

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

Velocity-mapping nuance (Marco, Jordan, Lin, Alex, Casey, Riley: 6/9)
  T29  T25's linear mapping (0.0 → 1, 1.0 → 127) is VERY different from
       musical perception. Real percussion velocity perception is
       roughly logarithmic. T29 documents the chosen mapping (linear is
       the v1.0 default for honesty about what we ship; a logarithmic
       option is tracked as an open question for v1.1).
  T30  Velocity from a *classifier confidence score* (0..1) is a poor
       proxy for percussion velocity. T29's `emit_velocity_from_score`
       carries a docstring warning; T30 asserts the public API exposes
       this with a deprecation marker so users don't accidentally rely
       on it for musical dynamics. RECORDED AS OQ — if the team agrees,
       remove emit_velocity_from_score from v1.0 entirely (it's a
       feature trap masquerading as a feature).

Tempo + time-signature export integrity (Marco, Lin, Alex, Sam,
Casey, Riley, Jordan: 7/9)
  T31  Time signature meta-event present at tick 0 reflects session
       time_signature (T11 covers tempo only; the spec implies both go
       in the meta track but no test enforces signature).
  T32  ticks_per_beat (PPQ) is documented and stable across versions
       (default 480; configurable). Tests both that the default is 480
       AND that overriding produces a file with the new PPQ.

Round-trip strictness on edge cases (Sam, Lin, Alex, Casey, Marco,
Riley: 6/9)
  T33  Round-trip with simultaneous events (two kicks at exactly the
       same timestamp) preserves both — current T19 only checks
       sorted-class-id equality which would pass even if duplicates
       were merged.
  T34  Round-trip with an event at t=0.0 (very first sample) does NOT
       lose the event (some MIDI parsers drop tick-0 events without a
       preceding meta event).

============================================================
v0.12 PANEL ADDITIONS (principal-engineer + Casey/Riley synthesis;
those reviewers rate-limited)
============================================================

Discoverability of velocity-from-score risk (TIGHTEN — T30 anchors a
docstring substring, which is brittle and not how Python signals
deprecated/risky parameters)
  T35  emit_velocity_from_score=True emits a runtime warning
       (UserWarning) the FIRST time it's invoked per process. Pinning
       the warning name (vs the docstring text) gives users a real
       signal — IDEs, linters, and pytest -W all surface it. v0.12
       also keeps T30's docstring check as a soft guarantee but T35
       is the load-bearing one.

Real-load smoke (STRONG — every existing round-trip test uses ≤ 4
events; a 32-bar session with hundreds of events exercises the
ticks-per-beat math and any per-event-list O(n²) bug)
  T36  Round-trip a 32-bar session with ~256 events at 1/16 grid
       preserves event count exactly. Today's tests don't exercise
       the realistic-load path; an O(n²) bug or PPQ-overflow at high
       event counts would ship undetected.

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OQ-1  Logarithmic velocity option for v1.1 (above).
OQ-2  Remove emit_velocity_from_score for v1.0? v0.12 (Casey): T35
      adds a warning-on-use which gives users a chance to opt out
      without removing the parameter entirely. Defer hard removal.
OQ-3  Export of unknown-class events at a configurable channel separate
      from channel 10 (so the user can route them to a different
      VST/sampler). [Jordan, Marco: 2/9 — defer to v1.1.]
OQ-4  SMF Type 0 (single-track) export option for compatibility with
      legacy DAWs. [Casey: 1/9 — REJECTED. Anyone using a DAW from
      before SMF Type 1 (1988) is on their own.]
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

@pytest.fixture
def default_taxonomy():
    from voxkit.core.taxonomy import TaxonomyConfig
    return TaxonomyConfig.default_v1_0()


def _make_event(t, cls, score=0.9):
    from voxkit.core.session import Event
    return Event(t=t, class_id=cls, score=score)


def _read_midi(path: Path):
    """Helper that reads a MIDI file and returns (tempo, list of (tick, type, note, velocity))."""
    import mido
    mid = mido.MidiFile(path)
    events = []
    tempo = None
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo = mido.tempo2bpm(msg.tempo)
            elif msg.type == "note_on":
                events.append((tick, "note_on", msg.note, msg.velocity, msg.channel))
            elif msg.type == "note_off":
                events.append((tick, "note_off", msg.note, msg.velocity, msg.channel))
    return mid.ticks_per_beat, tempo, events


# ---------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------

def test_T01_empty_event_list_produces_parseable_midi(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "empty.mid"
    export_midi(events=[], out_path=out, bpm=120, taxonomy=default_taxonomy)
    ppq, _, events = _read_midi(out)
    assert ppq > 0
    assert all(e[1] not in ("note_on", "note_off") for e in events)


def test_T02_single_kick_produces_one_note_on(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "kick.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
    )
    _, _, events = _read_midi(out)
    note_ons = [e for e in events if e[1] == "note_on" and e[3] > 0]
    assert len(note_ons) == 1


# ---------------------------------------------------------------
# GM drum mapping (Q66 defaults)
# ---------------------------------------------------------------

@pytest.mark.parametrize("cls,expected_note", [
    ("kick", 36),
    ("snare", 38),
    ("closed_hat", 42),
    ("open_hat", 46),
])
def test_T03_to_T06_gm_default_mapping(tmp_path, default_taxonomy, cls, expected_note):
    from voxkit.export.midi import export_midi
    out = tmp_path / f"{cls}.mid"
    export_midi(
        events=[_make_event(0.5, cls)],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
    )
    _, _, events = _read_midi(out)
    note_ons = [e for e in events if e[1] == "note_on" and e[3] > 0]
    assert note_ons[0][2] == expected_note


def test_T07_mapping_taken_from_taxonomy_not_hardcoded(tmp_path):
    """Q66: classifier and exporter must read mapping from TaxonomyConfig.
    A custom taxonomy with kick=60 must produce note 60, not 36."""
    from voxkit.core.taxonomy import TaxonomyConfig
    from voxkit.export.midi import export_midi
    custom = TaxonomyConfig(
        classes=("kick", "snare", "closed_hat", "open_hat"),
        midi_mapping={"kick": 60, "snare": 62, "closed_hat": 64, "open_hat": 65},
    )
    out = tmp_path / "custom.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],
        out_path=out, bpm=120, taxonomy=custom,
    )
    _, _, events = _read_midi(out)
    note_ons = [e for e in events if e[1] == "note_on" and e[3] > 0]
    assert note_ons[0][2] == 60


# ---------------------------------------------------------------
# Note timing
# ---------------------------------------------------------------

def test_T08_event_at_one_beat_lands_at_one_ppq_tick(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "timing.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],   # 0.5s @ 120 BPM = 1 beat
        out_path=out, bpm=120, taxonomy=default_taxonomy,
    )
    ppq, _, events = _read_midi(out)
    note_ons = [e for e in events if e[1] == "note_on" and e[3] > 0]
    assert abs(note_ons[0][0] - ppq) <= 1


def test_T09_simultaneous_events_have_same_tick(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "sim.mid"
    export_midi(
        events=[_make_event(0.5, "kick"), _make_event(0.5, "snare")],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
    )
    _, _, events = _read_midi(out)
    note_ons = sorted([e for e in events if e[1] == "note_on" and e[3] > 0])
    assert note_ons[0][0] == note_ons[1][0]


def test_T10_note_off_follows_note_on_within_short_duration(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "off.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
    )
    ppq, _, events = _read_midi(out)
    note_on_tick = [e[0] for e in events if e[1] == "note_on" and e[3] > 0][0]
    note_off_tick = [e[0] for e in events if e[1] == "note_off" or (e[1] == "note_on" and e[3] == 0)][0]
    duration_ticks = note_off_tick - note_on_tick
    # 30 ms @ 120 BPM = 0.06 beats → ~0.06 × ppq ticks
    assert 0 < duration_ticks <= int(0.1 * ppq)   # at most ~50ms


def test_T11_tempo_meta_event_reflects_session_bpm(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "tempo.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],
        out_path=out, bpm=92, taxonomy=default_taxonomy,
    )
    _, tempo, _ = _read_midi(out)
    assert abs(tempo - 92) < 0.5


# ---------------------------------------------------------------
# Unknown class handling
# ---------------------------------------------------------------

def test_T12_unknowns_excluded_by_default(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "with_unknown.mid"
    export_midi(
        events=[_make_event(0.5, "unknown"), _make_event(1.0, "kick")],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
    )
    _, _, events = _read_midi(out)
    note_ons = [e for e in events if e[1] == "note_on" and e[3] > 0]
    assert len(note_ons) == 1


def test_T13_include_unknowns_uses_configured_note(tmp_path):
    from voxkit.core.taxonomy import TaxonomyConfig
    from voxkit.export.midi import export_midi
    tax = TaxonomyConfig(
        classes=("kick",), midi_mapping={"kick": 36},
        unknown_class_id="unknown",
    )
    out = tmp_path / "unknown_kept.mid"
    export_midi(
        events=[_make_event(0.5, "unknown"), _make_event(1.0, "kick")],
        out_path=out, bpm=120, taxonomy=tax,
        include_unknowns=True, unknown_midi_note=37,
    )
    _, _, events = _read_midi(out)
    note_ons = [e for e in events if e[1] == "note_on" and e[3] > 0]
    notes = sorted(e[2] for e in note_ons)
    assert notes == [36, 37]


def test_T14_include_unknowns_without_mapping_raises(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "should_fail.mid"
    with pytest.raises(ValueError, match="unknown_midi_note"):
        export_midi(
            events=[_make_event(0.5, "unknown")],
            out_path=out, bpm=120, taxonomy=default_taxonomy,
            include_unknowns=True,
        )


# ---------------------------------------------------------------
# Pre-conditions
# ---------------------------------------------------------------

def test_T15_negative_timestamp_rejected(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    with pytest.raises(ValueError, match="timestamp"):
        export_midi(
            events=[_make_event(-0.1, "kick")],
            out_path=tmp_path / "x.mid", bpm=120, taxonomy=default_taxonomy,
        )


def test_T16_class_not_in_taxonomy_rejected(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    with pytest.raises(ValueError, match="taxonomy"):
        export_midi(
            events=[_make_event(0.5, "tabla")],
            out_path=tmp_path / "x.mid", bpm=120, taxonomy=default_taxonomy,
        )


def test_T17_zero_bpm_rejected(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    with pytest.raises(ValueError, match="bpm"):
        export_midi(
            events=[_make_event(0.5, "kick")],
            out_path=tmp_path / "x.mid", bpm=0, taxonomy=default_taxonomy,
        )


# ----- TIDY FIRST checkpoint -----
# Extract `_seconds_to_ticks(t, bpm, ppq)` to a pure helper. Pure
# structural change.


# ---------------------------------------------------------------
# MIDI round-trip (§7.5)
# ---------------------------------------------------------------

def _round_trip_events(events, bpm, taxonomy, tmp_path):
    from voxkit.export.midi import export_midi, import_midi
    out = tmp_path / "rt.mid"
    export_midi(events=events, out_path=out, bpm=bpm, taxonomy=taxonomy)
    return import_midi(path=out, taxonomy=taxonomy)


def test_T18_round_trip_preserves_event_count(tmp_path, default_taxonomy):
    events = [
        _make_event(0.10, "kick"),
        _make_event(0.25, "snare"),
        _make_event(0.50, "closed_hat"),
        _make_event(0.75, "kick"),
    ]
    rt = _round_trip_events(events, bpm=120, taxonomy=default_taxonomy, tmp_path=tmp_path)
    assert len(rt) == len(events)


def test_T19_round_trip_preserves_class_identities(tmp_path, default_taxonomy):
    events = [
        _make_event(0.10, "kick"),
        _make_event(0.25, "snare"),
        _make_event(0.50, "closed_hat"),
        _make_event(0.75, "open_hat"),
    ]
    rt = _round_trip_events(events, bpm=120, taxonomy=default_taxonomy, tmp_path=tmp_path)
    assert sorted(e.class_id for e in rt) == sorted(e.class_id for e in events)


def test_T20_round_trip_preserves_ordering(tmp_path, default_taxonomy):
    events = [
        _make_event(0.10, "kick"),
        _make_event(0.25, "snare"),
        _make_event(0.50, "closed_hat"),
    ]
    rt = _round_trip_events(events, bpm=120, taxonomy=default_taxonomy, tmp_path=tmp_path)
    assert [e.class_id for e in rt] == [e.class_id for e in events]


def test_T21_round_trip_timing_within_one_tick(tmp_path, default_taxonomy):
    events = [
        _make_event(0.10, "kick"),
        _make_event(0.25, "snare"),
        _make_event(0.50, "closed_hat"),
    ]
    rt = _round_trip_events(events, bpm=120, taxonomy=default_taxonomy, tmp_path=tmp_path)
    # 1 tick at ppq=480 and 120 BPM is ~1.04 ms; allow 2 ms.
    for orig, returned in zip(events, rt):
        assert abs(orig.t - returned.t) < 0.002


# ---------------------------------------------------------------
# Custom taxonomy (Q66)
# ---------------------------------------------------------------

def test_T22_5_class_taxonomy_exports(tmp_path):
    from voxkit.core.taxonomy import TaxonomyConfig
    from voxkit.export.midi import export_midi
    tax = TaxonomyConfig(
        classes=("kick", "snare", "closed_hat", "open_hat", "throat_bass"),
        midi_mapping={"kick": 36, "snare": 38, "closed_hat": 42, "open_hat": 46, "throat_bass": 35},
    )
    out = tmp_path / "five.mid"
    export_midi(
        events=[_make_event(0.5, "throat_bass")],
        out_path=out, bpm=120, taxonomy=tax,
    )
    _, _, events = _read_midi(out)
    note_ons = [e for e in events if e[1] == "note_on" and e[3] > 0]
    assert note_ons[0][2] == 35


def test_T23_different_mappings_produce_different_output(tmp_path):
    from voxkit.core.taxonomy import TaxonomyConfig
    from voxkit.export.midi import export_midi
    tax_a = TaxonomyConfig(
        classes=("kick",), midi_mapping={"kick": 36},
    )
    tax_b = TaxonomyConfig(
        classes=("kick",), midi_mapping={"kick": 60},
    )
    out_a = tmp_path / "a.mid"
    out_b = tmp_path / "b.mid"
    export_midi(events=[_make_event(0.5, "kick")], out_path=out_a, bpm=120, taxonomy=tax_a)
    export_midi(events=[_make_event(0.5, "kick")], out_path=out_b, bpm=120, taxonomy=tax_b)
    _, _, events_a = _read_midi(out_a)
    _, _, events_b = _read_midi(out_b)
    note_a = [e for e in events_a if e[1] == "note_on" and e[3] > 0][0][2]
    note_b = [e for e in events_b if e[1] == "note_on" and e[3] > 0][0][2]
    assert note_a == 36 and note_b == 60


# ---------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------

def test_T24_default_velocity_when_no_score(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    from voxkit.core.session import Event
    e = Event(t=0.5, class_id="kick", score=None)
    out = tmp_path / "vel.mid"
    export_midi(events=[e], out_path=out, bpm=120, taxonomy=default_taxonomy)
    _, _, events = _read_midi(out)
    note_ons = [evt for evt in events if evt[1] == "note_on" and evt[3] > 0]
    assert note_ons[0][3] == 100   # default


def test_T25_velocity_derived_from_score(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "scored.mid"
    export_midi(
        events=[_make_event(0.5, "kick", score=1.0)],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
        emit_velocity_from_score=True,
    )
    _, _, events = _read_midi(out)
    note_ons = [evt for evt in events if evt[1] == "note_on" and evt[3] > 0]
    assert note_ons[0][3] == 127


def test_T26_velocity_clamped_to_unit_range(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "clamp.mid"
    export_midi(
        events=[
            _make_event(0.5, "kick", score=0.0),
            _make_event(1.0, "kick", score=1.5),    # over-saturated
        ],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
        emit_velocity_from_score=True,
    )
    _, _, events = _read_midi(out)
    note_ons = sorted([evt for evt in events if evt[1] == "note_on" and evt[3] > 0])
    assert note_ons[0][3] >= 1
    assert note_ons[1][3] == 127


# ---------------------------------------------------------------
# File format
# ---------------------------------------------------------------

def test_T27_output_is_valid_midi_type_1(tmp_path, default_taxonomy):
    import mido
    from voxkit.export.midi import export_midi
    out = tmp_path / "fmt.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
    )
    mid = mido.MidiFile(out)
    assert mid.type == 1


def test_T28_percussion_on_channel_10(tmp_path, default_taxonomy):
    from voxkit.export.midi import export_midi
    out = tmp_path / "ch10.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
    )
    _, _, events = _read_midi(out)
    note_ons = [e for e in events if e[1] == "note_on" and e[3] > 0]
    # MIDI channels are 0-indexed in mido; channel 10 = index 9.
    assert note_ons[0][4] == 9


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T29_velocity_mapping_documented_as_linear(tmp_path, default_taxonomy):
    """T25 already enforces linear 0..1 → 1..127. T29 documents the
    panel choice and pins midpoint behavior so a future "improvement"
    to logarithmic doesn't silently land. If the panel approves a
    logarithmic option for v1.1, expose it as a separate parameter
    (e.g., velocity_curve='linear'|'log') and add a new test."""
    from voxkit.export.midi import export_midi
    out = tmp_path / "vel_mid.mid"
    export_midi(
        events=[_make_event(0.5, "kick", score=0.5)],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
        emit_velocity_from_score=True,
    )
    _, _, events = _read_midi(out)
    note_ons = [evt for evt in events if evt[1] == "note_on" and evt[3] > 0]
    # Linear: score 0.5 → velocity 64 (0.5 * 127 = 63.5, rounded to 64)
    assert 60 <= note_ons[0][3] <= 68, (
        f"midpoint score should produce midpoint velocity (linear); "
        f"got {note_ons[0][3]}"
    )


def test_T30_emit_velocity_from_score_carries_warning():
    """Classifier confidence score is NOT a percussion velocity. The
    public API parameter must be discoverable as such — either via a
    deprecation warning, a docstring marker, or both — so users don't
    silently get unmusical dynamics when they enable it."""
    import inspect
    from voxkit.export import midi
    sig = inspect.signature(midi.export_midi)
    assert "emit_velocity_from_score" in sig.parameters
    doc = inspect.getdoc(midi.export_midi) or ""
    # The docstring should explicitly warn about the proxy.
    assert "score" in doc.lower() and (
        "not" in doc.lower() or "proxy" in doc.lower() or "warn" in doc.lower()
    ), "export_midi docstring must warn that score → velocity is a proxy"


def test_T31_time_signature_meta_event_present(tmp_path, default_taxonomy):
    """The MIDI file must record both tempo (T11) and time signature.
    Without time signature, DAWs default to 4/4 and grid-display goes
    wrong on 3/4 / 7/8 sessions."""
    import mido
    from voxkit.export.midi import export_midi
    out = tmp_path / "ts.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],
        out_path=out, bpm=120, taxonomy=default_taxonomy,
        time_signature=(3, 4),
    )
    mid = mido.MidiFile(out)
    found = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == "time_signature":
                found = (msg.numerator, msg.denominator)
    assert found == (3, 4), f"expected 3/4 time signature meta-event, got {found}"


def test_T32_default_ppq_is_480_and_configurable(tmp_path, default_taxonomy):
    """480 PPQ is the de-facto DAW standard; ship that by default. Allow
    override for users who need higher resolution (e.g., 960 PPQ)."""
    import mido
    from voxkit.export.midi import export_midi

    out_default = tmp_path / "ppq_default.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],
        out_path=out_default, bpm=120, taxonomy=default_taxonomy,
    )
    assert mido.MidiFile(out_default).ticks_per_beat == 480

    out_high = tmp_path / "ppq_960.mid"
    export_midi(
        events=[_make_event(0.5, "kick")],
        out_path=out_high, bpm=120, taxonomy=default_taxonomy,
        ticks_per_beat=960,
    )
    assert mido.MidiFile(out_high).ticks_per_beat == 960


def test_T33_round_trip_preserves_simultaneous_duplicates(tmp_path, default_taxonomy):
    """T19 sorts class IDs and compares — would pass if a duplicate was
    silently merged. T33 enforces that two identical events at the same
    timestamp survive round-trip as TWO events."""
    events = [
        _make_event(0.5, "kick"),
        _make_event(0.5, "kick"),    # duplicate at same instant
    ]
    rt = _round_trip_events(events, bpm=120, taxonomy=default_taxonomy, tmp_path=tmp_path)
    kick_count = sum(1 for e in rt if e.class_id == "kick")
    assert kick_count == 2, f"duplicate at same instant lost in round-trip; got {kick_count}"


def test_T34_round_trip_preserves_event_at_t_zero(tmp_path, default_taxonomy):
    """An event at the very first sample is a real case (precise
    quantization to bar-1, beat-1). Some MIDI parsers drop tick-0
    note-ons that come before a tempo/meta event."""
    events = [_make_event(0.0, "kick"), _make_event(0.5, "snare")]
    rt = _round_trip_events(events, bpm=120, taxonomy=default_taxonomy, tmp_path=tmp_path)
    assert len(rt) == 2
    assert rt[0].class_id == "kick"
    assert abs(rt[0].t) < 0.002


# ---------------------------------------------------------------
# v0.12 panel additions (principal-engineer + Casey/Riley synthesis)
# ---------------------------------------------------------------

def test_T35_emit_velocity_from_score_emits_runtime_warning(
    tmp_path, default_taxonomy,
):
    """v0.12: T30 anchored a docstring substring — brittle and IDE-
    invisible. Pinning a runtime UserWarning gives users (and pytest -W,
    and IDEs, and linters) a real signal that classifier confidence
    score is NOT a percussion velocity."""
    import warnings
    from voxkit.export.midi import export_midi
    out = tmp_path / "warns.mid"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_midi(
            events=[_make_event(0.5, "kick", score=0.5)],
            out_path=out, bpm=120, taxonomy=default_taxonomy,
            emit_velocity_from_score=True,
        )
    relevant = [w for w in caught
                if issubclass(w.category, UserWarning)
                and "score" in str(w.message).lower()]
    assert relevant, (
        "emit_velocity_from_score=True must emit a UserWarning that "
        "mentions 'score' so users have a discoverable signal that "
        "classifier confidence is not a velocity proxy"
    )


def test_T36_round_trip_32_bar_session_with_many_events(
    tmp_path, default_taxonomy,
):
    """v0.12: every existing round-trip test uses ≤ 4 events. A 32-bar
    1/16 session has up to ~512 grid positions; exercise the realistic-
    load path so an O(n²) bug in the exporter or PPQ-overflow at high
    event counts is caught here, not in production."""
    bpm = 120
    bar_seconds = 4 * (60.0 / bpm)
    n_bars = 32
    n_events_per_bar = 8
    events = []
    for bar in range(n_bars):
        for slot in range(n_events_per_bar):
            t = bar * bar_seconds + slot * (bar_seconds / n_events_per_bar)
            cls = ("kick", "snare", "closed_hat", "open_hat")[slot % 4]
            events.append(_make_event(t, cls))

    rt = _round_trip_events(events, bpm=bpm, taxonomy=default_taxonomy,
                            tmp_path=tmp_path)
    assert len(rt) == len(events), (
        f"32-bar realistic load lost events: in={len(events)}, out={len(rt)}"
    )
