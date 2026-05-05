# SPDX-License-Identifier: GPL-3.0-or-later
"""Editor state and lane layout (Q54, Q66)."""

from __future__ import annotations


class EditorState:
    """Tracks first-run guided tour state (Q54)."""

    def __init__(self, tour_completed: bool) -> None:
        self._tour_completed = tour_completed
        self.tour_active: bool = False

    def on_event_observed(self, class_id: str) -> None:
        if class_id == "unknown" and not self._tour_completed:
            self.tour_active = True

    def complete_tour(self) -> None:
        self._tour_completed = True
        self.tour_active = False


class Lane:
    def __init__(self, label: str) -> None:
        self.label = label


class LaneLayout:
    def __init__(self, lanes: list[Lane]) -> None:
        self.lanes = lanes


def build_lane_layout(taxonomy) -> LaneLayout:
    """Build a lane layout with one lane per class plus the unknown lane."""
    lanes = [Lane(label=cls) for cls in taxonomy.classes]
    lanes.append(Lane(label=taxonomy.unknown_class_id))
    return LaneLayout(lanes)
