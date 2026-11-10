"""Measured topology, droplet formation and size/position must remain distinct."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mcs_date.experiment.phase import NodeFeatures, FlowRegime, classify_regime, assess_formation


def measured(**kwargs: object) -> NodeFeatures:
    base = NodeFeatures(q1=90, q2=20, source="camera", sample_count=900, window_s=180,
                        droplet_rate_hz=4, droplet_events=720, mean_diameter_ratio=1.8,
                        diameter_cv=0.08, stream_coverage=0.4, junction_visible=True,
                        breakup_distance_h=3, observation_extent_h=20)
    return replace(base, **kwargs)


def test_jetting_can_make_uniform_droplets_without_being_dripping() -> None:
    f = measured()
    assert classify_regime(f)[0] is FlowRegime.JETTING
    assert assess_formation(f)["droplets_observed"] is True
    assert assess_formation(f)["uniformity_candidate"] is True
    assert assess_formation(f)["best_operating_point_verified"] is False


def test_diameter_alone_cannot_promote_jetting_to_dripping() -> None:
    assert classify_regime(measured(mean_diameter_ratio=3.0, breakup_distance_h=10))[0] is FlowRegime.JETTING


def test_unseen_junction_is_unknown_even_with_low_cv_and_large_candidates() -> None:
    assert classify_regime(measured(junction_visible=False, breakup_distance_h=None))[0] is FlowRegime.UNKNOWN


@pytest.mark.parametrize("changes", [
    {"droplet_events": 0, "droplet_rate_hz": 0},
    {"droplet_events": 1},
    {"droplet_events": None},
    {"breakup_distance_h": 30},
    {"breakup_distance_h": math.nan},
])
def test_inadequate_breakup_evidence_is_unknown(changes: dict[str, object]) -> None:
    assert classify_regime(measured(**changes))[0] is FlowRegime.UNKNOWN


def test_thread_requires_sufficient_field_of_view_and_zero_events() -> None:
    f = measured(breakup_distance_h=None, droplet_events=0, droplet_rate_hz=0, stream_coverage=0.9)
    assert classify_regime(f)[0] is FlowRegime.THREADING
    assert classify_regime(replace(f, observation_extent_h=5))[0] is FlowRegime.UNKNOWN
    assert classify_regime(replace(f, droplet_events=None))[0] is FlowRegime.UNKNOWN


def test_high_cv_does_not_turn_real_jetting_into_unknown_or_dripping() -> None:
    f = measured(diameter_cv=0.5)
    assert classify_regime(f)[0] is FlowRegime.JETTING
    assert assess_formation(f)["uniformity_candidate"] is False


def test_no_events_or_high_cv_never_qualifies_as_uniform_dripping() -> None:
    f = measured(source="heuristic", junction_visible=False, breakup_distance_h=None,
                 stream_coverage=0.03, droplet_rate_hz=1, mean_diameter_ratio=1)
    assert classify_regime(replace(f, droplet_events=0))[0] is not FlowRegime.DRIPPING
    assert classify_regime(replace(f, diameter_cv=0.8))[0] is not FlowRegime.DRIPPING
