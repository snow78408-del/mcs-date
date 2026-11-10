"""流型分类与特征合并的单元测试。"""

from __future__ import annotations

import pytest

from mcs_date.experiment import FlowRegime, NodeFeatures, classify_regime, merge_features


def make_features(**overrides: float) -> NodeFeatures:
    base: dict[str, float] = {
        "q1": 80.0,
        "q2": 20.0,
        "droplet_rate_hz": 0.0,
        "mean_diameter_ratio": 0.5,
        "diameter_cv": 0.05,
        "stream_coverage": 0.8,
        "max_plug_length_ratio": 0.2,
        "sample_count": 30,
        "window_s": 180.0,
    }
    base.update(overrides)
    return NodeFeatures(**base)  # type: ignore[arg-type]


def test_threading_regime() -> None:
    regime, confidence = classify_regime(
        make_features(stream_coverage=0.85, droplet_rate_hz=0.2, mean_diameter_ratio=0.55)
    )
    assert regime is FlowRegime.THREADING
    assert 0.0 < confidence <= 1.0


def test_tubing_checked_before_threading() -> None:
    regime, _ = classify_regime(
        make_features(stream_coverage=0.96, mean_diameter_ratio=0.9, droplet_rate_hz=0.1)
    )
    assert regime is FlowRegime.TUBING


def test_jetting_regime() -> None:
    regime, _ = classify_regime(
        make_features(stream_coverage=0.45, droplet_rate_hz=3.0, mean_diameter_ratio=0.4)
    )
    assert regime is FlowRegime.JETTING


def test_dripping_regime() -> None:
    regime, _ = classify_regime(
        make_features(stream_coverage=0.05, droplet_rate_hz=0.5, mean_diameter_ratio=0.8)
    )
    assert regime is FlowRegime.DRIPPING


def test_displacement_regime() -> None:
    regime, _ = classify_regime(
        make_features(
            stream_coverage=0.30,
            droplet_rate_hz=0.4,
            mean_diameter_ratio=0.3,
            diameter_cv=0.6,
        )
    )
    assert regime is FlowRegime.DISPLACEMENT


def test_unknown_without_samples() -> None:
    regime, confidence = classify_regime(make_features(sample_count=0))
    assert regime is FlowRegime.UNKNOWN
    assert confidence == 0.0


def test_merge_features_weights_by_window() -> None:
    first = make_features(stream_coverage=0.4, window_s=60.0, sample_count=10)
    second = make_features(stream_coverage=0.8, window_s=120.0, sample_count=20)
    merged = merge_features([first, second])
    assert merged.window_s == pytest.approx(180.0)
    assert merged.sample_count == 30
    assert merged.stream_coverage == pytest.approx((0.4 * 60 + 0.8 * 120) / 180.0)


def test_merge_features_rejects_empty() -> None:
    with pytest.raises(ValueError):
        merge_features([])
