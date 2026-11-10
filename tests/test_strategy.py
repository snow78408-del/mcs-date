"""Bounded deterministic scan regressions, not physical-boundary validation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import pytest

from mcs_date.experiment import (
    ChipModelConfig,
    ChipSimulator,
    FlowRegime,
    ScanConfig,
    SpiderScanner,
    run_offline_scan,
    station_boundaries,
    summarize_regimes,
)


SMALL_CONFIG = ScanConfig(
    center_q1=70.0,
    center_q2=20.0,
    node_duration_s=5.0,
    startup_s=1.0,
    step_ratio=1.6,
    bisection_rounds=1,
    max_expand_steps=3,
    max_stations=3,
    max_nodes=40,
)


def noiseless_scan(config: ScanConfig):
    simulator = ChipSimulator(
        config=ChipModelConfig(
            jetting_center_q1=config.center_q1,
            jetting_center_q2=config.center_q2,
        ),
        noise=False,
    )
    return run_offline_scan(config, simulator=simulator)


@pytest.fixture(scope="module")
def result():
    return noiseless_scan(SMALL_CONFIG)


def test_default_target_is_jetting_not_a_required_dripping_window() -> None:
    config = ScanConfig()
    assert config.target_regime == "jetting"
    scan = run_offline_scan(replace(config, max_nodes=1))
    assert scan.center_node.regime is FlowRegime.JETTING
    assert scan.metadata["scan_target"] == "jetting"
    assert scan.metadata["model_kind"] == "jetting_prior"
    assert scan.metadata["physical_boundaries_verified"] is False


def test_center_node_starts_at_chosen_anchor(result) -> None:
    assert result.center_node is result.nodes[0]
    assert result.center_node.q1 == SMALL_CONFIG.center_q1
    assert result.center_node.q2 == SMALL_CONFIG.center_q2
    assert result.center_node.regime is FlowRegime.JETTING
    assert result.center_node.purpose == "station_base#0"
    assert result.center_node.started_s == SMALL_CONFIG.startup_s
    assert result.center_node.features.extra["evolution_stage"] == "near_anchor"


def test_scan_produces_nodes_and_both_jetting_bracket_sides(result) -> None:
    assert 6 <= len(result.nodes) <= SMALL_CONFIG.max_nodes
    kinds = {bracket.kind for bracket in result.brackets}
    assert kinds == {"jetting_lower", "jetting_upper"}
    assert {node.regime for node in result.nodes} == {FlowRegime.JETTING, FlowRegime.THREADING}


def test_nodes_are_unique_and_ordered(result) -> None:
    assert [node.index for node in result.nodes] == list(range(1, len(result.nodes) + 1))
    points = [(node.q1, node.q2) for node in result.nodes]
    assert len(set(points)) == len(points)
    started = [node.started_s for node in result.nodes]
    assert started == sorted(started)
    for previous, current in zip(result.nodes, result.nodes[1:]):
        assert current.started_s == pytest.approx(previous.elapsed_s + SMALL_CONFIG.startup_s)


def test_planned_duration_counts_every_startup_including_first(result) -> None:
    per_node = SMALL_CONFIG.startup_s + SMALL_CONFIG.node_duration_s
    assert result.total_duration_s == pytest.approx(len(result.nodes) * per_node)
    for offset, node in enumerate(result.nodes):
        assert node.started_s == pytest.approx(offset * per_node + SMALL_CONFIG.startup_s)
        assert node.duration_s == SMALL_CONFIG.node_duration_s
        assert node.features.window_s == SMALL_CONFIG.node_duration_s


def test_nodes_stay_in_configured_flow_and_safety_domain(result) -> None:
    stations = {node.q1 for node in result.nodes}
    assert 1 < len(stations) <= SMALL_CONFIG.max_stations
    for node in result.nodes:
        assert SMALL_CONFIG.q1_min <= node.q1 <= SMALL_CONFIG.q1_max
        assert SMALL_CONFIG.q2_min <= node.q2 <= SMALL_CONFIG.q2_max
        assert node.q1 >= SMALL_CONFIG.safety_q1_over_q2 * node.q2
    assert result.total_duration_s <= SMALL_CONFIG.max_duration_s


def test_station_boundaries_are_observed_intervals_not_exact_thresholds(result) -> None:
    table = station_boundaries(result)
    assert table
    assert {kind for entries in table.values() for kind in entries} == {"jetting_lower", "jetting_upper"}
    for station_q1, entries in table.items():
        for kind, bracket in entries.items():
            assert bracket.station_q1 == station_q1 == bracket.inside.q1 == bracket.outside.q1
            assert bracket.inside in result.nodes and bracket.outside in result.nodes
            assert bracket.inside.regime is FlowRegime.JETTING
            assert bracket.outside.regime is FlowRegime.THREADING
            assert bracket.resolution == "bracketed"
            low, high = bracket.q2_interval
            assert low < bracket.boundary_q2 < high
            assert bracket.boundary_q2 == pytest.approx(math.sqrt(low * high))
            assert bracket.relative_width == pytest.approx(high / low)
            assert 1.0 < bracket.relative_width <= SMALL_CONFIG.step_ratio + 1e-9
            if kind == "jetting_lower":
                assert bracket.outside.q2 < bracket.inside.q2
            else:
                assert bracket.outside.q2 > bracket.inside.q2


def test_summary_counts_exactly_the_observed_regimes(result) -> None:
    counts = summarize_regimes(result)
    assert counts == Counter(node.regime for node in result.nodes)
    assert sum(counts.values()) == len(result.nodes)
    assert set(counts) == {FlowRegime.JETTING, FlowRegime.THREADING}
    assert all(value > 0 for value in counts.values())


@pytest.mark.parametrize("max_nodes", [1, 4])
def test_node_budget_stops_with_real_uncovered_work(max_nodes: int) -> None:
    config = replace(SMALL_CONFIG, max_nodes=max_nodes)
    scan = noiseless_scan(config)
    assert len(scan.nodes) == max_nodes
    assert scan.stopped_reason
    assert scan.total_duration_s == max_nodes * (config.startup_s + config.node_duration_s)
    statuses = scan.metadata["station_status"]
    assert statuses
    assert any("budget_exhausted" in (item["jetting_lower"], item["jetting_upper"]) for item in statuses)


@pytest.mark.parametrize("budget", [6.0, 17.0, 18.0])
def test_time_budget_includes_all_startups_and_never_overruns(budget: float) -> None:
    config = replace(SMALL_CONFIG, max_duration_s=budget)
    scan = noiseless_scan(config)
    per_node = config.startup_s + config.node_duration_s
    assert len(scan.nodes) == int(budget // per_node)
    assert scan.total_duration_s == len(scan.nodes) * per_node <= budget
    assert scan.total_duration_s + per_node > budget
    assert scan.stopped_reason


def test_safe_domain_can_censor_both_sides_without_inventing_boundaries() -> None:
    config = replace(SMALL_CONFIG, max_stations=1, q2_min=19.0, q2_max=21.0)
    scan = noiseless_scan(config)
    assert {node.q2 for node in scan.nodes} == {19.0, 20.0, 21.0}
    assert all(node.regime is FlowRegime.JETTING for node in scan.nodes)
    assert scan.brackets == []
    assert station_boundaries(scan) == {}
    assert not scan.stopped_reason
    statuses = scan.metadata["station_status"]
    assert len(statuses) == 1
    assert statuses[0]["jetting_lower"] == "censored_by_safe_domain"
    assert statuses[0]["jetting_upper"] == "censored_by_safe_domain"


def test_unknown_observations_do_not_become_outside_boundary_evidence() -> None:
    sim = ChipSimulator(noise=False)

    def unknown_measurement(q1: float, q2: float, window_s: float):
        return replace(sim.features(q1, q2, window_s=window_s), sample_count=0)

    config = replace(SMALL_CONFIG, max_stations=1)
    scan = SpiderScanner(unknown_measurement, config=config).run()
    assert len(scan.nodes) > 1
    assert all(node.regime is FlowRegime.UNKNOWN for node in scan.nodes)
    assert scan.brackets == []
    assert scan.metadata["unresolved_intervals"]
    assert all(item["reason"] == "unknown_observation" for item in scan.metadata["unresolved_intervals"])


def test_explicit_dripping_target_reconstructs_only_optional_ratio_window_prior() -> None:
    """This checks search against a known synthetic prior, not real-chip topology."""
    config = replace(
        SMALL_CONFIG,
        target_regime="dripping",
        bisection_rounds=3,
        max_expand_steps=4,
        max_stations=2,
        max_nodes=60,
    )
    prior = ChipModelConfig(operating_mode="ratio_window", ratio_low=2.4, ratio_high=4.4, ratio_thread=6.0)
    scan = run_offline_scan(config, simulator=ChipSimulator(config=prior, noise=False))
    assert scan.center_node.regime is FlowRegime.DRIPPING
    assert scan.metadata["scan_target"] == "dripping"
    assert scan.metadata["model_kind"] == "ratio_window_prior"
    assert scan.metadata["physical_boundaries_verified"] is False
    table = station_boundaries(scan)
    assert len(table) == config.max_stations
    for station_q1, entries in table.items():
        assert set(entries) == {"droplet_lower", "droplet_upper"}
        for kind, bracket in entries.items():
            assert bracket.inside.regime is FlowRegime.DRIPPING
            if kind == "droplet_lower":
                ratio, outside = prior.ratio_high, FlowRegime.JETTING
                assert bracket.outside.q2 < bracket.inside.q2
            else:
                ratio, outside = prior.ratio_low, FlowRegime.TUBING
                assert bracket.outside.q2 > bracket.inside.q2
            assert bracket.outside.regime is outside
            low, high = bracket.q2_interval
            assert low <= station_q1 / ratio <= high
            assert bracket.boundary_ratio == pytest.approx(ratio, rel=0.08)
            assert bracket.relative_width <= config.min_ratio_step


def test_scanner_rejects_bad_center_before_measuring() -> None:
    calls = []
    with pytest.raises(ValueError, match="安全约束"):
        SpiderScanner(lambda *args: calls.append(args), config=ScanConfig(center_q1=10.0, center_q2=20.0))
    assert calls == []
