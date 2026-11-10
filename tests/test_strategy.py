"""蜘蛛式扫描策略测试：覆盖范围、边界夹逼与时间预算。"""

from __future__ import annotations

import pytest

from mcs_date.experiment import (
    FlowRegime,
    ScanConfig,
    SpiderScanner,
    run_offline_scan,
    station_boundaries,
    summarize_regimes,
)
from mcs_date.experiment.models import regime_group


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


@pytest.fixture(scope="module")
def result():
    return run_offline_scan(SMALL_CONFIG)


def test_center_node_starts_at_working_point(result) -> None:
    assert result.center_node.q1 == SMALL_CONFIG.center_q1
    assert result.center_node.q2 == SMALL_CONFIG.center_q2
    assert result.center_node.regime is FlowRegime.DRIPPING


def test_scan_produces_nodes_and_brackets(result) -> None:
    assert len(result.nodes) >= 6
    assert result.brackets, "扫描至少应夹逼出一个液滴窗口边界"
    assert len(result.nodes) <= SMALL_CONFIG.max_nodes


def test_nodes_are_unique_and_ordered(result) -> None:
    indices = [node.index for node in result.nodes]
    assert indices == list(range(1, len(indices) + 1))
    started = [node.started_s for node in result.nodes]
    assert started == sorted(started)


def test_planned_duration_matches_node_budget(result) -> None:
    per_node = SMALL_CONFIG.startup_s + SMALL_CONFIG.node_duration_s
    expected = (len(result.nodes) - 1) * per_node + SMALL_CONFIG.node_duration_s
    assert result.total_duration_s == pytest.approx(expected)


def test_station_boundaries_stay_inside_brackets(result) -> None:
    table = station_boundaries(result)
    assert table
    for station_q1, entries in table.items():
        assert station_q1 > 0
        for kind, bracket in entries.items():
            assert kind in {"droplet_upper", "droplet_lower"}
            low = min(bracket.inside.q2, bracket.outside.q2)
            high = max(bracket.inside.q2, bracket.outside.q2)
            assert low <= bracket.boundary_q2 <= high
            assert 1.0 <= bracket.relative_width <= SMALL_CONFIG.step_ratio + 1e-9


def test_lower_boundary_separates_droplets_from_jets(result) -> None:
    for bracket in result.boundaries("droplet_lower"):
        assert bracket.inside.regime is FlowRegime.DRIPPING
        assert bracket.outside.regime is FlowRegime.JETTING


def test_upper_boundary_separates_droplets_from_plugs(result) -> None:
    for bracket in result.boundaries("droplet_upper"):
        assert bracket.inside.regime is FlowRegime.DRIPPING
        assert bracket.outside.regime is FlowRegime.TUBING


def test_boundary_ratios_match_window(result) -> None:
    """在节点预算更大的一次扫描里，边界油水比应收敛到窗口参数附近。"""

    config = ScanConfig(
        center_q1=70.0,
        center_q2=20.0,
        node_duration_s=2.0,
        startup_s=0.5,
        bisection_rounds=2,
        max_expand_steps=4,
        max_stations=2,
        max_nodes=60,
    )
    scan = run_offline_scan(config)
    table = station_boundaries(scan)
    assert table
    expected = {
        "droplet_lower": scan.config and ChipWindow.lower(),
        "droplet_upper": ChipWindow.upper(),
    }
    for entries in table.values():
        for kind, bracket in entries.items():
            assert bracket.boundary_ratio == pytest.approx(expected[kind], rel=0.12)


class ChipWindow:
    """与 ``ChipSimulator()`` 默认窗口保持一致的辅助常量。"""

    @staticmethod
    def lower() -> float:
        from mcs_date.experiment import ChipSimulator

        return ChipSimulator().config.ratio_high

    @staticmethod
    def upper() -> float:
        from mcs_date.experiment import ChipSimulator

        return ChipSimulator().config.ratio_low


def test_summary_counts_all_regimes(result) -> None:
    counts = summarize_regimes(result)
    assert sum(counts.values()) == len(result.nodes)
    assert all(value > 0 for value in counts.values())
    assert all(regime is not FlowRegime.UNKNOWN for regime in counts)


def test_scanner_rejects_bad_center() -> None:
    with pytest.raises(ValueError):
        SpiderScanner(lambda *_: None, config=ScanConfig(center_q1=10.0, center_q2=20.0)).run()
