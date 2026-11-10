"""四射线小规模试验方案（pilot）的离线回归测试。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mcs_date.__main__ import main
from mcs_date.experiment.phase import FlowRegime
from mcs_date.experiment.pilot import (
    PilotConfig,
    build_pilot_points,
    pilot_scan_config,
    run_offline_pilot,
)


def test_default_plan_has_anchor_four_rays_and_revisit() -> None:
    config = PilotConfig()
    points, status = build_pilot_points(config)
    assert len(points) == 14 == config.planned_nodes
    assert points[0].ray == "anchor" and points[0].q1 == pytest.approx(90.0)
    assert points[-1].ray == "anchor_revisit"
    assert [p.ray for p in points].count("anchor") == 1
    for ray in ("R_down", "R_up", "S_down", "S_up"):
        ray_points = [p for p in points if p.ray == ray]
        assert len(ray_points) == 3
        assert status[ray]["censored_by"] is None
    # S 射线固定 R，R 射线固定 S（锚点值）。
    for point in points:
        if point.ray in {"S_down", "S_up"}:
            assert point.q1 / point.q2 == pytest.approx(4.5, rel=1e-9)
        if point.ray in {"R_down", "R_up"}:
            assert point.q1 + point.q2 == pytest.approx(110.0, rel=1e-9)
        assert point.q1 >= 2.0 * point.q2


def test_step_controls_geometric_spacing() -> None:
    points, _ = build_pilot_points(PilotConfig(step_fraction=0.25))
    r_down = [p for p in points if p.ray == "R_down"]
    ratios = [p.oil_water_ratio for p in r_down]
    assert ratios[1] / ratios[0] == pytest.approx(1 / 1.25, rel=1e-9)
    assert ratios[2] / ratios[1] == pytest.approx(1 / 1.25, rel=1e-9)


def test_unsafe_anchor_rejected() -> None:
    with pytest.raises(ValueError, match="安全约束"):
        PilotConfig(anchor_q1=30.0, anchor_q2=20.0).validate()
    with pytest.raises(ValueError, match="step_fraction"):
        PilotConfig(step_fraction=0.9).validate()
    with pytest.raises(ValueError, match="整数"):
        PilotConfig(steps_per_ray=2.5).validate()  # type: ignore[arg-type]


def test_domain_edge_is_censored_not_silently_dropped() -> None:
    config = PilotConfig(anchor_q1=90.0, anchor_q2=15.0, q2_max=16.0)
    points, status = build_pilot_points(config)
    assert status["R_down"]["censored_by"] == "q2_max"
    assert status["S_up"]["censored_by"] == "q2_max"
    assert status["R_up"]["censored_by"] is None
    assert len(points) < 14
    assert all(p.q2 <= 16.0 for p in points)


def test_scan_config_maps_budget_and_target() -> None:
    config = PilotConfig(steps_per_ray=2)
    scan = pilot_scan_config(config, config.planned_nodes)
    planned = config.planned_nodes
    assert scan.target_regime == "jetting"
    assert scan.max_nodes == planned
    assert scan.max_duration_s == pytest.approx(
        planned * (config.startup_s + config.node_duration_s) + 60.0
    )
    assert scan.max_duration_s > config.estimated_duration_s


def test_offline_pilot_records_evolution_and_crossings() -> None:
    result = run_offline_pilot(PilotConfig())
    assert result.stopped_reason == ""
    assert len(result.nodes) == result.metadata["planned_nodes"] == 14
    assert result.metadata["data_origin"] == "simulation"
    assert result.brackets == []

    by_purpose = {node.purpose: node for node in result.nodes}
    assert by_purpose["pilot:anchor#0"].features.extra["evolution_stage"] == "near_anchor"
    assert by_purpose["pilot:anchor_revisit#0"].q1 == pytest.approx(90.0)
    stages = {node.features.extra["evolution_stage"] for node in result.nodes}
    assert "size_change" in stages
    assert "unbroken_in_observation_window" in stages  # 射流→束流射线末端

    crossings = result.metadata["ray_crossings"]
    assert crossings, "默认 25% 步幅应在 R 射线命中射流→束流转换"
    nodes_by_index = {node.index: node for node in result.nodes}
    for crossing in crossings:
        assert crossing["resolution"] == "bracketed"
        inside = nodes_by_index[crossing["inside_index"]]
        outside = nodes_by_index[crossing["outside_index"]]
        assert inside.regime is FlowRegime.JETTING
        assert outside.regime is FlowRegime.THREADING
        if crossing["ray"] == "R_down":
            assert inside.q2 < outside.q2
        else:
            assert inside.q2 > outside.q2
        assert crossing["midpoint_q2"] == pytest.approx(
            math.sqrt(crossing["inside_q2"] * crossing["outside_q2"])
        )
    for node in result.nodes:
        if node.regime is FlowRegime.THREADING:
            assert node.features.droplet_events == 0


def test_pilot_outputs_are_simulation_labelled(tmp_path: Path) -> None:
    exit_code = main([
        "pilot", "--out", str(tmp_path), "--no-animate", "--quiet",
        "--anchor-q1", "90", "--anchor-q2", "20",
    ])
    assert exit_code == 0
    payload = json.loads((tmp_path / "pilot_result.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["data_origin"] == "simulation"
    assert payload["metadata"]["physical_boundaries_verified"] is False
    rows = (tmp_path / "pilot_plan.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 14
    assert rows[0].split(",")[0] == "ray"
    assert "not_run" not in "\n".join(rows)


def test_pilot_respects_time_budget(tmp_path: Path) -> None:
    exit_code = main([
        "pilot", "--out", str(tmp_path), "--no-animate", "--quiet", "--max-hours", "0.2",
    ])
    assert exit_code == 1
