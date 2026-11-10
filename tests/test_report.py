"""扫描结果序列化 / 反序列化 / CSV 导出测试。"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import replace

import pytest

from mcs_date.experiment import (
    BoundaryBracket,
    ScanConfig,
    load_result,
    result_from_dict,
    result_to_dict,
    run_offline_scan,
    save_nodes_csv,
    save_result,
)
from mcs_date.experiment.phase import assess_formation


SMALL = ScanConfig(
    center_q1=80.0,
    center_q2=20.0,
    node_duration_s=2.0,
    startup_s=0.5,
    bisection_rounds=1,
    max_expand_steps=2,
    max_stations=2,
    max_nodes=16,
)


@pytest.fixture(scope="module")
def result():
    return run_offline_scan(SMALL)


def test_json_roundtrip_keeps_nodes_and_brackets(result, tmp_path) -> None:
    path = tmp_path / "result.json"
    save_result(result, path, metadata={"chip": "test-chip", "operator": "pytest"})
    assert path.exists()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    assert raw["metadata"]["chip"] == "test-chip"
    for key, value in result.metadata.items():
        assert raw["metadata"][key] == value
    assert raw["metadata"]["data_origin"] == "simulation"
    assert raw["metadata"]["physical_boundaries_verified"] is False

    loaded = load_result(path)
    assert loaded.config == result.config
    assert len(loaded.nodes) == len(result.nodes)
    assert len(loaded.brackets) == len(result.brackets)
    assert loaded.center_node.q1 == result.center_node.q1
    assert loaded.center_node.regime is result.center_node.regime
    assert loaded.total_duration_s == pytest.approx(result.total_duration_s)
    for original, restored in zip(result.nodes, loaded.nodes):
        assert restored.index == original.index
        assert restored.q1 == pytest.approx(original.q1)
        assert restored.q2 == pytest.approx(original.q2)
        assert restored.regime is original.regime
        assert restored.features == original.features
    for original, restored in zip(result.brackets, loaded.brackets):
        assert restored.resolution == original.resolution
        assert restored.q2_interval == original.q2_interval
        assert restored.inside is loaded.nodes[result.nodes.index(original.inside)]
        assert restored.outside is loaded.nodes[result.nodes.index(original.outside)]
    for node, serialized in zip(result.nodes, raw["nodes"]):
        assert serialized["source"] == node.features.source
        assert serialized["formation"] == assess_formation(node.features, result.thresholds)


def test_nodes_csv_has_one_row_per_node(result, tmp_path) -> None:
    path = tmp_path / "nodes.csv"
    save_nodes_csv(result, path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.nodes)
    assert float(rows[0]["q1_ul_min"]) == pytest.approx(result.nodes[0].q1, abs=1e-3)
    assert rows[0]["regime"] == result.nodes[0].regime.value
    assert rows[0]["purpose"] == result.nodes[0].purpose
    for row, node in zip(rows, result.nodes):
        formation = assess_formation(node.features, result.thresholds)
        assert row["data_origin"] == "simulation"
        assert row["source"] == node.features.source
        for key in ("droplet_events", "droplets_observed", "uniformity_candidate",
                    "best_operating_point_verified"):
            assert row[key] == ("" if formation[key] is None else str(formation[key]))


def test_metadata_overrides_merge_without_mutating_original(result, tmp_path) -> None:
    original = copy.deepcopy(result.metadata)
    exported = result_to_dict(result, metadata={"operator": "pytest", "status": "reviewed"})
    assert exported["metadata"] == {**original, "operator": "pytest", "status": "reviewed"}
    assert result.metadata == original
    assert result_to_dict(result, metadata={})["metadata"] == original
    path = tmp_path / "merged.json"
    save_result(result, path, metadata={"chip": "test-chip"})
    assert load_result(path).metadata == {**original, "chip": "test-chip"}


@pytest.mark.parametrize("key", [
    "data_origin", "model_kind", "model_config", "random_seed",
    "physical_boundaries_verified", "calibration",
])
def test_reserved_provenance_rejects_conflicting_overrides(result, tmp_path, key) -> None:
    original = replace(result, metadata={**result.metadata, key: {"verified": False}})
    with pytest.raises(ValueError, match=key):
        save_result(original, tmp_path / "conflict.json", metadata={key: {"verified": True}})
    assert result_to_dict(original, metadata={key: {"verified": False}})["metadata"][key] == {
        "verified": False,
    }
    assert not (tmp_path / "conflict.json").exists()


def test_reserved_provenance_distinguishes_false_from_zero(result) -> None:
    original = replace(result, metadata={"physical_boundaries_verified": False})
    with pytest.raises(ValueError, match="physical_boundaries_verified"):
        result_to_dict(original, metadata={"physical_boundaries_verified": 0})


def _with_uncertain_bracket(result):
    inside = result.nodes[0]
    outside = next(node for node in result.nodes if node.q2 != inside.q2)
    bracket = BoundaryBracket(
        station_q1=inside.q1, kind="droplet_lower", direction="lower",
        inside=inside, outside=outside, boundary_q2=(inside.q2 + outside.q2) / 2,
        relative_width=max(inside.q2, outside.q2) / min(inside.q2, outside.q2),
        at_domain_edge=True, resolution="uncertain",
    )
    return replace(result, brackets=[bracket])


def test_uncertain_bracket_roundtrip(result, tmp_path) -> None:
    original = _with_uncertain_bracket(result)
    path = tmp_path / "uncertain.json"
    save_result(original, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["brackets"][0]["resolution"] == "uncertain"
    assert raw["brackets"][0]["q2_interval"] == list(original.brackets[0].q2_interval)
    restored = load_result(path)
    assert restored.brackets[0] == original.brackets[0]
    assert restored.brackets[0].inside is restored.center_node


def test_schema1_read_marks_uncertain_legacy_origin(result, tmp_path) -> None:
    original = _with_uncertain_bracket(result)
    raw = result_to_dict(original)
    raw["schema_version"] = 1
    raw["metadata"] = {"data_origin": "simulation", "chip": "legacy-chip"}
    for node in raw["nodes"]:
        node.pop("source")
        node.pop("formation")
        for key in ("source", "droplet_events", "junction_visible", "breakup_distance_h",
                    "observation_extent_h"):
            node["features"].pop(key)
    for bracket in raw["brackets"]:
        bracket.pop("resolution")
        bracket.pop("q2_interval")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.warns(UserWarning, match="timing and source are uncertain"):
        restored = load_result(path)
    assert restored.metadata == {"data_origin": "unspecified_legacy", "chip": "legacy-chip"}
    assert all(node.features.source == "unspecified_legacy" for node in restored.nodes)
    assert restored.center_node.started_s == result.center_node.started_s
    assert restored.brackets[0].resolution == "uncertain"
    assert restored.brackets[0].q2_interval == original.brackets[0].q2_interval
    # Re-saving must not silently upgrade provenance to a physical or simulated source.
    upgraded = result_to_dict(restored)
    assert upgraded["schema_version"] == 2
    assert upgraded["metadata"]["data_origin"] == "unspecified_legacy"
    assert result_from_dict(upgraded).metadata == restored.metadata


@pytest.mark.parametrize("version", [None, 0, 3, -1, True, 1.0, 2.0, "2"])
def test_rejects_unsupported_or_noninteger_versions(result, version) -> None:
    raw = result_to_dict(result)
    raw["schema_version"] = version
    with pytest.raises(ValueError, match="版本"):
        result_from_dict(raw)


@pytest.mark.parametrize("invalid_id", [-1, True, 1.5, "1"])
def test_rejects_invalid_node_ids(result, invalid_id) -> None:
    raw = result_to_dict(result)
    raw["nodes"][0]["index"] = invalid_id
    with pytest.raises(ValueError, match="ID"):
        result_from_dict(raw)


def test_rejects_duplicate_node_ids_on_read_and_write(result) -> None:
    raw = result_to_dict(result)
    raw["nodes"].append(copy.deepcopy(raw["nodes"][0]))
    with pytest.raises(ValueError, match="重复节点 ID"):
        result_from_dict(raw)
    with pytest.raises(ValueError, match="重复节点 ID"):
        result_to_dict(replace(result, nodes=[*result.nodes, result.nodes[0]]))


@pytest.mark.parametrize("reference", ["center_index", "inside_index", "outside_index"])
def test_rejects_missing_node_references(result, reference) -> None:
    raw = result_to_dict(_with_uncertain_bracket(result))
    target = raw if reference == "center_index" else raw["brackets"][0]
    target[reference] = max(node.index for node in result.nodes) + 1
    with pytest.raises(ValueError, match="节点引用不存在"):
        result_from_dict(raw)


def test_rejects_self_and_duplicate_bracket_references(result) -> None:
    raw = result_to_dict(_with_uncertain_bracket(result))
    raw["brackets"].append(copy.deepcopy(raw["brackets"][0]))
    with pytest.raises(ValueError, match="重复边界引用"):
        result_from_dict(raw)
    raw["brackets"].pop()
    bracket = raw["brackets"][0]
    bracket["outside_index"] = bracket["inside_index"]
    q2 = raw["nodes"][0]["q2"]
    bracket["q2_interval"] = [q2, q2]
    with pytest.raises(ValueError, match="同一节点"):
        result_from_dict(raw)


def test_rejects_inconsistent_interval_and_resolution(result) -> None:
    raw = result_to_dict(_with_uncertain_bracket(result))
    raw["brackets"][0]["q2_interval"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="q2_interval"):
        result_from_dict(raw)
    raw = result_to_dict(_with_uncertain_bracket(result))
    raw["brackets"][0]["resolution"] = "exact"
    with pytest.raises(ValueError, match="边界状态"):
        result_from_dict(raw)


@pytest.mark.parametrize("field", ["source", "formation"])
def test_rejects_inconsistent_derived_node_fields(result, field) -> None:
    raw = result_to_dict(result)
    raw["nodes"][0][field] = "not-the-observation"
    with pytest.raises(ValueError, match=field):
        result_from_dict(raw)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_json_save_rejects_nonfinite_metadata(result, tmp_path, number) -> None:
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        save_result(result, path, metadata={"invalid": number})
    assert not path.exists()


def test_json_save_rejects_nonfinite_feature_and_does_not_truncate(result, tmp_path) -> None:
    node = replace(result.nodes[0], features=replace(result.nodes[0].features, stream_coverage=float("nan")))
    nodes = [node, *result.nodes[1:]]
    invalid = replace(result, nodes=nodes, center_node=node, brackets=[])
    path = tmp_path / "existing.json"
    path.write_text("previous valid report", encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float values"):
        save_result(invalid, path)
    assert path.read_text(encoding="utf-8") == "previous valid report"


def test_json_read_rejects_nonstandard_nan(result, tmp_path) -> None:
    raw = result_to_dict(result)
    raw["metadata"]["invalid"] = float("nan")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="非有限数值"):
        load_result(path)
