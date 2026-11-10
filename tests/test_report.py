"""扫描结果序列化 / 反序列化 / CSV 导出测试。"""

from __future__ import annotations

import csv
import json

import pytest

from mcs_date.experiment import (
    ScanConfig,
    load_result,
    run_offline_scan,
    save_nodes_csv,
    save_result,
)


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
    assert raw["schema_version"] == 1
    assert raw["metadata"]["chip"] == "test-chip"

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
        assert restored.features.stream_coverage == pytest.approx(
            original.features.stream_coverage
        )


def test_nodes_csv_has_one_row_per_node(result, tmp_path) -> None:
    path = tmp_path / "nodes.csv"
    save_nodes_csv(result, path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.nodes)
    assert float(rows[0]["q1_ul_min"]) == pytest.approx(result.nodes[0].q1, abs=1e-3)
    assert rows[0]["regime"] == result.nodes[0].regime.value
    assert rows[0]["purpose"] == result.nodes[0].purpose
