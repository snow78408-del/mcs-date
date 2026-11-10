"""扫描结果的持久化（JSON / CSV）。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import BoundaryBracket, NodeRecord, ScanConfig, ScanResult
from .phase import FlowRegime, NodeFeatures, RegimeThresholds

SCHEMA_VERSION = 1


def _node_to_dict(node: NodeRecord) -> dict[str, Any]:
    return {
        "index": node.index,
        "q1": node.q1,
        "q2": node.q2,
        "regime": node.regime.value,
        "confidence": node.confidence,
        "started_s": node.started_s,
        "duration_s": node.duration_s,
        "purpose": node.purpose,
        "features": asdict(node.features),
    }


def _node_from_dict(data: dict[str, Any]) -> NodeRecord:
    features = NodeFeatures(**data["features"])
    return NodeRecord(
        index=int(data["index"]),
        q1=float(data["q1"]),
        q2=float(data["q2"]),
        regime=FlowRegime(data["regime"]),
        confidence=float(data["confidence"]),
        features=features,
        started_s=float(data["started_s"]),
        duration_s=float(data["duration_s"]),
        purpose=str(data.get("purpose", "")),
    )


def _bracket_to_dict(bracket: BoundaryBracket) -> dict[str, Any]:
    return {
        "station_q1": bracket.station_q1,
        "kind": bracket.kind,
        "direction": bracket.direction,
        "boundary_q2": bracket.boundary_q2,
        "relative_width": bracket.relative_width,
        "at_domain_edge": bracket.at_domain_edge,
        "inside_index": bracket.inside.index,
        "outside_index": bracket.outside.index,
    }


def result_to_dict(result: ScanResult, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata or dict(result.metadata),
        "config": asdict(result.config),
        "thresholds": asdict(result.thresholds),
        "center_index": result.center_node.index,
        "stopped_reason": result.stopped_reason,
        "nodes": [_node_to_dict(node) for node in result.nodes],
        "brackets": [_bracket_to_dict(bracket) for bracket in result.brackets],
    }


def result_from_dict(data: dict[str, Any]) -> ScanResult:
    version = int(data.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(f"不支持的扫描结果版本: {version}")
    nodes = [_node_from_dict(item) for item in data["nodes"]]
    by_index = {node.index: node for node in nodes}
    brackets: list[BoundaryBracket] = []
    for item in data.get("brackets", []):
        brackets.append(
            BoundaryBracket(
                station_q1=float(item["station_q1"]),
                kind=str(item["kind"]),
                direction=str(item["direction"]),
                inside=by_index[int(item["inside_index"])],
                outside=by_index[int(item["outside_index"])],
                boundary_q2=float(item["boundary_q2"]),
                relative_width=float(item["relative_width"]),
                at_domain_edge=bool(item.get("at_domain_edge", False)),
            )
        )
    center_index = int(data["center_index"])
    return ScanResult(
        config=ScanConfig(**data["config"]),
        center_node=by_index[center_index],
        nodes=nodes,
        brackets=brackets,
        thresholds=RegimeThresholds(**data["thresholds"]),
        metadata=dict(data.get("metadata", {})),
        stopped_reason=str(data.get("stopped_reason", "")),
    )


def save_result(result: ScanResult, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result_to_dict(result, metadata=metadata)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_result(path: str | Path) -> ScanResult:
    return result_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_nodes_csv(result: ScanResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index", "q1_ul_min", "q2_ul_min", "flow_ratio", "regime",
                "confidence", "droplet_rate_hz", "mean_diameter_ratio",
                "stream_coverage", "max_plug_length_ratio", "started_s",
                "duration_s", "purpose",
            ]
        )
        for node in result.nodes:
            writer.writerow(
                [
                    node.index, f"{node.q1:.4f}", f"{node.q2:.4f}",
                    f"{node.flow_ratio:.6f}", node.regime.value,
                    f"{node.confidence:.4f}",
                    f"{node.features.droplet_rate_hz:.4f}",
                    f"{node.features.mean_diameter_ratio:.4f}",
                    f"{node.features.stream_coverage:.4f}",
                    f"{node.features.max_plug_length_ratio:.4f}",
                    f"{node.started_s:.1f}", f"{node.duration_s:.1f}", node.purpose,
                ]
            )
    return path
