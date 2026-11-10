"""扫描结果的持久化（JSON / CSV），保留观测来源及区间不确定性。"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import BoundaryBracket, NodeRecord, ScanConfig, ScanResult
from .phase import FlowRegime, NodeFeatures, RegimeThresholds, assess_formation, classify_regime

SCHEMA_VERSION = 2
_RESERVED_PROVENANCE_KEYS = frozenset({
    "data_origin", "model_kind", "model_config", "random_seed",
    "physical_boundaries_verified", "calibration",
})


def _node_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"节点 ID 必须是非负整数: {value!r}")
    return value


def _node_map(nodes: list[NodeRecord]) -> dict[int, NodeRecord]:
    by_index: dict[int, NodeRecord] = {}
    for node in nodes:
        index = _node_id(node.index)
        if index in by_index:
            raise ValueError(f"重复节点 ID: {index}")
        by_index[index] = node
    return by_index


def _reference(by_index: dict[int, NodeRecord], value: Any) -> NodeRecord:
    index = _node_id(value)
    if index not in by_index:
        raise ValueError(f"节点引用不存在: {index}")
    return by_index[index]


def _validate_references(result: ScanResult) -> None:
    by_index = _node_map(result.nodes)
    if _reference(by_index, result.center_node.index) != result.center_node:
        raise ValueError("中心节点引用与节点记录不一致")
    seen = set()
    for bracket in result.brackets:
        for node in (bracket.inside, bracket.outside):
            if _reference(by_index, node.index) != node:
                raise ValueError("边界节点引用与节点记录不一致")
        if bracket.inside.index == bracket.outside.index:
            raise ValueError("边界两端不能引用同一节点")
        key = (bracket.station_q1, bracket.kind, bracket.direction,
               tuple(sorted((bracket.inside.index, bracket.outside.index))))
        if key in seen:
            raise ValueError("重复边界引用")
        seen.add(key)
        if bracket.resolution not in {"bracketed", "uncertain"}:
            raise ValueError(f"不支持的边界状态: {bracket.resolution!r}")
        if bracket.resolution == "bracketed":
            target = FlowRegime.JETTING if bracket.kind.startswith("jetting_") else FlowRegime.DRIPPING
            if (bracket.inside.regime is not target or bracket.outside.regime in {target, FlowRegime.UNKNOWN}
                    or bracket.inside.q1 != bracket.station_q1 or bracket.outside.q1 != bracket.station_q1):
                raise ValueError("有效边界端点必须为同站位的目标流型和已知非目标流型")


def _merged_metadata(result: ScanResult, overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(result.metadata)
    for key, value in (overrides or {}).items():
        if key in _RESERVED_PROVENANCE_KEYS and key in merged:
            # JSON比较区分 False/0 等值但含义不同的来源字段，也不受字典顺序影响。
            before = json.dumps(merged[key], sort_keys=True, allow_nan=False)
            after = json.dumps(value, sort_keys=True, allow_nan=False)
            if before != after:
                raise ValueError(f"不能覆盖保留的来源元数据: {key}")
        merged[key] = value
    return merged


def _node_to_dict(node: NodeRecord, thresholds: RegimeThresholds) -> dict[str, Any]:
    return {
        "index": node.index,
        "q1": node.q1,
        "q2": node.q2,
        "regime": node.regime.value,
        "confidence": node.confidence,
        "started_s": node.started_s,
        "duration_s": node.duration_s,
        "purpose": node.purpose,
        "source": node.features.source,
        "formation": assess_formation(node.features, thresholds),
        "features": asdict(node.features),
    }


def _node_from_dict(data: dict[str, Any], *, legacy: bool,
                    thresholds: RegimeThresholds) -> NodeRecord:
    feature_data = dict(data["features"])
    if legacy:
        feature_data["source"] = "unspecified_legacy"
    features = NodeFeatures(**feature_data)
    if not legacy:
        if data["source"] != features.source:
            raise ValueError("节点 source 与观测特征来源不一致")
        if data["formation"] != assess_formation(features, thresholds):
            raise ValueError("节点 formation 与观测特征/阈值不一致")
    return NodeRecord(
        index=_node_id(data["index"]),
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
        "resolution": bracket.resolution,
        "q2_interval": list(bracket.q2_interval),
        "inside_index": bracket.inside.index,
        "outside_index": bracket.outside.index,
    }


def result_to_dict(result: ScanResult, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    _validate_references(result)
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": _merged_metadata(result, metadata),
        "config": asdict(result.config),
        "thresholds": asdict(result.thresholds),
        "center_index": result.center_node.index,
        "stopped_reason": result.stopped_reason,
        "nodes": [_node_to_dict(node, result.thresholds) for node in result.nodes],
        "brackets": [_bracket_to_dict(bracket) for bracket in result.brackets],
    }


def result_from_dict(data: dict[str, Any]) -> ScanResult:
    version = data.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version not in {1, SCHEMA_VERSION}:
        raise ValueError(f"不支持的扫描结果版本: {version!r}")
    legacy = version == 1
    metadata = dict(data.get("metadata", {}))
    if legacy:
        warnings.warn(
            "Legacy schema 1: timing and source are uncertain; "
            "旧版计时/来源不确定，data_origin 强制设为 unspecified_legacy。",
            UserWarning, stacklevel=2,
        )
        metadata["data_origin"] = "unspecified_legacy"
    thresholds = RegimeThresholds(**data["thresholds"])
    nodes = [_node_from_dict(item, legacy=legacy, thresholds=thresholds) for item in data["nodes"]]
    by_index = _node_map(nodes)
    brackets: list[BoundaryBracket] = []
    for item in data.get("brackets", []):
        bracket = BoundaryBracket(
            station_q1=float(item["station_q1"]),
            kind=str(item["kind"]),
            direction=str(item["direction"]),
            inside=_reference(by_index, item["inside_index"]),
            outside=_reference(by_index, item["outside_index"]),
            boundary_q2=float(item["boundary_q2"]),
            relative_width=float(item["relative_width"]),
            at_domain_edge=bool(item.get("at_domain_edge", False)),
            resolution="uncertain" if legacy else str(item["resolution"]),
        )
        if not legacy and item["q2_interval"] != list(bracket.q2_interval):
            raise ValueError("边界 q2_interval 与端点观测不一致")
        brackets.append(bracket)
    result = ScanResult(
        config=ScanConfig(**data["config"]),
        center_node=_reference(by_index, data["center_index"]),
        nodes=nodes,
        brackets=brackets,
        thresholds=thresholds,
        metadata=metadata,
        stopped_reason=str(data.get("stopped_reason", "")),
    )
    _validate_references(result)
    return result


def save_result(result: ScanResult, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    payload = result_to_dict(result, metadata=metadata)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限数值: {value}")


def load_result(path: str | Path) -> ScanResult:
    return result_from_dict(json.loads(Path(path).read_text(encoding="utf-8"),
                                       parse_constant=_reject_nonfinite_constant))


def save_nodes_csv(result: ScanResult, path: str | Path) -> Path:
    _validate_references(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index", "q1_ul_min", "q2_ul_min", "flow_ratio", "regime",
                "confidence", "droplet_rate_hz", "mean_diameter_ratio",
                "stream_coverage", "max_plug_length_ratio", "started_s",
                "duration_s", "purpose", "data_origin", "source", "droplet_events",
                "droplets_observed", "uniformity_candidate", "best_operating_point_verified",
                "diameter_cv", "junction_visible", "breakup_distance_h", "observation_extent_h",
                "diameter_change_from_anchor", "breakup_shift_h_from_anchor", "evolution_stage",
            ]
        )
        for node in result.nodes:
            formation = assess_formation(node.features, result.thresholds)
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
                    result.metadata.get("data_origin", "unspecified"), node.features.source,
                    formation["droplet_events"], formation["droplets_observed"],
                    formation["uniformity_candidate"], formation["best_operating_point_verified"],
                    node.features.diameter_cv, node.features.junction_visible,
                    node.features.breakup_distance_h, node.features.observation_extent_h,
                    node.features.extra.get("diameter_change_from_anchor"),
                    node.features.extra.get("breakup_shift_h_from_anchor"),
                    node.features.extra.get("evolution_stage", "unknown"),
                ]
            )
    return path
