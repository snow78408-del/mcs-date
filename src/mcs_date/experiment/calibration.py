"""Read-only, evidence-led calibration of the oil Q1 / water Q2 formation window.

Only explicitly reviewed/confirmed formation labels constrain the window. Size
measurements, long candidates and jetting alone do not establish (non-)formation:
a jet may break up downstream. The ratio-only model is a hypothesis, not a phase
classifier or a claim that the middle band is optimal dripping. Numeric defaults
remain priors for API/simulator compatibility, never thresholds tuned for reachability.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
DEFAULT_HISTORY_ROOT = Path(os.environ.get("MCS_QT_DATA_DIR", r"E:\MCS_QT_Data"))
# Explicit reviewed manual input: (Q1, Q2, label, confidence, nonempty source).
# no_droplet has unknown side; use no_droplet_high / no_droplet_low for a boundary.
FlowObservation = tuple[float, float, str, str, str]
SizeObservation = tuple[float, float, float, str]
FALLBACK_SIZE = (92.2242, -0.02273, -0.41219)
_REVIEWED = {"reviewed", "confirmed"}
_CONFIDENCES = {"prior", "low", "medium", "high", "conflict"}
_MANUAL_LABELS = {"droplet", "no_droplet", "no_droplet_high", "no_droplet_low",
                  "dripping", "jetting", "threading", "tubing"}


def _positive(value: Any) -> bool:
    return _finite(value) and value > 0


def _flow(q1: Any, q2: Any) -> tuple[float, float]:
    if not _positive(q1) or not _positive(q2):
        raise ValueError("Q1 and Q2 must be finite positive numbers")
    q1, q2 = float(q1), float(q2)
    if not math.isfinite(q1 / q2) or not math.isfinite(q1 + q2):
        raise ValueError("flow ratio and total flow must be finite")
    return q1, q2


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class CalibrationResult:
    """Numeric predictions plus auditable evidence; confidence must accompany predictions."""

    ratio_low: float = 2.2
    ratio_high: float = 4.185
    ratio_thread: float = 6.0
    size_intercept_um: float = FALLBACK_SIZE[0]
    size_q1_coef_um_per_ulmin: float = FALLBACK_SIZE[1]
    size_q2_coef_um_per_ulmin: float = FALLBACK_SIZE[2]
    nominal_cross_section_um: float = 50.0
    measurement_scale_um_per_px: float = 2.2667
    ratio_low_confidence: str = "prior"
    ratio_high_confidence: str = "prior"
    ratio_thread_confidence: str = "prior"
    droplet_points: int = 0
    no_droplet_points: int = 0
    size_points: int = 0
    survey_records: int = 0
    survey_reviewed: int = 0
    sources: tuple[str, ...] = ()
    notes: str = ""
    valid_label_points: int = 0
    response_library_records: int = 0
    response_library_flow_points: int = 0
    response_library_size_candidates: int = 0
    size_model_confidence: str = "prior"
    calibration_status: str = "prior"
    window_identified: bool = False
    ratio_low_interval: tuple[float, float] | None = None
    ratio_high_interval: tuple[float, float] | None = None
    observation_ledger: tuple[dict[str, Any], ...] = ()
    interval_constraints: tuple[dict[str, Any], ...] = ()
    source_provenance: tuple[dict[str, Any], ...] = ()
    coverage: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("ratio_low", "ratio_high", "ratio_thread", "size_intercept_um",
                     "nominal_cross_section_um", "measurement_scale_um_per_px"):
            if not _positive(getattr(self, name)):
                raise ValueError(f"{name} must be finite and positive")
        if not self.ratio_low < self.ratio_high < self.ratio_thread:
            raise ValueError("ratios must satisfy ratio_low < ratio_high < ratio_thread")
        for name in ("size_q1_coef_um_per_ulmin", "size_q2_coef_um_per_ulmin"):
            if not _finite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("ratio_low_confidence", "ratio_high_confidence", "ratio_thread_confidence",
                     "size_model_confidence"):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in _CONFIDENCES:
                raise ValueError(f"invalid {name}")
        for name in ("droplet_points", "no_droplet_points", "size_points", "survey_records",
                     "survey_reviewed", "valid_label_points", "response_library_records",
                     "response_library_flow_points", "response_library_size_candidates"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("ratio_low_interval", "ratio_high_interval"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (tuple, list)) or len(value) != 2
                                      or not all(_positive(x) for x in value)
                                      or value[0] >= value[1]):
                raise ValueError(f"invalid {name}")
        if not isinstance(self.window_identified, bool):
            raise ValueError("window_identified must be boolean")
        if (not isinstance(self.calibration_status, str)
                or self.calibration_status not in {"prior", "partial", "bounded", "conflict"}):
            raise ValueError("invalid calibration_status")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        if not isinstance(self.coverage, dict):
            raise ValueError("coverage must be an object")
        for name in ("sources", "warnings"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) for x in value):
                raise ValueError(f"{name} must be a string sequence")
        for name in ("observation_ledger", "interval_constraints", "source_provenance"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or not all(isinstance(x, dict) for x in value):
                raise ValueError(f"{name} must be an object sequence")
        # Reject nonfinite values even inside newly added ledger/coverage fields.
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("calibration must contain finite JSON values") from exc

    def regime_name(self, q1: float, q2: float) -> str:
        """Legacy ratio-model prediction, NOT a reviewed observation or formation verdict."""
        q1, q2 = _flow(q1, q2)
        r = q1 / q2
        if r < self.ratio_low:
            return "tubing"
        if r <= self.ratio_high:
            return "dripping"
        if r <= self.ratio_thread:
            return "jetting"
        return "threading"

    def droplet_diameter_um(self, q1: float, q2: float) -> float:
        """Candidate-size model; does not prove dripping or physical scale validity."""
        q1, q2 = _flow(q1, q2)
        value = (self.size_intercept_um + self.size_q1_coef_um_per_ulmin * (q1 - 60.0)
                 + self.size_q2_coef_um_per_ulmin * (q2 - 20.0))
        if not math.isfinite(value):
            raise ValueError("size prediction overflow")
        return max(1.0, value)

    def droplet_q1_window(self, q2: float) -> tuple[float, float]:
        """Legacy numeric candidate window; inspect confidence/intervals before using it."""
        _flow(1.0, q2)
        low, high = self.ratio_low * q2, self.ratio_high * q2
        if not math.isfinite(high):
            raise ValueError("window prediction overflow")
        return low, high

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        for name in ("sources", "warnings", "observation_ledger", "interval_constraints",
                     "source_provenance"):
            data[name] = list(data[name])
        for name in ("ratio_low_interval", "ratio_high_interval"):
            if data[name] is not None:
                data[name] = list(data[name])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationResult":
        if not isinstance(data, dict):
            raise ValueError("calibration must be a JSON object")
        version = data.get("schema_version", 1)
        if type(version) is not int or version not in {1, SCHEMA_VERSION}:
            raise ValueError("unsupported calibration schema_version")
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for name in ("sources", "warnings", "observation_ledger", "interval_constraints",
                     "source_provenance", "ratio_low_interval", "ratio_high_interval"):
            if name in payload and payload[name] is not None:
                if not isinstance(payload[name], (list, tuple)):
                    raise ValueError(f"{name} must be an array")
                payload[name] = tuple(payload[name])
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        return cls.from_dict(_json_loads(Path(path).read_text(encoding="utf-8")))


def _json_loads(text: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(text, parse_constant=reject)


def _read_json(path: Path, warnings: list[str] | None = None) -> Any | None:
    try:
        return _json_loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        if warnings is not None:
            warnings.append(f"unavailable/invalid JSON {path}: {exc}")
        return None


def _provenance(path: Path, warnings: list[str]) -> dict[str, Any]:
    item: dict[str, Any] = {"source": str(path), "access": "read_only"}
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        item["sha256"] = digest.hexdigest()
    except OSError as exc:
        item["sha256"] = None
        warnings.append(f"cannot hash {path}: {exc}")
    return item


def _campaign(root: Path, warnings: list[str]
              ) -> tuple[list[SizeObservation], list[dict[str, Any]]]:
    path = root / "experiments/experiment_20261016/campaign_analysis/fit_results.json"
    data = _read_json(path, warnings)
    points: list[SizeObservation] = []
    ledger: list[dict[str, Any]] = []
    if data is None:
        return points, ledger
    if not isinstance(data, dict) or not isinstance(data.get("stages"), list):
        warnings.append(f"invalid campaign schema: {path}")
        return points, ledger
    for i, stage in enumerate(data["stages"]):
        try:
            if not isinstance(stage, dict):
                raise ValueError("stage must be object")
            q1, q2 = _flow(stage.get("q1"), stage.get("q2"))
            quantiles = stage.get("D_quantiles")
            if (not isinstance(quantiles, list) or len(quantiles) < 2
                    or not all(_positive(x) for x in quantiles)
                    or quantiles != sorted(quantiles)):
                raise ValueError("D_quantiles must be ordered positive finite numbers")
            d = float(quantiles[1])
        except ValueError as exc:
            warnings.append(f"invalid campaign stage {path}#{i}: {exc}")
            continue
        source = f"{path}#stages/{i}"
        points.append((q1, q2, d, source))
        ledger.append({"source": source, "kind": "size_candidate", "q1": q1, "q2": q2,
                       "ratio": q1 / q2, "total_flow": q1 + q2,
                       "diameter_um": d, "window_eligible": False,
                       "exclusion_reason": "size does not establish formation or optimal dripping"})
    return points, ledger


def collect_campaign_size(root: Path = DEFAULT_HISTORY_ROOT) -> list[SizeObservation]:
    """Campaign candidate-size observations only; never inferred formation labels."""
    return _campaign(Path(root), [])[0]


def _library(root: Path, warnings: list[str]
             ) -> tuple[list[SizeObservation], list[dict[str, Any]], int]:
    path = root / "models/response_library.sqlite3"
    out: list[SizeObservation] = []
    ledger: list[dict[str, Any]] = []
    if not path.is_file():
        warnings.append(f"missing response library: {path}")
        return out, ledger, 0
    if path.with_name(path.name + "-wal").exists():
        warnings.append(f"SQLite WAL present; skip incomplete snapshot "
                        f"without touching sidecars: {path}")
        return out, ledger, 0
    try:
        # immutable forbids journal/SHM side effects; WAL snapshots were excluded above.
        uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as con:
            con.execute("PRAGMA query_only=ON")
            rows = con.execute("SELECT payload FROM experiments").fetchall()
    except sqlite3.Error as exc:
        warnings.append(f"unreadable response library {path}: {exc}")
        return out, ledger, 0
    for i, (payload,) in enumerate(rows):
        source = f"{path}#experiments/{i}"
        try:
            data = _json_loads(payload)
            if not isinstance(data, dict):
                raise ValueError("payload must be object")
            after = data.get("q_after")
            if not isinstance(after, list) or len(after) != 2:
                raise ValueError("q_after must contain Q1 and Q2")
            q1, q2 = _flow(*after)
        except (ValueError, TypeError) as exc:
            warnings.append(f"invalid response payload {source}: {exc}")
            continue
        entry = {"source": source, "kind": "response_coverage", "q1": q1, "q2": q2,
                 "ratio": q1 / q2, "total_flow": q1 + q2, "window_eligible": False,
                 "exclusion_reason": "response candidate lengths are not reviewed flow labels"}
        lengths = data.get("lengths_px")
        if isinstance(lengths, list) and lengths and all(_positive(x) for x in lengths):
            lengths = sorted(lengths)
            length = float(lengths[len(lengths) // 2])
            out.append((q1, q2, length, source))
            entry["candidate_length_px"] = length
        else:
            warnings.append(f"missing/invalid candidate lengths (coverage retained): {source}")
        ledger.append(entry)
    return out, ledger, len(rows)


def collect_response_library(root: Path = DEFAULT_HISTORY_ROOT) -> list[SizeObservation]:
    """Legacy tuples: Q1, Q2, median candidate length in PIXELS, source; NOT size-fit data."""
    return _library(Path(root), [])[0]


def _record(record: dict[str, Any], source: str, *, manual: bool = False) -> dict[str, Any]:
    q1 = record.get("q1", record.get("q1_command_ul_min", record.get("q1_ul_min")))
    q2 = record.get("q2", record.get("q2_command_ul_min", record.get("q2_ul_min")))
    q1, q2 = _flow(q1, q2)
    review = str(record.get("review_status", "pending")).lower()
    regime = str(record.get("regime", "unknown")).lower()
    label = str(record.get("formation_label", record.get("droplet_label", "unknown"))).lower()
    side = str(record.get("boundary_side", record.get("side", "unknown"))).lower()
    side = {"high_side": "high", "low_side": "low"}.get(side, side)
    formation: bool | None = None
    explicit = record.get("droplet_formation")
    if explicit is not None and not isinstance(explicit, bool):
        raise ValueError("droplet_formation must be boolean or null")
    if explicit is not None:
        formation = explicit
    if label in {"droplet", "no_droplet", "no_droplet_high", "no_droplet_low"}:
        by_label = label == "droplet"
        if formation is not None and formation != by_label:
            raise ValueError("conflicting formation labels in record")
        formation = by_label
        if label.endswith(("_high", "_low")):
            asserted_side = label.rsplit("_", 1)[1]
            if side in {"high", "low"} and side != asserted_side:
                raise ValueError("conflicting boundary sides in record")
            side = asserted_side
    # Dripping explicitly means observed formation, jetting/threading alone do not.
    if regime in {"dripping", "droplet"}:
        if formation is False:
            raise ValueError("dripping conflicts with no_droplet")
        formation = True
    if regime in {"no_droplet", "no_droplet_high", "no_droplet_low"}:
        if formation is True:
            raise ValueError("no_droplet conflicts with droplet_formation")
        formation = False
        if regime.endswith(("_high", "_low")):
            asserted_side = regime.rsplit("_", 1)[1]
            if side in {"high", "low"} and side != asserted_side:
                raise ValueError("conflicting boundary sides in record")
            side = asserted_side
    confidence = str(record.get("confidence", "medium")).lower()
    if confidence not in {"low", "medium", "high"}:
        raise ValueError("reviewed observation confidence must be low/medium/high")
    eligible = review in _REVIEWED and formation is not None
    return {"source": source, "kind": "manual_review" if manual else "session_survey",
            "q1": q1, "q2": q2, "ratio": q1 / q2, "total_flow": q1 + q2,
            "regime": regime, "droplet_formation": formation, "boundary_side": side,
            "review_status": review, "confidence": confidence, "window_eligible": eligible,
            "flow_basis": record.get("flow_basis", "manual_input" if manual else "unspecified"),
            "exclusion_reason": (None if eligible else "not reviewed/confirmed" if review not in
                                 _REVIEWED else "regime alone does not establish formation")}


def _survey(root: Path, warnings: list[str], *, limit: int = 400
            ) -> tuple[int, int, list[dict[str, Any]], list[Path]]:
    if type(limit) is not int or limit <= 0:
        raise ValueError("survey limit must be a positive integer")
    total = reviewed = 0
    ledger: list[dict[str, Any]] = []
    paths: list[Path] = []
    experiments = root / "experiments"
    if not experiments.is_dir():
        warnings.append(f"missing survey directory: {experiments}")
        return total, reviewed, ledger, paths
    try:
        # Only metadata files are read; no frame/image/video contents are opened.
        for index, path in enumerate(sorted(experiments.rglob("session_summary.json"))):
            if index >= limit:
                warnings.append(f"survey file limit {limit} reached; coverage is incomplete")
                break
            paths.append(path)
            data = _read_json(path, warnings)
            if data is None:
                continue
            survey = data.get("regime_survey") if isinstance(data, dict) else None
            if not isinstance(survey, dict) or not isinstance(survey.get("records"), list):
                warnings.append(f"missing/invalid regime_survey schema: {path}")
                continue
            for i, record in enumerate(survey["records"]):
                total += 1
                source = f"{path}#regime_survey/records/{i}"
                if not isinstance(record, dict):
                    warnings.append(f"invalid survey record: {source}")
                    continue
                if str(record.get("review_status", "")).lower() in _REVIEWED:
                    reviewed += 1
                try:
                    ledger.append(_record(record, source))
                except ValueError as exc:
                    warnings.append(f"invalid survey record {source}: {exc}")
    except OSError as exc:
        warnings.append(f"cannot enumerate survey metadata: {exc}")
    return total, reviewed, ledger, paths


def collect_session_survey(root: Path = DEFAULT_HISTORY_ROOT, *,
                           limit: int = 400) -> tuple[int, int]:
    """Return total and explicitly reviewed/confirmed survey counts (not valid-label counts)."""
    total, reviewed, _, _ = _survey(Path(root), [], limit=limit)
    return total, reviewed


def _size_fit(points: Iterable[SizeObservation]) -> tuple[tuple[float, float, float], bool]:
    rows = list(points)
    if len(rows) < 3:
        return FALLBACK_SIZE, False
    try:
        import numpy as np
    except ImportError:
        return FALLBACK_SIZE, False
    try:
        a = np.array([[1.0, q1 - 60.0, q2 - 20.0] for q1, q2, _, _ in rows], dtype=float)
        b = np.array([d for _, _, d, _ in rows], dtype=float)
        coef, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
        if (rank < 3 or not np.all(np.isfinite(coef)) or coef[0] <= 0
                or singular[-1] <= 0 or singular[0] / singular[-1] > 1e10):
            return FALLBACK_SIZE, False
        return (float(coef[0]), float(coef[1]), float(coef[2])), True
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        return FALLBACK_SIZE, False


def _fit_size(points: Iterable[SizeObservation]) -> tuple[float, float, float]:
    """Backward-compatible candidate-size least squares, with rank/finite checks."""
    valid = []
    for q1, q2, diameter, source in points:
        try:
            q1, q2 = _flow(q1, q2)
        except ValueError:
            continue
        if _positive(diameter):
            valid.append((q1, q2, diameter, source))
    return _size_fit(valid)[0]


def _boundaries(observations: list[dict[str, Any]], warnings: list[str]
                ) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]], bool]:
    positives = [x for x in observations if x["droplet_formation"] is True]
    negatives = [x for x in observations if x["droplet_formation"] is False]
    constraints: list[dict[str, Any]] = []
    for entry in observations:
        formation, side = entry["droplet_formation"], entry["boundary_side"]
        kind = "positive_inside_window" if formation else f"negative_{side}_side"
        constraints.append({"kind": kind, "ratio": entry["ratio"],
                            "total_flow": entry["total_flow"], "source": entry["source"]})
    if not positives:
        return {}, constraints, False
    p_min, p_max = min(x["ratio"] for x in positives), max(x["ratio"] for x in positives)
    conflict = False
    for negative in negatives:
        r, side = negative["ratio"], negative["boundary_side"]
        if p_min <= r <= p_max or (side == "high" and r <= p_max) or (side == "low" and r >= p_min):
            conflict = True
            same_total = any(math.isclose(negative["total_flow"], x["total_flow"], rel_tol=1e-8)
                             for x in positives)
            warnings.append("conflicting ratio constraints" + (" / total-flow-confounded overlap"
                            if not same_total else "") + f": {negative['source']}")
    if conflict:
        return {}, constraints, True
    bounds: dict[str, tuple[float, float]] = {}
    highs = [x["ratio"] for x in negatives if x["boundary_side"] == "high"]
    lows = [x["ratio"] for x in negatives if x["boundary_side"] == "low"]
    if highs:
        bounds["high"] = p_max, min(highs)
    if lows:
        bounds["low"] = max(lows), p_min
    if any(x["boundary_side"] not in {"high", "low"} for x in negatives):
        warnings.append("no_droplet records without explicit side do not bound r_high or r_low")
    return bounds, constraints, False


def learn_calibration(history_root: str | Path | None = None, *,
                      extra_observations: Iterable[FlowObservation] = ()) -> CalibrationResult:
    """Learn only from reviewed evidence; absent, conflicting or insufficient data stays prior.

    ``extra_observations`` are explicit reviewed human inputs, not inferred labels.
    Tuples retain the original five-field API; use ``no_droplet_high``/``no_droplet_low``
    to assert the side, and a nonempty traceable source. Plain ``no_droplet`` is valid
    evidence of absence but cannot silently be assigned to the high boundary.
    """
    root = Path(history_root) if history_root is not None else DEFAULT_HISTORY_ROOT
    warnings: list[str] = []
    notes = ["Numeric defaults are priors, not validated thresholds or best-dripping claims",
             "Jetting may break up downstream; jetting alone is not no_droplet",
             "Ratio intervals constrain formation capability, not breakup location or stability"]
    manual: list[dict[str, Any]] = []
    for index, item in enumerate(extra_observations):
        if not isinstance(item, (list, tuple)) or len(item) != 5:
            raise ValueError("manual observation must be (Q1, Q2, label, confidence, source)")
        q1, q2, label, confidence, source = item
        if not isinstance(label, str) or label not in _MANUAL_LABELS:
            raise ValueError(f"invalid manual observation label: {label!r}")
        if not isinstance(confidence, str) or confidence not in {"low", "medium", "high"}:
            raise ValueError("invalid manual observation confidence")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("manual observation source must be nonempty")
        record = {"q1": q1, "q2": q2, "regime": label,
                  "formation_label": label, "confidence": confidence, "review_status": "reviewed"}
        entry = _record(record, source, manual=True)
        entry["manual_input_index"] = index
        entry["input_sha256"] = hashlib.sha256(json.dumps(item, ensure_ascii=False,
                                                        allow_nan=False).encode()).hexdigest()
        manual.append(entry)
    ledger: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    if root.is_dir():
        sizes, campaign_ledger = _campaign(root, warnings)
        library, library_ledger, library_total = _library(root, warnings)
        total, reviewed, surveys, paths = _survey(root, warnings)
        ledger.extend(campaign_ledger + library_ledger + surveys)
        fixed = [root / "experiments/experiment_20261016/campaign_analysis/fit_results.json",
                 root / "models/response_library.sqlite3"]
        provenance = [_provenance(p, warnings) for p in fixed + paths if p.is_file()]
    else:
        sizes, library, library_ledger, library_total, total, reviewed = [], [], [], 0, 0, 0
        warnings.append(f"missing history directory: {root}; "
                        "all unsupported parameters remain priors")
    ledger.extend(manual)
    hashes = {x["source"]: x.get("sha256") for x in provenance}
    for entry in ledger:
        entry["source_sha256"] = hashes.get(entry["source"].split("#", 1)[0])
    eligible = [x for x in ledger if x["window_eligible"]]
    positives = [x for x in eligible if x["droplet_formation"] is True]
    negatives = [x for x in eligible if x["droplet_formation"] is False]
    bounds, constraints, conflict = _boundaries(eligible, warnings)
    low, high = 2.2, 4.185
    low_conf = high_conf = "prior"
    boundary_conf = "low" if any(x["confidence"] == "low" for x in eligible) else "medium"
    if not conflict:
        if "low" in bounds:
            low = math.sqrt(bounds["low"][0]) * math.sqrt(bounds["low"][1])
            low_conf = boundary_conf
        if "high" in bounds:
            high = math.sqrt(bounds["high"][0]) * math.sqrt(bounds["high"][1])
            high_conf = boundary_conf
        if not low < high < 6.0:
            # Do not move a prior threshold to make observations reachable.
            warnings.append("learned intervals incompatible with remaining prior model; "
                            "no numeric calibration")
            low, high, low_conf, high_conf, conflict = 2.2, 4.185, "prior", "prior", True
    if conflict:
        warnings.append("ratio-only window is not identifiable; retain priors, "
                        "inspect constraints by total flow")
    size_coef, size_fitted = _size_fit(sizes)
    if not size_fitted:
        warnings.append("size fit unavailable or rank-deficient/ill-conditioned; "
                        "size coefficients remain prior")
    else:
        notes.append("Size fit uses campaign candidate diameters only; "
                     "physical scale is not independently validated")
    if not eligible:
        warnings.append("zero valid reviewed formation labels; formation window remains prior")
    if not bounds:
        notes.append("No two-sided reviewed boundary bracket available")
    notes.append("r_low/r_high geometric midpoints summarize intervals, "
                 "not exact transition estimates")
    valid_flows = [x for x in ledger if "ratio" in x]
    coverage = {"response_library_records": library_total,
                "response_library_flow_points": len(library_ledger),
                "response_library_size_candidates": len(library),
                "campaign_size_points": len(sizes), "survey_records": total,
                "survey_reviewed": reviewed, "valid_label_points": len(eligible),
                "ratio_range": ([min(x["ratio"] for x in valid_flows),
                                 max(x["ratio"] for x in valid_flows)] if valid_flows else None),
                "total_flow_range": ([min(x["total_flow"] for x in valid_flows),
                                      max(x["total_flow"] for x in valid_flows)]
                                     if valid_flows else None)}
    identified = low_conf != "prior" and high_conf != "prior" and not conflict
    return CalibrationResult(
        ratio_low=low, ratio_high=high, ratio_low_confidence=low_conf,
        ratio_high_confidence=high_conf, size_intercept_um=size_coef[0],
        size_q1_coef_um_per_ulmin=size_coef[1], size_q2_coef_um_per_ulmin=size_coef[2],
        droplet_points=len(positives), no_droplet_points=len(negatives), size_points=len(sizes),
        survey_records=total, survey_reviewed=reviewed, valid_label_points=len(eligible),
        response_library_records=library_total, response_library_flow_points=len(library_ledger),
        response_library_size_candidates=len(library),
        size_model_confidence="low" if size_fitted else "prior",
        calibration_status=("conflict" if conflict else "bounded" if identified else "partial"
                            if low_conf != "prior" or high_conf != "prior" else "prior"),
        window_identified=identified, ratio_low_interval=bounds.get("low"),
        ratio_high_interval=bounds.get("high"), observation_ledger=tuple(ledger),
        interval_constraints=tuple(constraints), source_provenance=tuple(provenance),
        coverage=coverage, warnings=tuple(warnings),
        sources=tuple(dict.fromkeys([x["source"] for x in provenance] +
                                    [x["source"] for x in manual])), notes="; ".join(notes))


def summarize_calibration(result: CalibrationResult) -> dict[str, Any]:
    """CLI/report summary preserving v1 keys while exposing evidence limitations."""
    return {"ratio_window": [result.ratio_low, result.ratio_high],
            "ratio_thread": result.ratio_thread,
            "confidence": {"ratio_low": result.ratio_low_confidence,
                           "ratio_high": result.ratio_high_confidence,
                           "ratio_thread": result.ratio_thread_confidence,
                           "size_model": result.size_model_confidence},
            "evidence": {"droplet_points": result.droplet_points,
                         "no_droplet_points": result.no_droplet_points,
                         "size_points": result.size_points, "survey_records": result.survey_records,
                         "survey_reviewed": result.survey_reviewed,
                         "valid_label_points": result.valid_label_points,
                         "response_library_records": result.response_library_records,
                         "response_library_flow_points": result.response_library_flow_points,
                         "response_library_size_candidates": (
                             result.response_library_size_candidates)},
            "size_model_um": {"intercept": round(result.size_intercept_um, 4),
                              "per_ulmin_q1": round(result.size_q1_coef_um_per_ulmin, 5),
                              "per_ulmin_q2": round(result.size_q2_coef_um_per_ulmin, 5)},
            "calibration_status": result.calibration_status,
            "window_identified": result.window_identified,
            "ratio_low_interval": result.ratio_low_interval,
            "ratio_high_interval": result.ratio_high_interval,
            "interval_constraints": list(result.interval_constraints),
            "coverage": result.coverage, "warnings": list(result.warnings),
            "source_provenance": list(result.source_provenance),
            "sources": list(result.sources), "notes": result.notes}
