"""Calibration evidence tests: no local historical data or frame access required."""

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import pytest

from mcs_date.experiment.calibration import (
    FALLBACK_SIZE,
    CalibrationResult,
    _fit_size,
    collect_campaign_size,
    collect_response_library,
    collect_session_survey,
    learn_calibration,
    summarize_calibration,
)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def survey(root: Path, records: list[object]) -> Path:
    path = root / "experiments/run/session_summary.json"
    write_json(path, {"regime_survey": {"records": records}})
    return path


def record(q1=70.0, q2=20.0, regime="dripping", review_status="reviewed", **kwargs):
    return {"q1_command_ul_min": q1, "q2_command_ul_min": q2,
            "regime": regime, "review_status": review_status, **kwargs}


def campaign(root: Path, stages: list[object]) -> Path:
    path = root / "experiments/experiment_20261016/campaign_analysis/fit_results.json"
    write_json(path, {"stages": stages})
    return path


def library(root: Path, payloads: list[object]) -> Path:
    path = root / "models/response_library.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE experiments (payload TEXT)")
        connection.executemany("INSERT INTO experiments VALUES (?)",
                               [(json.dumps(x),) for x in payloads])
    return path


def test_no_history_is_entirely_prior_with_zero_labels(tmp_path):
    root = tmp_path / "missing"
    result = learn_calibration(root)
    assert not root.exists()  # especially no accidental SQLite DB creation
    assert result.ratio_low_confidence == result.ratio_high_confidence == "prior"
    assert result.ratio_thread_confidence == result.size_model_confidence == "prior"
    assert result.valid_label_points == result.droplet_points == result.no_droplet_points == 0
    assert not result.observation_ledger
    assert not result.window_identified
    assert result.calibration_status == "prior"
    assert any("missing history directory" in warning for warning in result.warnings)
    assert result.ratio_high == CalibrationResult().ratio_high


def test_unreviewed_sizes_candidates_and_long_jet_are_not_labels(tmp_path):
    campaign(tmp_path, [{"q1": 70, "q2": 20, "D_quantiles": [90, 92, 94]}])
    survey(tmp_path, [record(review_status="pending"),
                      record(q1=90.09, regime="jetting", review_status="reviewed"),
                      record(q1=100, regime="no_droplet_high", review_status="pending")])
    result = learn_calibration(tmp_path)
    assert result.size_points == 1
    assert result.survey_records == 3 and result.survey_reviewed == 1
    assert result.valid_label_points == 0
    assert result.ratio_high_confidence == "prior"
    assert all(not item["window_eligible"] for item in result.observation_ledger)
    assert result.size_model_confidence == "prior"  # rank insufficient


def test_reviewed_survey_learns_high_interval_and_hashes_sources(tmp_path):
    path = survey(tmp_path, [record(), record(q1=90, regime="no_droplet", boundary_side="high",
                                            review_status="confirmed")])
    before = path.read_bytes()
    result = learn_calibration(tmp_path)
    assert result.valid_label_points == 2
    assert result.droplet_points == result.no_droplet_points == 1
    assert result.ratio_high_interval == (3.5, 4.5)
    assert result.ratio_high == pytest.approx(math.sqrt(3.5 * 4.5))
    assert result.ratio_high_confidence == "medium"
    assert result.ratio_low_confidence == "prior"
    assert result.calibration_status == "partial" and not result.window_identified
    assert result.observation_ledger[0]["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    assert collect_session_survey(tmp_path) == (2, 2)


def test_jetting_with_explicit_downstream_breakup_is_positive_not_negative(tmp_path):
    survey(tmp_path, [record(regime="jetting", droplet_formation=True),
                      record(q1=90, regime="jetting"),
                      record(q1=100, regime="jetting", droplet_formation=False,
                             boundary_side="high")])
    result = learn_calibration(tmp_path)
    assert result.droplet_points == 1 and result.no_droplet_points == 1
    assert result.valid_label_points == 2
    assert result.ratio_high_interval == (3.5, 5.0)
    assert not result.observation_ledger[1]["window_eligible"]


def test_low_side_never_becomes_high_side_or_plain_negative_assumed_high(tmp_path):
    result = learn_calibration(tmp_path, extra_observations=[
        (70, 20, "droplet", "high", "operator: near-junction breakup"),
        (40, 20, "no_droplet_low", "high", "operator: low side"),
        (100, 20, "no_droplet", "medium", "operator: side not reviewed"),
    ])
    assert result.ratio_low_interval == (2.0, 3.5)
    assert result.ratio_low_confidence == "medium"
    assert result.ratio_high_interval is None
    assert result.ratio_high_confidence == "prior"
    assert result.ratio_high == 4.185
    assert result.no_droplet_points == 2
    assert any("without explicit side" in x for x in result.warnings)
    assert all(x["input_sha256"] for x in result.observation_ledger)


def test_both_boundaries_can_be_bounded_by_reviewed_manual_evidence(tmp_path):
    result = learn_calibration(tmp_path, extra_observations=[
        (60, 20, "droplet", "high", "operator:A"),
        (40, 20, "no_droplet_low", "high", "operator:B"),
        (90, 20, "no_droplet_high", "high", "operator:C"),
    ])
    assert result.window_identified and result.calibration_status == "bounded"
    assert result.ratio_low_interval == (2, 3)
    assert result.ratio_high_interval == (3, 4.5)


@pytest.mark.parametrize("negative", [
    record(q1=70, regime="no_droplet", boundary_side="high"),
    record(q1=35, q2=10, regime="no_droplet", boundary_side="high"),
    record(q1=60, regime="no_droplet", boundary_side="high"),
])
def test_conflicts_including_total_flow_overlap_retain_prior(tmp_path, negative):
    survey(tmp_path, [record(q1=60), record(q1=80), negative])
    result = learn_calibration(tmp_path)
    assert result.calibration_status == "conflict"
    assert result.ratio_high == 4.185
    assert result.ratio_high_confidence == "prior"
    assert result.ratio_high_interval is None
    assert not result.window_identified
    assert any("conflicting ratio constraints" in warning for warning in result.warnings)


def test_prior_thresholds_not_adjusted_to_accommodate_unreachable_bracket(tmp_path):
    result = learn_calibration(tmp_path, extra_observations=[
        (180, 20, "droplet", "high", "operator:P"),
        (200, 20, "no_droplet_high", "high", "operator:N"),
    ])
    assert result.ratio_high == 4.185 and result.ratio_thread == 6.0
    assert result.calibration_status == "conflict"
    assert result.ratio_high_confidence == "prior"
    assert result.ratio_high_interval == (9, 10)  # constraint retained, not discarded


def test_response_library_45_records_are_coverage_not_regime_or_size_fit(tmp_path):
    path = library(tmp_path, [{"q_after": [50 + i, 20], "lengths_px": [10, 20, 5000],
                               "regime": "dripping", "review_status": "reviewed"}
                              for i in range(45)])
    before = path.read_bytes()
    result = learn_calibration(tmp_path)
    assert result.response_library_records == result.response_library_flow_points == 45
    assert result.response_library_size_candidates == 45
    assert result.valid_label_points == result.size_points == 0
    assert result.ratio_high_confidence == result.size_model_confidence == "prior"
    assert all(not x["window_eligible"] for x in result.observation_ledger)
    assert result.coverage["response_library_records"] == 45
    assert summarize_calibration(result)["evidence"]["response_library_records"] == 45
    assert len(collect_response_library(tmp_path)) == 45
    assert path.read_bytes() == before
    assert not path.with_name(path.name + "-journal").exists()


def test_missing_lengths_still_count_response_flow_coverage(tmp_path):
    library(tmp_path, [{"q_after": [50, 20]}, {"q_after": [60, 20], "lengths_px": [math.nan]},
                       [], {"q_after": [0, 20], "lengths_px": [1]}])
    result = learn_calibration(tmp_path)
    assert result.response_library_records == 4
    assert result.response_library_flow_points == 1  # NaN JSON is rejected as a whole payload
    assert result.response_library_size_candidates == 0
    assert len(result.warnings) >= 4


def test_contradictory_record_fields_are_excluded_with_diagnostics(tmp_path):
    survey(tmp_path, [record(regime="no_droplet_high", boundary_side="low"),
                      record(regime="dripping", droplet_formation=False),
                      record(regime="no_droplet", droplet_formation=True)])
    result = learn_calibration(tmp_path)
    assert result.valid_label_points == 0
    assert any("conflicting boundary sides" in x for x in result.warnings)
    assert any("dripping conflicts" in x for x in result.warnings)
    assert any("no_droplet conflicts" in x for x in result.warnings)


def test_full_rank_candidate_size_fit_never_creates_flow_labels(tmp_path):
    campaign(tmp_path, [{"q1": 60, "q2": 20, "D_quantiles": [90, 92, 94]},
                        {"q1": 70, "q2": 20, "D_quantiles": [89, 91, 93]},
                        {"q1": 60, "q2": 25, "D_quantiles": [88, 90, 92]}])
    result = learn_calibration(tmp_path)
    assert result.size_points == 3 and result.size_model_confidence == "low"
    assert result.size_intercept_um == pytest.approx(92)
    assert result.size_q1_coef_um_per_ulmin == pytest.approx(-0.1)
    assert result.size_q2_coef_um_per_ulmin == pytest.approx(-0.4)
    assert result.valid_label_points == 0
    assert result.ratio_high_confidence == "prior"


def test_rank_deficient_size_fit_falls_back(tmp_path):
    points = [(50 + i * 10, 20, 92 - i, "test") for i in range(3)]
    assert _fit_size(points) == FALLBACK_SIZE
    campaign(tmp_path, [{"q1": q1, "q2": q2, "D_quantiles": [d - 1, d, d + 1]}
                        for q1, q2, d, _ in points])
    result = learn_calibration(tmp_path)
    assert result.size_points == 3
    assert result.size_model_confidence == "prior"
    assert any("rank-deficient" in x for x in result.warnings)


@pytest.mark.parametrize("bad", [[], {"stages": {}}, {"stages": [None, {}, {"q1": "70"}]},
                                 {"stages": [{"q1": 70, "q2": 20,
                                              "D_quantiles": [90, math.nan, 100]}]}])
def test_bad_campaign_schema_is_safe(tmp_path, bad):
    path = tmp_path / "experiments/experiment_20261016/campaign_analysis/fit_results.json"
    write_json(path, bad)
    result = learn_calibration(tmp_path)
    assert result.size_points == result.valid_label_points == 0
    assert result.warnings
    assert collect_campaign_size(tmp_path) == []


def test_bad_json_and_sqlite_and_survey_schema_are_safe(tmp_path):
    path = campaign(tmp_path, [])
    path.write_text("{bad json", encoding="utf-8")
    database = tmp_path / "models/response_library.sqlite3"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"not SQLite")
    survey(tmp_path, [None, record(q1="inf"), record(q2=0),
                      record(droplet_formation="yes"), record(confidence="nonsense")])
    result = learn_calibration(tmp_path)
    assert result.valid_label_points == 0 and result.survey_records == 5
    assert result.response_library_records == 0
    assert any("unreadable response library" in x for x in result.warnings)
    assert any("invalid JSON" in x for x in result.warnings)
    assert not result.window_identified


@pytest.mark.parametrize("item", [
    (math.nan, 20, "droplet", "high", "test"),
    (70, math.inf, "droplet", "high", "test"),
    (70, 0, "droplet", "high", "test"),
    (-1, 20, "droplet", "high", "test"),
    (True, 20, "droplet", "high", "test"),
    (70, 20, "candidate", "high", "test"),
    (70, 20, "droplet", "prior", "test"),
    (70, 20, "droplet", [], "test"),
    (10 ** 1000, 20, "droplet", "high", "test"),
    (70, 20, "droplet", "high", ""),
    (70, 20, "droplet"),
])
def test_invalid_manual_inputs_are_rejected(tmp_path, item):
    with pytest.raises(ValueError):
        learn_calibration(tmp_path, extra_observations=[item])


def test_serialization_roundtrip_and_old_api_compatible(tmp_path):
    result = learn_calibration(tmp_path, extra_observations=[
        (70, 20, "droplet", "high", "operator:P"),
        (90, 20, "no_droplet_high", "high", "operator:N"),
    ])
    path = result.save(tmp_path / "result.json")
    assert CalibrationResult.load(path) == result
    assert CalibrationResult.from_dict(json.loads(json.dumps(result.as_dict()))) == result
    old = CalibrationResult.from_dict({"schema_version": 1, "ratio_high": 4.0,
                                       "sources": ["old.json"], "unknown_future_key": 1})
    assert old.ratio_high == 4.0 and old.sources == ("old.json",)
    assert old.regime_name(70, 20) == "dripping"
    assert old.droplet_q1_window(20) == (44, 80)
    assert old.droplet_diameter_um(60, 20) > 0
    summary = summarize_calibration(result)
    assert {"ratio_window", "ratio_thread", "confidence", "evidence", "size_model_um",
            "sources", "notes"} <= summary.keys()


@pytest.mark.parametrize("data", [[], {"schema_version": 999}, {"ratio_high": math.nan},
                                  {"ratio_low": 5}, {"size_q1_coef_um_per_ulmin": math.inf},
                                  {"sources": "not-an-array"}, {"coverage": []},
                                  {"valid_label_points": -1}, {"warnings": [1]},
                                  {"notes": {}}, {"ratio_high_confidence": []},
                                  {"calibration_status": {}}, {"ratio_high": 10 ** 1000},
                                  {"ratio_high_interval": [4, 3]},
                                  {"coverage": {"bad": math.nan}}])
def test_invalid_result_schema_rejected(data):
    with pytest.raises(ValueError):
        CalibrationResult.from_dict(data)


def test_size_fit_missing_numpy_falls_back(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("numpy deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_numpy)
    assert _fit_size([(60, 20, 92, "A"), (70, 20, 91, "B"),
                      (60, 25, 90, "C")]) == FALLBACK_SIZE


def test_low_confidence_review_does_not_become_medium_confidence(tmp_path):
    result = learn_calibration(tmp_path, extra_observations=[
        (70, 20, "droplet", "low", "operator:P"),
        (90, 20, "no_droplet_high", "high", "operator:N"),
    ])
    assert result.ratio_high_confidence == "low"


def test_sqlite_connection_is_read_only_and_schema_errors_are_safe(tmp_path, monkeypatch):
    path = library(tmp_path, [{"q_after": [50, 20], "lengths_px": [1]}])
    original = sqlite3.connect
    calls = []

    def read_only_connect(database, **kwargs):
        calls.append((database, kwargs))
        connection = original(database, **kwargs)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO experiments VALUES ('{}')")
        return connection

    monkeypatch.setattr(sqlite3, "connect", read_only_connect)
    assert len(collect_response_library(tmp_path)) == 1
    assert calls[0] == (path.resolve().as_uri() + "?mode=ro&immutable=1", {"uri": True})
    monkeypatch.setattr(sqlite3, "connect", original)
    with original(path) as connection:
        connection.execute("DROP TABLE experiments")
    result = learn_calibration(tmp_path)
    assert result.response_library_records == 0
    assert any("no such table" in x for x in result.warnings)


def test_sqlite_wal_snapshot_is_not_silently_ignored_or_modified(tmp_path):
    path = library(tmp_path, [{"q_after": [50, 20], "lengths_px": [1]}])
    wal = path.with_name(path.name + "-wal")
    wal.write_bytes(b"uncheckpointed snapshot placeholder")
    before = path.read_bytes(), wal.read_bytes()
    result = learn_calibration(tmp_path)
    assert result.response_library_records == 0
    assert any("SQLite WAL present" in x for x in result.warnings)
    assert (path.read_bytes(), wal.read_bytes()) == before
    assert not path.with_name(path.name + "-shm").exists()


def test_load_nonfinite_json_and_query_invalid_flow_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"ratio_high": NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        CalibrationResult.load(path)
    result = CalibrationResult()
    for q1, q2 in [(70, 0), (math.nan, 20), (70, -1)]:
        with pytest.raises(ValueError):
            result.regime_name(q1, q2)
        with pytest.raises(ValueError):
            result.droplet_diameter_um(q1, q2)
    with pytest.raises(ValueError):
        result.droplet_q1_window(0)
