"""Deterministic safety regressions; no hardware, recordings, or history required."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

import pytest

from mcs_date.experiment.models import ScanConfig
from mcs_date.experiment.phase import FlowRegime, NodeFeatures, classify_regime
from mcs_date.experiment.runner import (
    ExperimentRunner,
    SimulatedMeasurement,
    VirtualClock,
)
from mcs_date.experiment.strategy import SpiderScanner, station_boundaries


class MeasuredTopology:
    """A measurement fixture: classification still uses real topology evidence."""

    def __init__(
        self,
        regime_at: Callable[[float, float], FlowRegime] | None = None,
    ) -> None:
        self.regime_at = regime_at or (
            lambda _q1, q2: FlowRegime.JETTING
            if 14.0 < q2 < 28.0 else FlowRegime.THREADING
        )
        self.calls: list[tuple[float, float, float]] = []

    def __call__(self, q1: float, q2: float, window_s: float) -> NodeFeatures:
        self.calls.append((q1, q2, window_s))
        regime = self.regime_at(q1, q2)
        forming = regime in {FlowRegime.JETTING, FlowRegime.DRIPPING}
        # Short test windows must still contain enough *deduplicated* events.
        events = max(3, math.ceil(5.0 * window_s)) if forming else 0
        return NodeFeatures(
            q1=q1,
            q2=q2,
            window_s=window_s,
            sample_count=max(3, math.ceil(10.0 * window_s)),
            source="real",
            droplet_events=events,
            droplet_rate_hz=events / window_s,
            mean_diameter_ratio=0.65 if regime is FlowRegime.DRIPPING else 0.4,
            diameter_cv=0.05,
            stream_coverage=0.4 if forming else 0.85,
            junction_visible=regime is not FlowRegime.UNKNOWN,
            breakup_distance_h=(0.5 if regime is FlowRegime.DRIPPING else 2.0)
            if forming else None,
            observation_extent_h=25.0 if regime is FlowRegime.THREADING else None,
        )


@pytest.fixture
def config() -> ScanConfig:
    return ScanConfig(
        node_duration_s=1.0,
        startup_s=0.25,
        q2_min=5.0,
        q2_max=45.0,
        max_stations=1,
        max_expand_steps=4,
        bisection_rounds=2,
        max_nodes=50,
        max_duration_s=100.0,
    )


@pytest.fixture
def measured_topology() -> MeasuredTopology:
    return MeasuredTopology()


def constant_measurement(regime: FlowRegime) -> MeasuredTopology:
    return MeasuredTopology(lambda _q1, _q2: regime)


def test_fixture_exercises_topology_not_simulation_heuristics(measured_topology) -> None:
    for q2, expected in [(20.0, FlowRegime.JETTING), (5.0, FlowRegime.THREADING)]:
        features = measured_topology(90.0, q2, 0.1)
        assert features.source == "real"
        assert classify_regime(features)[0] is expected
        if expected is FlowRegime.JETTING:
            assert features.droplet_events >= 3
            assert features.breakup_distance_h == 2.0
    unknown = constant_measurement(FlowRegime.UNKNOWN)(90.0, 20.0, 0.1)
    assert classify_regime(unknown) == (FlowRegime.UNKNOWN, 0.0)


def test_good_jetting_anchor_brackets_default_target_on_both_sides(
    config, measured_topology,
) -> None:
    assert ScanConfig().target_regime == "jetting"
    result = SpiderScanner(measured_topology, config).run()
    assert result.center_node.regime is FlowRegime.JETTING
    assert result.metadata["scan_target"] == "jetting"
    table = station_boundaries(result)[config.center_q1]
    assert set(table) == {"jetting_lower", "jetting_upper"}
    for kind, expected_boundary in [("jetting_lower", 14.0), ("jetting_upper", 28.0)]:
        bracket = table[kind]
        assert bracket.inside.regime is FlowRegime.JETTING
        assert bracket.outside.regime is FlowRegime.THREADING
        lo, hi = bracket.q2_interval
        assert lo <= expected_boundary <= hi
        assert bracket.boundary_q2 == pytest.approx(math.sqrt(lo * hi))
        assert bracket.relative_width == pytest.approx(hi / lo)
        assert bracket.resolution == "bracketed"
    assert all(node.regime is not FlowRegime.DRIPPING for node in result.nodes)


def test_dripping_is_an_explicit_optional_scan_target(config) -> None:
    measurement = MeasuredTopology(
        lambda _q1, q2: FlowRegime.DRIPPING
        if 14.0 < q2 < 28.0 else FlowRegime.THREADING
    )
    result = SpiderScanner(measurement, replace(config, target_regime="dripping")).run()
    assert result.center_node.regime is FlowRegime.DRIPPING
    assert result.metadata["scan_target"] == "dripping"
    assert set(station_boundaries(result)[config.center_q1]) == {
        "droplet_lower", "droplet_upper",
    }


@pytest.mark.parametrize("fail_at", [1, 2])
def test_scanner_does_not_swallow_measurement_runtime_error(
    config, measured_topology, fail_at,
) -> None:
    failure = RuntimeError("measurement transport failed")
    calls = 0

    def measure(q1, q2, window_s):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise failure
        return measured_topology(q1, q2, window_s)

    with pytest.raises(RuntimeError) as caught:
        SpiderScanner(measure, config).run()
    assert caught.value is failure
    assert calls == fail_at


def test_unknown_midpoint_preserves_valid_endpoints_but_marks_uncertain(config) -> None:
    midpoint = math.sqrt(20.0 * 10.0)

    def regime_at(_q1, q2):
        if math.isclose(q2, midpoint):
            return FlowRegime.UNKNOWN
        return FlowRegime.JETTING if 19.0 < q2 < 25.0 else FlowRegime.THREADING

    cfg = replace(config, step_ratio=2.0, max_expand_steps=1, q2_min=10.0, q2_max=40.0)
    result = SpiderScanner(MeasuredTopology(regime_at), cfg).run()
    bracket, = result.boundaries("jetting_lower")
    assert bracket.resolution == "uncertain"
    assert bracket.inside.q2 == 20.0
    assert bracket.outside.q2 == 10.0
    assert bracket.outside.regime is FlowRegime.THREADING
    assert bracket.q2_interval == (10.0, 20.0)
    assert any(
        gap["reason"] == "unknown_during_refinement"
        for gap in result.metadata["unresolved_intervals"]
    )
    assert "jetting_lower" not in station_boundaries(result).get(cfg.center_q1, {})
    assert result.metadata["station_status"][0]["jetting_lower"] == "uncertain"


@pytest.mark.parametrize("search_rounds, discovered", [(0, False), (1, True), (2, True)])
def test_window_search_discovers_narrow_target_between_identical_non_targets(
    config, search_rounds, discovered,
) -> None:
    cfg = replace(
        config, center_q2=10.0, q2_min=5.0, q2_max=20.0, step_ratio=2.0,
        max_expand_steps=1, window_search_rounds=search_rounds,
    )
    measure = MeasuredTopology(
        lambda _q1, q2: FlowRegime.JETTING
        if 13.5 < q2 < 15.0 else FlowRegime.THREADING
    )
    result = SpiderScanner(measure, cfg).run()
    assert result.center_node.regime is FlowRegime.THREADING
    assert next(n for n in result.nodes if n.q2 == 20.0).regime is FlowRegime.THREADING
    assert any(n.regime is FlowRegime.JETTING for n in result.nodes) is discovered
    if discovered:
        assert any(n.purpose == "window_search" for n in result.nodes)
        assert set(station_boundaries(result)[cfg.center_q1]) == {
            "jetting_lower", "jetting_upper",
        }
    else:
        assert not result.brackets


@pytest.mark.parametrize("safety_ratio, q2_max", [(3.0, 100.0), (2.0, 25.0)])
def test_target_reaching_configured_safe_domain_is_censored(
    config, safety_ratio, q2_max,
) -> None:
    cfg = replace(config, q2_min=10.0, q2_max=q2_max, safety_q1_over_q2=safety_ratio)
    result = SpiderScanner(constant_measurement(FlowRegime.JETTING), cfg).run()
    status, = result.metadata["station_status"]
    assert status["safe_q2_range"] == [10.0, min(q2_max, cfg.center_q1 / safety_ratio)]
    assert status["jetting_lower"] == "censored_by_safe_domain"
    assert status["jetting_upper"] == "censored_by_safe_domain"
    assert not result.brackets  # A safe-domain edge is not an observed transition.


@pytest.mark.parametrize(
    "regime, expected_status, unresolved",
    [
        (FlowRegime.UNKNOWN, "unknown_observations", True),
        (FlowRegime.THREADING, "not_observed_at_search_resolution", False),
    ],
)
def test_unknown_observations_are_not_claimed_to_be_absent_boundaries(
    config, regime, expected_status, unresolved,
) -> None:
    result = SpiderScanner(constant_measurement(regime), config).run()
    assert not result.brackets
    assert not station_boundaries(result)
    status, = result.metadata["station_status"]
    assert status["jetting_lower"] == expected_status
    assert status["jetting_upper"] == expected_status
    assert not status["target_observed"]
    assert bool(result.metadata["unresolved_intervals"]) is unresolved
    if unresolved:
        assert all(
            entry["reason"] == "unknown_observation"
            for entry in result.metadata["unresolved_intervals"]
        )


@pytest.mark.parametrize("budget_kind", ["nodes", "time"])
def test_scanner_stops_at_node_or_complete_node_time_budget(config, budget_kind) -> None:
    per_node = config.startup_s + config.node_duration_s
    cfg = replace(config, max_nodes=2) if budget_kind == "nodes" else replace(
        config, max_duration_s=2 * per_node,
    )
    measure = constant_measurement(FlowRegime.THREADING)
    result = SpiderScanner(measure, cfg).run()
    assert len(result.nodes) == len(measure.calls) == 2
    assert result.stopped_reason
    assert result.total_duration_s == pytest.approx(2 * per_node)
    assert result.total_duration_s <= cfg.max_duration_s
    assert result.metadata["station_status"][0]["jetting_upper"] == "budget_exhausted"


def test_scanner_never_emits_unsafe_or_duplicate_nodes_across_stations(config) -> None:
    cfg = replace(
        config, center_q1=90.0, center_q2=20.0, safety_q1_over_q2=3.0,
        q1_min=2.0, q1_max=180.0, q2_min=10.0, q2_max=60.0,
        station_step_ratio=3.0, max_stations=5, max_expand_steps=6,
        max_nodes=100, max_duration_s=200.0,
    )
    measure = constant_measurement(FlowRegime.THREADING)
    result = SpiderScanner(measure, cfg).run()
    assert len(result.nodes) == len(measure.calls)
    assert len({(n.q1, n.q2) for n in result.nodes}) == len(result.nodes)
    assert [n.index for n in result.nodes] == list(range(1, len(result.nodes) + 1))
    for node in result.nodes:
        assert cfg.q1_min <= node.q1 <= cfg.q1_max
        assert cfg.q2_min <= node.q2 <= cfg.q2_max
        assert node.q1 >= cfg.safety_q1_over_q2 * node.q2
    unsafe_stations = [
        row for row in result.metadata["station_status"]
        if row["safe_q2_range"][1] < cfg.q2_min
    ]
    assert unsafe_stations
    assert all(row["nodes"] == 0 for row in unsafe_stations)
    assert all(row["jetting_upper"] == "outside_safe_domain" for row in unsafe_stations)


@pytest.mark.parametrize(
    "field, value",
    [("q1", 120.0), ("q2", 30.0), ("q1", float("nan")),
     ("window_s", 0.5), ("window_s", float("nan"))],
)
def test_strategy_rejects_measurements_from_other_flows_or_windows(
    config, measured_topology, field, value,
) -> None:
    def measure(q1, q2, window_s):
        return replace(measured_topology(q1, q2, window_s), **{field: value})

    with pytest.raises(ValueError):
        SpiderScanner(measure, config).run()


class StubPump:
    """Runner protocol only: deliberately cannot open any real device."""

    def __init__(self, *, apply_error=None, stop_error=None, clock=None, delay_s=0.0):
        self.apply_error = apply_error
        self.stop_error = stop_error
        self.clock = clock
        self.delay_s = delay_s
        self.applied: list[tuple[float, float]] = []
        self.stop_calls = 0

    def apply(self, q1, q2):
        self.applied.append((q1, q2))
        if self.apply_error is not None:
            raise self.apply_error
        if self.clock is not None:
            self.clock.sleep(self.delay_s)
        return q1, q2

    def stop(self):
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error


class WindowMeasurement:
    def __init__(self, *, error=None, fraction=1.0):
        self.topology = MeasuredTopology()
        self.error = error
        self.fraction = fraction
        self.starts: list[float] = []

    def features(self, q1, q2, *, window_s, clock):
        self.starts.append(clock.now())
        if self.error is not None:
            raise self.error
        clock.sleep(window_s * self.fraction)
        return self.topology(q1, q2, window_s)


def test_runner_virtual_clock_includes_every_startup_and_full_window(config) -> None:
    cfg = replace(config, max_nodes=3)
    clock = VirtualClock(start=123.0)
    pump = StubPump()
    measurement = WindowMeasurement()
    result = ExperimentRunner(pump, measurement, clock, scan_config=cfg).run()
    per_node = cfg.startup_s + cfg.node_duration_s
    assert len(result.nodes) == len(pump.applied) == 3
    assert measurement.starts == pytest.approx([
        123.0 + cfg.startup_s + i * per_node for i in range(3)
    ])
    assert [node.started_s for node in result.nodes] == pytest.approx([
        cfg.startup_s + i * per_node for i in range(3)
    ])
    assert result.total_duration_s == pytest.approx(3 * per_node)
    assert result.metadata["actual_elapsed_s"] == pytest.approx(result.total_duration_s)
    assert clock.now() == pytest.approx(123.0 + 3 * per_node)
    assert result.metadata["clock_kind"] == "VirtualClock"
    assert result.metadata["status"] == "partial"
    assert result.metadata["stop_call_succeeded"] is True
    assert pump.stop_calls == 1
    for node in result.nodes:
        extra = node.features.extra
        assert extra["observe_end_monotonic"] - extra["observe_start_monotonic"] == pytest.approx(cfg.node_duration_s)
        assert extra["parameter_readback_q1"] == node.q1
        assert extra["parameter_readback_q2"] == node.q2


def test_runner_completed_scan_reports_actual_and_planned_duration(config) -> None:
    cfg = replace(config, max_expand_steps=1, bisection_rounds=0)
    pump = StubPump()
    clock = VirtualClock()
    result = ExperimentRunner(pump, WindowMeasurement(), clock, scan_config=cfg).run()
    assert not result.stopped_reason
    assert result.metadata["status"] == "completed"
    assert result.metadata["actual_elapsed_s"] == pytest.approx(
        len(result.nodes) * (cfg.startup_s + cfg.node_duration_s)
    )
    assert result.total_duration_s == pytest.approx(result.metadata["actual_elapsed_s"])
    assert pump.stop_calls == 1


def test_runner_distinguishes_actual_pump_overhead_from_planned_time(config) -> None:
    cfg = replace(config, max_nodes=2)
    clock = VirtualClock(50.0)
    pump = StubPump(clock=clock, delay_s=0.1)
    result = ExperimentRunner(pump, WindowMeasurement(), clock, scan_config=cfg).run()
    assert len(result.nodes) == 2
    planned = 2 * (cfg.startup_s + cfg.node_duration_s)
    assert result.total_duration_s == pytest.approx(planned)
    assert result.metadata["actual_elapsed_s"] == pytest.approx(planned + 0.2)
    assert pump.stop_calls == 1


def test_simulated_measurement_advances_clock_for_whole_window() -> None:
    class SimulatorStub:
        def features(self, q1, q2, *, window_s):
            return MeasuredTopology()(q1, q2, window_s)

    clock = VirtualClock(17.0)
    measurement = SimulatedMeasurement(simulator=SimulatorStub())
    features = measurement.features(90.0, 20.0, window_s=0.5, clock=clock)
    assert clock.now() == 17.5
    assert features.window_s == 0.5
    assert classify_regime(features)[0] is FlowRegime.JETTING


@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_runner_rejects_measurement_returning_before_full_window(config, fraction) -> None:
    pump = StubPump()
    runner = ExperimentRunner(
        pump, WindowMeasurement(fraction=fraction), VirtualClock(), scan_config=config,
    )
    with pytest.raises(ValueError, match="观测窗口"):
        runner.run()
    assert pump.stop_calls == 1
    assert runner.scanner is not None
    assert runner.scanner._nodes == []


@pytest.mark.parametrize("failure_site", ["measurement", "pump"])
@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
def test_runner_always_stops_after_failure_or_keyboard_interrupt(
    config, failure_site, exception_type,
) -> None:
    failure = exception_type("original failure")
    pump = StubPump(apply_error=failure if failure_site == "pump" else None)
    measurement = WindowMeasurement(error=failure if failure_site == "measurement" else None)
    with pytest.raises(exception_type) as caught:
        ExperimentRunner(pump, measurement, VirtualClock(), scan_config=config).run()
    assert caught.value is failure
    assert pump.stop_calls == 1
    assert len(pump.applied) == 1
    assert bool(measurement.starts) is (failure_site == "measurement")


def test_runner_surfaces_stop_failure_after_otherwise_successful_scan(config) -> None:
    failure = RuntimeError("stop acknowledgement missing")
    pump = StubPump(stop_error=failure)
    logs = []
    with pytest.raises(RuntimeError) as caught:
        ExperimentRunner(
            pump, WindowMeasurement(), VirtualClock(), logger=logs.append,
            scan_config=replace(config, max_nodes=1),
        ).run()
    assert caught.value is failure
    assert pump.stop_calls == 1
    assert any("[SAFETY][ERROR]" in line for line in logs)
    assert not any("停机调用已成功返回" in line for line in logs)


@pytest.mark.parametrize("failure_site", ["measurement", "pump"])
@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
def test_stop_failure_does_not_replace_original_exception(
    config, failure_site, exception_type,
) -> None:
    primary = exception_type("original failure")
    cleanup = RuntimeError("stop acknowledgement missing")
    pump = StubPump(
        apply_error=primary if failure_site == "pump" else None, stop_error=cleanup,
    )
    measurement = WindowMeasurement(error=primary if failure_site == "measurement" else None)
    logs = []
    with pytest.raises(exception_type) as caught:
        ExperimentRunner(
            pump, measurement, VirtualClock(), logger=logs.append, scan_config=config,
        ).run()
    assert caught.value is primary
    assert pump.stop_calls == 1
    notes = getattr(primary, "__notes__", [])
    cleanup_detail = " ".join(notes) + str(getattr(primary, "cleanup_error", ""))
    assert "stop acknowledgement missing" in cleanup_detail
    assert any("[SAFETY][ERROR]" in line for line in logs)


@pytest.mark.parametrize("overrun_site", ["pump", "measurement"])
def test_runner_actual_time_budget_rejects_unfinishable_or_overrun_node(
    config, overrun_site,
) -> None:
    cfg = replace(config, max_duration_s=config.startup_s + config.node_duration_s)
    clock = VirtualClock()
    pump = StubPump(clock=clock, delay_s=0.1 if overrun_site == "pump" else 0.0)
    measurement = WindowMeasurement(fraction=1.1 if overrun_site == "measurement" else 1.0)
    runner = ExperimentRunner(pump, measurement, clock, scan_config=cfg)
    with pytest.raises(TimeoutError):
        runner.run()
    assert pump.stop_calls == 1
    assert bool(measurement.starts) is (overrun_site == "measurement")
    assert runner.scanner._nodes == []


def test_runner_budget_overrun_after_completed_nodes_returns_partial(config) -> None:
    """P1回归：预算在第二节点前耗尽时必须交回已完成的partial，而不是丢结果。"""
    per_node = config.startup_s + config.node_duration_s
    cfg = replace(config, max_duration_s=2 * per_node + 0.05, max_nodes=3)
    clock = VirtualClock()
    pump = StubPump(clock=clock, delay_s=0.1)
    runner = ExperimentRunner(pump, WindowMeasurement(), clock, scan_config=cfg)
    result = runner.run()
    assert len(result.nodes) == len(pump.applied) == 1
    assert result.stopped_reason  # 预算结束必须显式说明未覆盖区域未知
    assert result.metadata["status"] == "partial"
    assert result.metadata["stop_call_succeeded"] is True
    assert pump.stop_calls == 1
    assert result.nodes[0].features.extra["parameter_readback_q1"] == result.nodes[0].q1


def test_runner_zero_completed_nodes_budget_is_clear_failure(config) -> None:
    """预算连首节点都装不下时不能伪装partial：明确失败并尝试停泵。"""
    cfg = replace(config, max_duration_s=config.node_duration_s)
    clock = VirtualClock()
    pump = StubPump(clock=clock, delay_s=0.1)
    runner = ExperimentRunner(pump, WindowMeasurement(), clock, scan_config=cfg)
    with pytest.raises(RuntimeError, match="未产生有效观测"):
        runner.run()
    assert pump.stop_calls == 1
    assert runner.scanner._nodes == []


@pytest.mark.parametrize(
    "field",
    ["center_q1", "center_q2", "node_duration_s", "startup_s", "step_ratio",
     "min_ratio_step", "q1_min", "q1_max", "q2_min", "q2_max",
     "station_step_ratio", "safety_q1_over_q2", "max_duration_s"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_config_rejects_nonfinite_numeric_limits(config, field, value) -> None:
    with pytest.raises(ValueError):
        replace(config, **{field: value}).validate()


@pytest.mark.parametrize("ratio", [0.0, 1.0, 1.99])
def test_config_cannot_relax_fixed_pump_safety_ratio(config, ratio) -> None:
    with pytest.raises(ValueError):
        replace(config, safety_q1_over_q2=ratio).validate()


def test_config_rejects_center_unsafe_under_stricter_configured_ratio(config) -> None:
    with pytest.raises(ValueError):
        replace(config, safety_q1_over_q2=5.0).validate()


@pytest.mark.parametrize("target", ["threading", "unknown", ""])
def test_config_rejects_unsupported_scan_targets(config, target) -> None:
    with pytest.raises(ValueError):
        replace(config, target_regime=target).validate()
