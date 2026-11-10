from __future__ import annotations

from dataclasses import replace

import pytest

from mcs_date.pump import protocol, service
from mcs_date.pump.client import PDUReply, PumpClientError
from mcs_date.pump.models import ChannelParams
from mcs_date.pump.service import (
    PumpController,
    PumpError,
    PumpSafetyError,
    SerialPumpBackend,
    SimulatedPumpBackend,
    channel_params_for_flow,
    flow_from_channel_params,
)


def rsp(params: ChannelParams) -> bytes:
    return b"RSP" + protocol.pdu_wsp(
        channel=params.channel,
        mode=params.mode,
        syringe_code=params.syringe_code,
        dispense_value=params.dispense_value,
        dispense_unit=params.dispense_unit,
        infuse_time_value=params.infuse_time_value,
        infuse_time_unit=params.infuse_time_unit,
        withdraw_time_value=params.withdraw_time_value,
        withdraw_time_unit=params.withdraw_time_unit,
        repeat_count=params.repeat_count,
        interval_value=params.interval_value,
    )[3:]


class FakeClient:
    """Only fake PDUs; no serial transport is constructed or opened."""

    def __init__(self) -> None:
        self.open = False
        self.disconnects = 0
        self.connect_error: Exception | None = None
        self.rss_reply: bytes | None = b"RSS\x03\x00" + b"\x00\x00\x00" * 4
        self.rse_replies: list[bytes | None | Exception] = []
        self.run_state = 0
        self.requests: list[tuple[str, bytes, int | None]] = []
        self.params: dict[int, bytes] = {}
        self.readback_overrides: dict[int, bytes] = {}

    def connect(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.open = True

    def disconnect(self) -> None:
        self.disconnects += 1
        self.open = False

    def is_connected(self) -> bool:
        return self.open

    def send_pdu(
        self, pdu: bytes, expect_cmd: str | None = None,
        allow_no_reply: bool = False, retries: int | None = None,
    ) -> PDUReply | None:
        assert self.open, "The backend must not query a disconnected transport"
        command = pdu[:3].decode("ascii")
        assert command == expect_cmd
        self.requests.append((command, pdu, retries))
        reply: bytes | None | Exception
        if command == "RSS":
            reply = self.rss_reply
        elif command == "WSE":
            self.run_state = pdu[3]
            reply = None
        elif command == "RSE":
            if self.rse_replies:
                reply = self.rse_replies[0]
                if len(self.rse_replies) > 1:
                    self.rse_replies.pop(0)
            else:
                reply = b"RSE" + bytes([self.run_state, 0])
        elif command == "WSP":
            self.params[pdu[3]] = b"RSP" + pdu[3:]
            reply = None
        elif command == "RSP":
            reply = self.readback_overrides.get(pdu[3], self.params[pdu[3]])
        else:
            assert command == "WSS"
            reply = None
        if isinstance(reply, Exception):
            raise reply
        if reply is None:
            assert allow_no_reply or command in {"RSS", "RSE"}
            return None
        return PDUReply(raw_frame=b"", cmd=command, pdu=reply, addr=1)


@pytest.fixture
def serial_backend(monkeypatch: pytest.MonkeyPatch) -> tuple[SerialPumpBackend, FakeClient]:
    client = FakeClient()
    monkeypatch.setattr(service, "PumpClient", lambda *_args: client)
    monkeypatch.setattr(service.time, "sleep", lambda _seconds: None)
    backend = SerialPumpBackend()
    backend.connect()
    client.requests.clear()
    return backend, client


@pytest.mark.parametrize("sys_state", [1, 7, 2, 4, 8, 16, 31])
def test_stop_rejects_system_or_any_channel_still_running(
    serial_backend: tuple[SerialPumpBackend, FakeClient], sys_state: int,
) -> None:
    backend, client = serial_backend
    client.rse_replies = [b"RSE" + bytes([sys_state, 0])]
    with pytest.raises(PumpSafetyError, match="仍在运行"):
        backend.stop()
    assert [cmd for cmd, _, _ in client.requests] == ["WSE", "RSE"] * 3
    assert all(retries == 1 for _, _, retries in client.requests)


@pytest.mark.parametrize("reply", [None, b"", b"RSE", b"RSE\x20\x00", b"RSE\x00\x01",
                                  PumpClientError("RSE lost")])
def test_stop_rejects_missing_malformed_or_unknown_readback(
    serial_backend: tuple[SerialPumpBackend, FakeClient], reply: bytes | None | Exception,
) -> None:
    backend, client = serial_backend
    client.rse_replies = [reply]
    with pytest.raises(PumpSafetyError, match="停泵未确认") as caught:
        backend.stop()
    assert caught.value.__cause__ is not None
    assert len(client.requests) == 6


def test_stop_eventually_confirms_all_four_channels_stopped(
    serial_backend: tuple[SerialPumpBackend, FakeClient],
) -> None:
    backend, client = serial_backend
    client.rse_replies = [b"RSE\x07\x00", b"RSE\x10\x00", b"RSE\x00\x00"]
    backend.stop()
    assert len(client.requests) == 6


def test_disconnected_stop_never_claims_success(
    serial_backend: tuple[SerialPumpBackend, FakeClient],
) -> None:
    backend, client = serial_backend
    client.open = False
    with pytest.raises(PumpSafetyError, match="未连接"):
        backend.stop()
    assert client.requests == []


def test_connection_lost_during_stop_cannot_be_reported_as_stopped(
    serial_backend: tuple[SerialPumpBackend, FakeClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, client = serial_backend
    send = client.send_pdu

    def lose_connection(
        pdu: bytes, expect_cmd: str | None = None,
        allow_no_reply: bool = False, retries: int | None = None,
    ) -> PDUReply | None:
        if expect_cmd == "RSE":
            client.open = False
            raise PumpClientError("transport disconnected")
        return send(pdu, expect_cmd, allow_no_reply, retries)

    monkeypatch.setattr(client, "send_pdu", lose_connection)
    with pytest.raises(PumpSafetyError, match="未连接"):
        backend.stop()
    assert not backend.connected
    assert [cmd for cmd, _, _ in client.requests] == ["WSE"]


def test_query_forwards_explicit_retries(
    serial_backend: tuple[SerialPumpBackend, FakeClient],
) -> None:
    backend, client = serial_backend
    backend._query("RSE", protocol.pdu_rse(), retries=4)
    assert client.requests[-1][2] == 4


@pytest.mark.parametrize("reply", [b"RSE\x10\x00", None, b"malformed",
                                  PumpClientError("RSE unavailable")])
def test_apply_never_writes_parameters_before_verified_stop(
    serial_backend: tuple[SerialPumpBackend, FakeClient], reply: bytes | None | Exception,
) -> None:
    backend, client = serial_backend
    client.rse_replies = [reply]
    with pytest.raises(PumpSafetyError):
        backend.apply(80.0, 20.0)
    assert not any(cmd in {"WSP", "WSS"} for cmd, _, _ in client.requests)


def test_apply_writes_and_starts_only_after_stopped_readback(
    serial_backend: tuple[SerialPumpBackend, FakeClient],
) -> None:
    backend, client = serial_backend
    backend.apply(80.0, 20.0)
    commands = [cmd for cmd, _, _ in client.requests]
    assert commands[:2] == ["WSE", "RSE"]
    assert commands.index("WSP") > commands.index("RSE")
    assert client.run_state == 7
    assert backend.read() == pytest.approx((80.0, 20.0))


def test_serial_rejects_unsafe_quantization_before_writes(
    serial_backend: tuple[SerialPumpBackend, FakeClient],
) -> None:
    backend, client = serial_backend
    # Requested ratio is exactly 2, but independently rounded times violate it.
    assert 12.26 == 2 * 6.13
    with pytest.raises(PumpSafetyError, match="Q1>=2"):
        backend.apply(12.26, 6.13)
    assert not any(cmd == "WSP" for cmd, _, _ in client.requests)
    assert client.run_state == 0


def test_serial_checks_actual_readback_ratio_before_starting(
    serial_backend: tuple[SerialPumpBackend, FakeClient],
) -> None:
    backend, client = serial_backend
    client.readback_overrides[1] = rsp(replace(
        channel_params_for_flow(1, 80.0), dispense_value=9995, dispense_unit=3,
    ))
    with pytest.raises(PumpSafetyError, match="Q1>=2"):
        backend.apply(80.0, 40.0)
    assert set(client.params) == {1, 2}
    assert client.run_state == 0
    assert not any(cmd == "WSE" and pdu[3] != 0 for cmd, pdu, _ in client.requests)


@pytest.mark.parametrize("q1,q2", [(float("nan"), 1.0), (10.0, float("nan")),
                                   (float("inf"), 1.0), (10.0, float("inf")),
                                   (-2.0, -1.0), (10.0, -1.0), (0.0, 0.0)])
def test_all_entrypoints_reject_nonfinite_or_nonpositive_flows(
    serial_backend: tuple[SerialPumpBackend, FakeClient], q1: float, q2: float,
) -> None:
    backend, client = serial_backend
    sim = SimulatedPumpBackend()
    sim.connect()
    for pump in (backend, sim, PumpController(sim)):
        with pytest.raises(PumpSafetyError):
            pump.apply(q1, q2)
    assert client.requests == []
    assert sim.read() == (0.0, 0.0)


@pytest.mark.parametrize("flow", [float("nan"), float("inf"), -1.0, 0.0, 1e-300, 1e300])
def test_parameter_generation_rejects_invalid_or_unrepresentable_flow(flow: float) -> None:
    with pytest.raises(PumpSafetyError):
        channel_params_for_flow(1, flow)


@pytest.mark.parametrize("flow", [0.5, 0.001, 1e-5, 1e-7, 0.001 / (9999 * 6),
                                  20.0, 80.0, 180.0, 9999 * 10 / (10 / 600)])
def test_long_durations_and_low_flows_are_represented_not_clamped(flow: float) -> None:
    params = channel_params_for_flow(1, flow)
    assert 1 <= params.dispense_value <= 9999
    assert 10 <= params.infuse_time_value <= 9999
    assert flow_from_channel_params(params) == pytest.approx(flow, rel=0.002, abs=0.0)
    if flow == 0.5:
        assert params.dispense_value < 1000


@pytest.mark.parametrize("q1,q2", [(0.03, 0.015), (0.002, 0.001)])
def test_simulator_rejects_unsafe_or_zero_quantized_flows(q1: float, q2: float) -> None:
    sim = SimulatedPumpBackend()
    sim.connect()
    with pytest.raises(PumpSafetyError):
        sim.apply(q1, q2)
    assert sim.read() == (0.0, 0.0)


def test_readback_flow_must_be_finite_and_positive() -> None:
    params = channel_params_for_flow(1, 80.0)
    for bad in (replace(params, dispense_value=0), replace(params, infuse_time_value=0),
                replace(params, dispense_unit=99)):
        with pytest.raises(PumpSafetyError):
            flow_from_channel_params(bad)


@pytest.mark.parametrize("identity_reply", [None, b"bad RSS"])
def test_connect_identity_failure_releases_serial_handle(
    monkeypatch: pytest.MonkeyPatch, identity_reply: bytes | None,
) -> None:
    client = FakeClient()
    client.rss_reply = identity_reply
    monkeypatch.setattr(service, "PumpClient", lambda *_args: client)
    backend = SerialPumpBackend()
    with pytest.raises((PumpError, ValueError)):
        backend.connect()
    assert client.disconnects == 1
    assert not backend.connected
    assert backend._setup is None


def test_cannot_connect_cleans_up_and_stop_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    error = PumpClientError("cannot connect")
    client.connect_error = error
    monkeypatch.setattr(service, "PumpClient", lambda *_args: client)
    monkeypatch.setattr(service.time, "sleep", lambda _seconds: None)
    backend = SerialPumpBackend()
    with pytest.raises(PumpClientError) as caught:
        backend.connect()
    assert caught.value is error
    assert client.disconnects == 1
    with pytest.raises(PumpSafetyError):
        backend.stop()


def test_controller_apply_failure_attempts_stop_and_preserves_both_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = SimulatedPumpBackend()
    original = PumpError("parameter update failed")
    cleanup = PumpSafetyError("still running")
    stopped: list[bool] = []

    def fail_apply(_q1: float, _q2: float) -> None:
        raise original

    def fail_stop() -> None:
        stopped.append(True)
        raise cleanup

    monkeypatch.setattr(sim, "apply", fail_apply)
    monkeypatch.setattr(sim, "stop", fail_stop)
    controller = PumpController(sim)
    with pytest.raises(PumpError) as caught:
        controller.apply(80.0, 20.0)
    assert caught.value is original
    assert stopped == [True]
    assert getattr(original, "pump_cleanup_errors") == (cleanup,)
    assert controller.target is None


def test_controller_read_failure_after_apply_stops_pump(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = SimulatedPumpBackend()
    controller = PumpController(sim)
    controller.connect()

    def fail_read() -> tuple[float, float]:
        raise PumpError("readback lost")

    monkeypatch.setattr(sim, "read", fail_read)
    with pytest.raises(PumpError, match="readback lost"):
        controller.apply(80.0, 20.0)
    assert sim._q1 == sim._q2 == 0.0
    assert controller.target is None


def test_controller_rejects_unsafe_actual_readback_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = SimulatedPumpBackend()
    controller = PumpController(sim)
    controller.connect()
    monkeypatch.setattr(sim, "read", lambda: (79.99, 40.0))
    with pytest.raises(PumpSafetyError):
        controller.apply(80.0, 40.0)
    assert sim._q1 == sim._q2 == 0.0
    assert controller.target is None


def test_close_releases_handle_without_hiding_stop_failure(
    serial_backend: tuple[SerialPumpBackend, FakeClient],
) -> None:
    backend, client = serial_backend
    client.rse_replies = [b"RSE\x01\x00"]
    with pytest.raises(PumpSafetyError, match="仍在运行"):
        PumpController(backend).close()
    assert client.disconnects == 1
    assert not backend.connected


def test_close_preserves_stop_and_resource_release_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = SimulatedPumpBackend()
    stop_error = PumpSafetyError("cannot verify stop")
    close_error = PumpError("release failed")
    released: list[bool] = []

    def fail_stop() -> None:
        raise stop_error

    def fail_close() -> None:
        released.append(True)
        raise close_error

    monkeypatch.setattr(sim, "stop", fail_stop)
    monkeypatch.setattr(sim, "close", fail_close)
    with pytest.raises(PumpSafetyError) as caught:
        PumpController(sim).close()
    assert caught.value is stop_error
    assert released == [True]
    assert getattr(stop_error, "pump_cleanup_errors") == (close_error,)


def test_close_releases_resources_even_if_stop_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = SimulatedPumpBackend()
    sim.connect()
    interrupt = KeyboardInterrupt("cancel during stop")

    def interrupt_stop() -> None:
        raise interrupt

    monkeypatch.setattr(sim, "stop", interrupt_stop)
    with pytest.raises(KeyboardInterrupt) as caught:
        PumpController(sim).close()
    assert caught.value is interrupt
    assert not sim.connected


def test_apply_interruption_stops_pump(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = SimulatedPumpBackend()
    controller = PumpController(sim)
    controller.connect()
    apply = sim.apply

    def interrupt_after_start(q1: float, q2: float) -> None:
        apply(q1, q2)
        raise KeyboardInterrupt("cancel after start")

    monkeypatch.setattr(sim, "apply", interrupt_after_start)
    with pytest.raises(KeyboardInterrupt):
        controller.apply(80.0, 20.0)
    assert sim.read() == (0.0, 0.0)
    assert controller.target is None


def test_cleanup_error_fallback_supports_python_310() -> None:
    class LegacyError(PumpError):
        add_note = None

    original = LegacyError("original")
    cleanup = PumpSafetyError("cleanup failed")
    service._attach_cleanup_error(original, "stop", cleanup)
    assert original.args[0] == "original"
    assert "cleanup failed" in str(original)
    assert getattr(original, "pump_cleanup_errors") == (cleanup,)
