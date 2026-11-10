"""命令行入口测试（全部走离线/合成路径，不依赖相机与串口）。"""

from __future__ import annotations

from mcs_date.__main__ import main


def test_no_arguments_prints_help(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "usage" in output.lower()


def test_version_flag_exits_zero() -> None:
    try:
        main(["--version"])
    except SystemExit as exc:  # argparse 以 SystemExit(0) 结束
        assert exc.code == 0
    else:  # pragma: no cover - argparse 实现变化时的兜底
        raise AssertionError("--version 应触发 SystemExit")


def test_pump_subcommand_uses_simulator(capsys) -> None:
    assert main(["pump", "--q1", "80", "--q2", "20"]) == 0
    output = capsys.readouterr().out
    assert "readback" in output
    assert "80" in output


def test_vision_subcommand_with_synthetic_frames(tmp_path, capsys) -> None:
    assert main(["vision", "--synthetic", "--max-frames", "3", "--out", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert '"frames": 3' in output
    assert (tmp_path / "vision_preview.png").exists()


def test_scan_subcommand_without_animation(tmp_path, capsys) -> None:
    assert main(
        [
            "scan",
            "--node-seconds",
            "2",
            "--no-animate",
            "--out",
            str(tmp_path),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert '"nodes"' in output
    assert (tmp_path / "scan_result.json").exists()
    assert (tmp_path / "nodes.csv").exists()
