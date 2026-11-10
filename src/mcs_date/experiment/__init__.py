"""实验层：流型定义、芯片仿真、蜘蛛式扫描、结果持久化与动画。"""

from __future__ import annotations

from .animate import DiagramConfig, DiagramRenderer, AnimationArtifacts, animate_scan
from .calibration import (
    CalibrationResult,
    DEFAULT_HISTORY_ROOT,
    learn_calibration,
    summarize_calibration,
)
from .chip import ChipFluids, ChipModelConfig, ChipSimulator
from .models import (
    BoundaryBracket,
    NodeRecord,
    ScanConfig,
    ScanResult,
    classify_boundary_kind,
)
from .phase import (
    FlowRegime,
    NodeFeatures,
    RegimeThresholds,
    assess_formation,
    classify_regime,
    merge_features,
)
from .pilot import (
    PilotConfig,
    PilotPoint,
    PilotScanner,
    build_pilot_points,
    pilot_scan_config,
    run_offline_pilot,
)
from .report import load_result, result_from_dict, result_to_dict, save_nodes_csv, save_result
from .runner import (
    Clock,
    ExperimentRunner,
    RealClock,
    SimulatedMeasurement,
    VirtualClock,
    run_offline_scan,
)
from .strategy import SpiderScanner, station_boundaries, summarize_regimes

__all__ = [
    "AnimationArtifacts",
    "BoundaryBracket",
    "ChipFluids",
    "ChipModelConfig",
    "ChipSimulator",
    "CalibrationResult",
    "Clock",
    "DEFAULT_HISTORY_ROOT",
    "DiagramConfig",
    "DiagramRenderer",
    "ExperimentRunner",
    "FlowRegime",
    "NodeFeatures",
    "NodeRecord",
    "PilotConfig",
    "PilotPoint",
    "PilotScanner",
    "RealClock",
    "RegimeThresholds",
    "ScanConfig",
    "ScanResult",
    "SimulatedMeasurement",
    "SpiderScanner",
    "VirtualClock",
    "animate_scan",
    "assess_formation",
    "build_pilot_points",
    "classify_boundary_kind",
    "classify_regime",
    "learn_calibration",
    "load_result",
    "merge_features",
    "pilot_scan_config",
    "result_from_dict",
    "result_to_dict",
    "run_offline_pilot",
    "run_offline_scan",
    "save_nodes_csv",
    "save_result",
    "station_boundaries",
    "summarize_regimes",
    "summarize_calibration",
]
