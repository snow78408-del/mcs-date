"""历史数据标定：把 MCS_QT 的离线实验记录整理成“液滴形成窗口”参数。

设计原则
--------
* **只读**：所有历史数据（``E:/MCS_QT_Data``）只读，不做任何写入。
* **可缺省**：数据目录不存在或缺少某个文件时，退回内置先验并明确标注
  置信度；``mcs-date`` 的离线演示因此可以在任何机器上运行。
* **可追溯**：每条观测都带来源文件与置信度，写入 ``CalibrationResult.sources``，
  输出的 JSON 可直接进版本库/报告。

窗口模型（与 :mod:`mcs_date.experiment.chip` 一致）
---------------------------------------------------
用户 2026-11-10 的口述物理：油水流量比 ``r = Q1/Q2`` 太低时连续相切割
能力不足，太高时拖拽占优，只有中间一段能稳定形成液滴。因此二相图上
的“液滴区”是两条近似直线夹出的条带::

    Q2 = Q1 / r_low   （上边界：r 低于此值 → 栓塞/无滴）
    Q2 = Q1 / r_high  （下边界：r 高于此值 → 射流/束流）

``r_low`` 没有可用的“无滴”证据：泵安全约束要求 ``Q1 >= 2*Q2``（``r >= 2``），
而历史最低成滴点是 ``r = 2.5``（50/20，用户评价“液滴效果较理想”）。因此
``r_low = 2.2`` 只是**先验**（介于安全下限与最佳成滴点之间，可在 r∈[2,2.5]
做一次确认实验），置信度标为 ``prior``；``r_high`` 由“已确认成滴”与
“长结构/无滴”两类记录夹逼得到。
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
DEFAULT_HISTORY_ROOT = Path(os.environ.get("MCS_QT_DATA_DIR", r"E:\MCS_QT_Data"))

#: 一次“流量点 → 是否成滴”的历史观测。
#: label 取 ``droplet`` / ``no_droplet``；``confidence`` 取 high/medium/low。
FlowObservation = tuple[float, float, str, str, str]
#: (Q1, Q2, 液滴直径 µm, 来源) —— 尺寸观测。
SizeObservation = tuple[float, float, float, str]


def _curated_flow_observations() -> list[FlowObservation]:
    """人工整理的流型证据（来自 MCS_QT 报告/用户反馈，随仓库一起分发）。

    这些记录是“标定的锚点”，来源路径在 ``sources`` 里给出。注意历史上
    所有 ``session_summary.json`` 的 ``regime_survey`` 记录都是
    ``unknown/pending``，所以不能直接当标签用。
    """

    campaign = "experiment_20261016/campaign_analysis/report.md"
    scan = "experiments/regime_scan_20261008"
    junction = "experiments/experiment_20261016/junction_observation_20261016.md"
    feedback = "用户 2026-10-08 反馈（regime_scan 分析报告引用）"
    return [
        # -- 已确认成滴（10/16 campaign 的 12 个稳定段）-------------------
        (70.0, 20.0, "droplet", "high", campaign),
        (60.0, 20.0, "droplet", "high", campaign),
        (70.0, 25.0, "droplet", "high", campaign),
        (50.0, 20.0, "droplet", "high", campaign),
        (65.0, 22.0, "droplet", "high", campaign),
        (70.0, 18.0, "droplet", "high", campaign),
        # -- 用户主观“液滴效果较理想”：给了 r_low 的下方参照 -------------
        (50.0, 20.0, "droplet", "high", junction),
        # -- 非滴落证据：长结构 / 看不到完整液滴 --------------------------
        (90.09, 20.0, "no_droplet", "medium", scan),
        (100.0, 20.0, "no_droplet", "low", feedback),
    ]


#: 无法访问数据目录时使用的历史拟合常数（10/16 campaign report）。
FALLBACK_SIZE = (92.2242, -0.02273, -0.41219)


@dataclass(frozen=True)
class CalibrationResult:
    """一次历史标定的产物；可直接序列化为 JSON。"""

    ratio_low: float = 2.2
    ratio_high: float = 4.185
    ratio_thread: float = 6.0
    size_intercept_um: float = FALLBACK_SIZE[0]
    size_q1_coef_um_per_ulmin: float = FALLBACK_SIZE[1]
    size_q2_coef_um_per_ulmin: float = FALLBACK_SIZE[2]
    nominal_cross_section_um: float = 50.0
    measurement_scale_um_per_px: float = 2.2667
    ratio_low_confidence: str = "prior"
    ratio_high_confidence: str = "medium"
    ratio_thread_confidence: str = "prior"
    droplet_points: int = 0
    no_droplet_points: int = 0
    size_points: int = 0
    survey_records: int = 0
    survey_reviewed: int = 0
    sources: tuple[str, ...] = ()
    notes: str = ""

    # -- 窗口查询 -----------------------------------------------------------
    def regime_name(self, q1: float, q2: float) -> str:
        """返回 ``dripping`` / ``jetting`` / ``threading`` / ``tubing``。"""

        r = q1 / q2 if q2 > 0 else math.inf
        if r < self.ratio_low:
            return "tubing"
        if r <= self.ratio_high:
            return "dripping"
        if r <= self.ratio_thread:
            return "jetting"
        return "threading"

    def droplet_diameter_um(self, q1: float, q2: float) -> float:
        """历史尺寸拟合（仅液滴窗口内有意义）。"""

        value = (
            self.size_intercept_um
            + self.size_q1_coef_um_per_ulmin * (q1 - 60.0)
            + self.size_q2_coef_um_per_ulmin * (q2 - 20.0)
        )
        return max(1.0, value)

    def droplet_q1_window(self, q2: float) -> tuple[float, float]:
        """给定 Q2，液滴窗口覆盖的 Q1 区间。"""

        return (self.ratio_low * q2, self.ratio_high * q2)

    # -- 序列化 -------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        data["sources"] = list(self.sources)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationResult":
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "sources" in payload:
            payload["sources"] = tuple(payload["sources"])
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 历史数据读取（全部只读、容错）
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect_campaign_size(root: Path = DEFAULT_HISTORY_ROOT) -> list[SizeObservation]:
    """读取 10/16 campaign 的 12 段液滴尺寸观测。"""

    path = root / "experiments" / "experiment_20261016" / "campaign_analysis" / "fit_results.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return []
    out: list[SizeObservation] = []
    for stage in data.get("stages", []):
        q1, q2 = stage.get("q1"), stage.get("q2")
        quantiles = stage.get("D_quantiles")
        if not isinstance(q1, (int, float)) or not isinstance(q2, (int, float)):
            continue
        if not isinstance(quantiles, list) or len(quantiles) < 2:
            continue
        out.append((float(q1), float(q2), float(quantiles[1]), str(path.name)))
    return out


def collect_response_library(root: Path = DEFAULT_HISTORY_ROOT) -> list[SizeObservation]:
    """读取响应库 45 条实验的首末流量与液滴长度中位数（仅作覆盖统计）。"""

    path = root / "models" / "response_library.sqlite3"
    if not path.exists():
        return []
    out: list[SizeObservation] = []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            rows = con.execute("select payload from experiments").fetchall()
    except sqlite3.Error:
        return []
    for (payload,) in rows:
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        lengths = [x for x in (data.get("lengths_px") or []) if isinstance(x, (int, float))]
        if not lengths:
            continue
        lengths.sort()
        q_after = data.get("q_after")
        if not isinstance(q_after, list) or len(q_after) < 2:
            continue
        out.append((float(q_after[0]), float(q_after[1]), float(lengths[len(lengths) // 2]), "response_library.sqlite3"))
    return out


def collect_session_survey(root: Path = DEFAULT_HISTORY_ROOT, *, limit: int = 400) -> tuple[int, int]:
    """汇总 ``session_summary.json`` 的流型记录数/已复核数（全部为未复核）。"""

    total = reviewed = 0
    experiments = root / "experiments"
    if not experiments.exists():
        return (0, 0)
    for index, path in enumerate(experiments.rglob("session_summary.json")):
        if index >= limit:
            break
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        survey = data.get("regime_survey") or {}
        for record in survey.get("records") or []:
            total += 1
            if str(record.get("review_status", "")).lower() in {"reviewed", "confirmed"}:
                reviewed += 1
    return (total, reviewed)


def _fit_size(points: Iterable[SizeObservation]) -> tuple[float, float, float]:
    """对尺寸观测做最小二乘 ``D = a + b*(Q1-60) + c*(Q2-20)``。"""

    rows = [(q1, q2, d) for q1, q2, d, _src in points]
    if len(rows) < 3:
        return FALLBACK_SIZE
    try:
        import numpy as np

        a = np.array([[1.0, q1 - 60.0, q2 - 20.0] for q1, q2, _ in rows], dtype=float)
        b = np.array([d for _, _, d in rows], dtype=float)
        coef, *_ = np.linalg.lstsq(a, b, rcond=None)
        return (float(coef[0]), float(coef[1]), float(coef[2]))
    except (ImportError, ValueError):
        return FALLBACK_SIZE


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def learn_calibration(
    history_root: str | Path | None = None,
    *,
    extra_observations: Iterable[FlowObservation] = (),
) -> CalibrationResult:
    """汇总历史证据，学习液滴窗口与尺寸模型。

    数据目录缺失时返回内置先验（置信度标为 ``prior``），不会抛异常。
    """

    root = Path(history_root) if history_root is not None else DEFAULT_HISTORY_ROOT
    observations = list(_curated_flow_observations()) + list(extra_observations)
    sources: list[str] = []
    notes: list[str] = []

    if root.exists():
        sources.append(str(root))
        sizes = collect_campaign_size(root)
        if sizes:
            sources.append(str(root / "experiments" / "experiment_20261016" / "campaign_analysis" / "fit_results.json"))
        library = collect_response_library(root)
        if library:
            sources.append(str(root / "models" / "response_library.sqlite3"))
        total, reviewed = collect_session_survey(root)
        if total:
            sources.append(str(root / "experiments" / "*" / "session_summary.json"))
            notes.append(f"session_summary 共 {total} 条流型记录，已复核 {reviewed} 条")
            if reviewed == 0:
                notes.append("历史 session 的 regime_survey 全为 unknown/pending，未当作标签使用")
    else:
        sizes, library, total, reviewed = [], [], 0, 0
        notes.append(f"历史数据目录不存在：{root}，使用内置先验")

    positives = [(q1, q2) for q1, q2, label, _c, _s in observations if label == "droplet"]
    negatives = [(q1, q2) for q1, q2, label, _c, _s in observations if label == "no_droplet"]

    ratio_high = 4.185
    ratio_high_conf = "prior"
    if positives and negatives:
        r_pos = max(q1 / q2 for q1, q2 in positives if q2 > 0)
        r_neg = min(q1 / q2 for q1, q2 in negatives if q2 > 0)
        if r_neg > r_pos:
            ratio_high = math.sqrt(r_pos * r_neg)
            ratio_high_conf = "medium"
            notes.append(f"r_high 由 {r_pos:.3f}（成滴）与 {r_neg:.3f}（无滴）夹逼得到")
        else:
            ratio_high = r_pos
            ratio_high_conf = "low"
            notes.append("成滴/无滴证据区间重叠，r_high 取最大成滴比值，置信度低")

    size_intercept, size_q1, size_q2 = _fit_size(sizes)
    if sizes:
        notes.append(f"尺寸模型由 {len(sizes)} 段液滴观测拟合")
    notes.append(
        "r_low=2.2 为先验：泵安全下限 r>=2，历史最低成滴点 r=2.5，"
        "r∈[2,2.5] 尚无实验，建议做一次确认节点"
    )

    return CalibrationResult(
        ratio_low=2.2,
        ratio_high=round(ratio_high, 3),
        ratio_thread=6.0,
        size_intercept_um=size_intercept,
        size_q1_coef_um_per_ulmin=size_q1,
        size_q2_coef_um_per_ulmin=size_q2,
        ratio_low_confidence="prior",
        ratio_high_confidence=ratio_high_conf,
        ratio_thread_confidence="prior",
        droplet_points=len(positives),
        no_droplet_points=len(negatives),
        size_points=len(sizes),
        survey_records=total,
        survey_reviewed=reviewed,
        sources=tuple(dict.fromkeys(sources)),
        notes="；".join(notes),
    )


def summarize_calibration(result: CalibrationResult) -> dict[str, Any]:
    """给 CLI/报告用的紧凑摘要。"""

    return {
        "ratio_window": [result.ratio_low, result.ratio_high],
        "ratio_thread": result.ratio_thread,
        "confidence": {
            "ratio_low": result.ratio_low_confidence,
            "ratio_high": result.ratio_high_confidence,
            "ratio_thread": result.ratio_thread_confidence,
        },
        "evidence": {
            "droplet_points": result.droplet_points,
            "no_droplet_points": result.no_droplet_points,
            "size_points": result.size_points,
            "survey_records": result.survey_records,
            "survey_reviewed": result.survey_reviewed,
        },
        "size_model_um": {
            "intercept": round(result.size_intercept_um, 4),
            "per_ulmin_q1": round(result.size_q1_coef_um_per_ulmin, 5),
            "per_ulmin_q2": round(result.size_q2_coef_um_per_ulmin, 5),
        },
        "sources": list(result.sources),
        "notes": result.notes,
    }
