# mcs-date

面向微流控芯片的**两相流实验系统**：实时视频液滴识别、注射泵连接控制、以及“蜘蛛式”分节点扫描自动测绘**二相图**（流型图）并输出缓慢成图的动画。

本仓库从 [MCS_QT](https://github.com/snow78408-del/MCS_QT) 移植了视觉分析与泵通信的核心代码（原工程未做任何修改），并新增了实验扫描、二相图建模、历史数据标定与动画层。所有功能都可以在没有相机、没有串口、没有显示器的机器上以**离线仿真**方式跑通。

![二相图最终结果](docs/examples/phase_diagram_final.png)

## 核心物理口径：液滴窗口

本芯片的实测经验（用户 2026-11-10 反馈，已写入模型与测试）是：

- 油水比 **r = Q1/Q2 太低** → 连续相切割能力不足 → 不能形成液滴（宽大栓塞，`tubing`）；
- 油水比 **r 太高** → 连续相拖拽/剪切占优 → 也不能形成液滴（细射流 `jetting`，继续升高变成稳定束流 `threading`）；
- **只有中间一段 r 窗口能形成液滴**（`dripping`），且实际设备通常达不到最佳滴落点、更多落在**射流侧**。

因此二相图上的“束流边界”在这里被重新解释为**液滴窗口的上/下边界**：

- `droplet_lower`（下边界，r = r_high）：再往高油水比走就进入射流；
- `droplet_upper`（上边界，r = r_low）：再往低油水比走就变成栓塞。

## 三部分系统

| 部分 | 模块 | 作用 |
| --- | --- | --- |
| ① 视觉 | `mcs_date.vision` | 实时视频采集/读帧 → 管道区域定位 → 液滴检测、跟踪、尺寸与流量统计 |
| ② 泵 | `mcs_date.pump` | TS 注射泵 1200/8/E/1 串口协议；真实串口后端 + 仿真后端；流量设定、回读校验、安全停机 |
| ③ 实验 | `mcs_date.experiment` | 蜘蛛式节点扫描 → 流型分类 → 液滴窗口边界夹逼 → 历史数据标定 → JSON/CSV 结果 → 二相图动画 |

```
src/mcs_date/
├── vision/        # 移植自 MCS_QT backend/vision（裁剪版）
│   ├── pipeline.py        VisionPipeline：帧 → VisionResult
│   ├── detector.py        液滴检测（胶囊轮廓/插头几何）
│   ├── channel_region.py  管壁定位与校正
│   ├── tracker.py / kalman_tracker.py / nearest_tracker.py
│   ├── metrics.py / bead_counter.py
│   └── cameras/models.py  相机数据模型（真实 SDK 适配器未移植）
├── pump/          # 移植自 MCS_QT backend/pump_hardware（裁剪版）
│   ├── protocol.py        WSS/RSS/WSP/RSP/WSE/RSE 帧编解码
│   ├── client.py          PumpClient（pyserial 可选）
│   ├── service.py         PumpBackend 抽象 / SimulatedPumpBackend / SerialPumpBackend / PumpController
│   └── invariants.py      Q1 >= 2*Q2 安全约束
├── experiment/
│   ├── phase.py           FlowRegime、NodeFeatures、RegimeThresholds、classify_regime
│   ├── chip.py            ChipSimulator：由流量点生成流型与带噪测量特征
│   ├── calibration.py     历史数据标定：液滴窗口、液滴尺寸模型、置信度与来源
│   ├── models.py          ScanConfig / NodeRecord / BoundaryBracket / ScanResult
│   ├── strategy.py        SpiderScanner：从稳定工作点向两侧扩展 + 二分夹逼边界
│   ├── runner.py          ExperimentRunner（真实时钟/虚拟时钟）、run_offline_scan
│   ├── report.py          JSON/CSV 持久化（schema_version=1）
│   └── animate.py         OpenCV 渲染 + 纯 Python GIF89a 编码器
├── demo.py        三部分离线演示（合成视频、仿真泵、离线扫描 + 动画）
└── __main__.py    CLI：demo / learn / scan / animate / vision / pump
```

## 快速开始

```bash
git clone https://github.com/snow78408-del/mcs-date.git
cd mcs-date
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"

# 一次跑通三部分（离线，无需硬件）
python -m mcs_date demo --out out
```

如果暂时不想安装（或没有网络装 setuptools），也可以直接用源码目录运行：

```powershell
$env:PYTHONPATH = "src"     # Windows PowerShell
# export PYTHONPATH=src     # Linux/macOS
python -m mcs_date demo --out out
```

`demo` 会依次输出：

- `out/demo_vision.png`：合成微通道视频上的液滴检测标注（三帧拼图）；
- 泵设定 (80, 20) µL/min 的回读值、故意违反 `Q1 >= 2*Q2` 时的安全报错、停泵回读；
- `out/calibration.json`、`out/scan_result.json`、`out/nodes.csv`、`out/phase_diagram.gif`、`out/phase_diagram_final.png`。

## 命令行

```text
python -m mcs_date demo    --out out [--quiet]
python -m mcs_date learn   --out out [--history-root E:\MCS_QT_Data] [--quiet]
python -m mcs_date scan    --out out [--center-q1 70] [--center-q2 20] [--node-seconds 180] [--no-animate]
python -m mcs_date animate --result out/scan_result.json --out out
python -m mcs_date vision  --synthetic [--max-frames 120] [--out out]
python -m mcs_date vision  --video input.mp4   [--wall-top 0.30 --wall-bottom 0.70]
python -m mcs_date vision  --camera 0
python -m mcs_date pump    [--q1 80] [--q2 20] [--port COM3]
```

不带参数时打印帮助并返回 0。

## ① 视觉：液滴识别

`VisionPipeline.process_frame(frame, timestamp=..., current_wall_lines=[...])` 接收一帧 BGR 图像并返回 `VisionResult`（`detections` / `tracking` / `beads` / `metrics` / `channel_region` 等）。

- `current_wall_lines` 传入两条归一化管壁直线（`{"x1","y1","x2","y2"}`，坐标 ∈ [0,1]）时，直接按手动管壁做 ROI 校正，跳过自动标定；不传则走 `ChannelRegionDetector` 自动定位。
- 管道区域未定位（`status` 为 `collecting`/`rejected`）时返回空检测结果。
- `detections.centers`、`detections.radii` 可直接用于画圈与计数；尺寸/插头长度等字段在标定后才有意义。

CLI 的 `--synthetic` 使用内置合成视频（亮通道 + 深色液滴，与 MCS_QT 采集视频极性一致）验证整条链路。真实相机路径用 OpenCV `VideoCapture`；MCS_QT 的海康/GenTL/大恒等厂商 SDK 适配器没有移植，需要时可在 `vision/cameras/` 下按 `FramePacket` 约定接入。

## ② 泵：连接与控制

- 协议：TS 注射泵 1200/8/E/1，串口 `1200 8E1`，地址 1..31；帧格式 `FLAG(0xE9) + 转义(addr,len,pdu,fcs)`，`fcs` 为 XOR 校验。支持 `WSS/RSS`（使能与回读）、`WSP/RSP`（流量参数与回读）、`WSE/RSE`（运行状态）。
- 通道约定：**CH1 = 连续相（油）总流量 Q1**，**CH2 = 分散相（水）流量 Q2**。
- 安全约束：`Q1 >= 2 * Q2`（可用 `STRICT_Q1_Q2_GAP_UL_MIN=0.2`、`STRICT_Q1_Q2_RATIO=2.0` 调整），由 `PumpController.apply()` 在写泵前强制校验。
- `SimulatedPumpBackend`：无硬件时完整模拟设定/回读/停泵，用于测试与演示。
- `SerialPumpBackend`：真实串口后端（需 `pip install -e ".[hardware]"` 安装 pyserial），包含注册通道、写参数、逐通道回读校验、启动与停泵确认。

```python
from mcs_date.pump import PumpController, SimulatedPumpBackend

pump = PumpController(SimulatedPumpBackend())
pump.connect()
print(pump.apply(80.0, 20.0))   # -> (80.0, 20.0)
pump.stop()
pump.close()
```

## ③ 实验：蜘蛛式扫描与二相图

### 扫描策略

`SpiderScanner` 从**已知稳定工作点**（默认 Q1=70、Q2=20 µL/min，即历史恒流基准 70/20）出发：

1. 以 `station_step_ratio=1.8` 沿 Q1 向左右交替扩展出多个“蜘蛛腿”站位（默认最多 7 个），保持 Q1/Q2 中心比例不变；
2. 每个站位固定 Q1，沿 Q2 以 `step_ratio=1.6` 乘法步幅向低、向高两个方向逐点测量；
3. 每个节点：换流量 → 稳定等待 `startup_s=30 s` → 观测 `node_duration_s=180 s`（约 3 分钟/节点）；
4. 低 Q2 方向已进入 `threading`、高 Q2 方向已进入 `tubing`，或达到节点预算即收手；
5. 首次出现流型变化时记录边界区间，再用几何二分（默认 3 轮）夹逼 `droplet_lower` / `droplet_upper` 的位置。

默认完整扫描约 **56 个节点、21 个边界、约 3.3 小时**（离线仿真用虚拟时钟，秒级跑完）。

### 流型与边界

`FlowRegime`：`dripping`（滴落/成滴）、`jetting`（射流）、`threading`（稳定束流）、`tubing`（栓塞/充满）、`displacement`（浸润/指进）、`unknown`。

流型分类由 `classify_regime(NodeFeatures)` 依据观测窗口内的连续液流覆盖率、液滴生成频率、液滴直径与变异系数等特征判定，阈值集中在 `RegimeThresholds`，可用真实数据标定。

边界类型（`BoundaryBracket.kind`）：

- `droplet_lower`：滴落 ↔ 射流，**液滴窗口下边界**（油水比 r 的较高侧）；
- `droplet_upper`：滴落 ↔ 栓塞，**液滴窗口上边界**（油水比 r 的较低侧）；
- `thread_onset`：射流 ↔ 稳定束流之间（辅助信息）。

在二相图上，两条边界之间的带状区域就是“能形成液滴”的流量窗口；动画里两条折线随节点逐点出现，最终形成完整的流型分区。

### 历史数据标定（`learn`）

`mcs_date.experiment.calibration` 会只读扫描 `MCS_QT_Data`（路径可用环境变量 `MCS_QT_DATA_DIR` 覆盖）并产出 `calibration.json`：

| 证据 | 用法 |
| --- | --- |
| `experiments/experiment_20261016/campaign_analysis/fit_results.json` | 12 段液滴尺寸（末 2 分钟中位长度 → 等效直径），拟合 `D ≈ 92.30 − 0.0221(Q1−60) − 0.3674(Q2−20) µm` |
| `models/response_library.sqlite3` | 45 条换流量实验（前后流量、基线长度、30 个长度样本、是否接受），用于核对流量覆盖 |
| `experiments/*/session_summary.json` | 194 条 `regime_survey` 记录——全部 `unknown/pending`，**不作为标签**，只统计证据缺口 |
| 人工整理的观测 | 70/20、60/20、70/25、50/20、65/22、70/18 记为成滴；90.09/20 记“长结构/无滴”；100/20 记“疑束流” |

由此得到默认液滴窗口（可用 `MCS_QT_DATA_DIR` 指向别处后重跑 `learn` 更新）：

| 参数 | 值 | 置信度 | 说明 |
| --- | --- | --- | --- |
| `r_high`（→ 射流侧） | 4.185 | medium | 由成滴最大 r=3.889（70/18）与无滴最小 r=4.505（90.09/20）几何夹逼 |
| `r_low`（→ 栓塞侧） | 2.2 | **prior** | 泵安全约束要求 r≥2，历史最低成滴点 r=2.5（50/20 评价“较理想”），r∈[2,2.5] 尚无实验 |
| `r_thread` | 6.0 | prior | 射流 → 稳定束流的工程分界 |

`ChipSimulator` 默认使用这组标定值；`ChipModelConfig.from_calibration(result)` 可注入任意重算结果。**局限**：历史 session 没有人工复核的流型标签，`r_high` 只有两点夹逼、`r_low` 是先验；真实使用前建议在 r∈[2,2.5] 和 r∈[3.9,4.5] 各补确认节点，并把 `nodes.csv` 与视频回放引入复核流程。

### 结果文件

`scan_result.json`（schema_version=1）包含配置、中心节点、全部节点、边界区间与阈值；`calibration.json` 记录标定参数、置信度、证据计数与来源；`nodes.csv` 每个节点一行，便于用 Excel/pandas 复核。`load_result()` 可回读后重新生成动画。

### 动画

`animate_scan(result, out_dir)` 生成：

- `phase_diagram.gif`：每测一个节点画一帧，流型区域按已测节点做最近邻填充，边界逐条出现，最后定格在完整二相图（可循环播放）；
- `phase_diagram_final.png`：最终静态图。

渲染只依赖 OpenCV（numpy），GIF 由内置纯 Python GIF89a 编码器写出，不需要 matplotlib/Pillow/ffmpeg。若环境里恰好装有 Pillow，会额外用系统中文字体（如 `msyh.ttc`）渲染中文标签；没有 Pillow 时自动回退为英文标签，不影响成图与动画生成。

## 与文献的关系（流型判据）

边界模型主要参考两篇工作，并结合本芯片的历史实验做角色重映射：

1. Cubaud & Mason, *Capillary threads and viscous droplets in square microchannels*, Phys. Fluids 20, 053302 (2008)：方形截面十字通道中以两相毛细数 `Ca_i = μ_i J_i / σ`（表观速度 `J_i = Q_i/h²`）划分流型；稳定束流直径近似 `δ/h ≈ √(Q1/(2Q2))`（本芯片角色互换后为 `√(Q2/(2Q1))`，即 `√(1/(2r))`）；`Ca ≈ 0.1` 附近是滴落 ↔ 射流/束流转折。
2. Kovalchuk et al. 2019（DOI 10.1007/s10404-019-2269-z，MCS_QT 的 `cross_junction_literature_20261014` 报告）：固定分散相流量、改变连续相流量时观察到**重入式 jetting → dripping → jetting**（总油水比 3.33→jetting，4.0/10→dripping，11.1→jetting），支持本仓库“中间窗口成滴、两侧不成滴”的非单调结构。

**注意角色映射**：文献的 L1 是更粘的中心相、L2 是鞘相；本芯片 CH1（油）是两侧鞘相、CH2（水）是中心相，且两篇文献的几何/流量范围都不同。因此 `ChipModelConfig` 中的窗口参数是**本芯片待标定参数**，不直接套用文献常数。

## 安全与真实硬件

- 所有硬件写操作都有超时、回读校验与异常传播；扫描结束（含异常）一定停泵。
- 真实相机/串口都通过同一套抽象接入，测试与 CI 不需要硬件。
- 像素标定（`pixel_to_micron`、通道宽度）缺失时，视觉只适合预览；不要用未标定的尺寸做定量结论。
- 仿真器的流型阈值只是“可跑通”的工程近似，**不能替代真实实验标定**。

## 开发

```bash
pip install -e ".[dev]"
pytest            # 52 个测试，全部离线
ruff check .
```

测试覆盖：流型分类、液滴窗口边界与标定覆盖、扫描边界夹逼与时间预算、JSON/CSV 往返、GIF 编码往返（自写解码器校验）、视觉合成帧检测、泵仿真与安全约束、CLI 各子命令（含 `learn` 的历史数据读取容错）。

## 许可证

[MIT](LICENSE)

## 致谢

视觉与泵通信核心代码来自 [MCS_QT](https://github.com/snow78408-del/MCS_QT)（本仓库仅复制裁剪，未修改原工程）。
