# mcs-date

微流控两相流的**离线探索原型**：液滴视觉分析、双通道注射泵通信，以及以操作员选定工作点为起点的“蜘蛛式”节点扫描、结果存档与逐步成图。

> **能力边界**：当前 `demo` / `scan` 都是仿真，不是真实自动实验。独立 `vision` 命令可以读取视频或 OpenCV 相机，独立 `pump --port` 可以操作串口；二者尚未接成有物理标定、交汇点/断裂位置与唯一成滴事件测量的真实闭环适配器。工业相机 SDK 未移植；本轮没有硬件测试，不宣称真实端到端自动实验已经就绪。

视觉和泵通信核心裁剪自 [MCS_QT](https://github.com/snow78408-del/MCS_QT)，没有修改原工程。新增功能与证据限制见下文及[参考依据](<docs/references/README.md>)。

## 实验目标：通常从射流成滴点向周围探索

操作员先选一个当前认为最合适的 **jetting（射流成滴）工作点**，再探索周围流量：先看到液滴**尺寸**变化，继续偏离后，下游**断裂/形成位置**迁移，最后在给定观察窗口内表现为连续液丝、不再断裂。这里的“最好”是操作员选择，不是程序自动寻优或已经验证的最优点；默认 `(Q1,Q2)=(90,20) µL/min` 只是演示锚点。

- **jetting 可以形成液滴**：液丝在交汇点下游断裂；不能把 `jetting` 直接标为 `no_droplet`。
- **dripping** 指交汇点附近断裂，是可选研究模式，不是日常实验的默认目标。
- **threading** 表示在充分且已标定的观察范围内未见断裂，不等于在任意更远下游都永不成滴。
- 仅有液滴尺寸、CV、覆盖率或逐帧检测次数，不能独立证明断裂位置或流型；还需交汇点可见性、唯一事件计数、位置与视野长度。证据不全就保留 `unknown`。
- “观察到液滴”“均匀性候选”“射流/滴落流型”“已验证最佳点”是不同结论。当前均匀性规则只是工程筛选，不会确认最佳工作点。

### 仿真先验，不是学出的物理边界

默认 `jetting` 仿真在 `log(Q1)` 与 `log(Q1/Q2)` 空间构造以锚点为中心的椭圆偏离量 `rho`：`rho < 1` 生成射流成滴，之外生成窗口内未断裂的连续液丝；先改变尺寸，再让断裂位置向下游迁移。**中心、半径、阶段阈值及这套顺序都是说明性 PRIOR，不是历史数据学出的物理定律**。真实芯片未必按同样方向或幅度变化。

可选 `--target-regime dripping` 使用 `ratio_window` 示意模式。其 `ratio_low=2.2`、`ratio_high=4.185`、`ratio_thread=6.0` **目前全部为 prior**，不是确认的成滴范围，也不是默认 jetting 模式的真实边界。模型与数据来源保存在结果中，不能把仿真图当成实测相图。

## 快速开始

要求 Python ≥ 3.10；依赖见[项目配置](<pyproject.toml>)。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 无需相机、串口或显示器；仅离线演示
python -m mcs_date demo --out out
```

已有依赖时也可直接运行源码：

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"  # Windows 输出中文/µ 符号时避免旧代码页编码错误
python -m mcs_date demo --out out
```

演示包含合成帧视觉、仿真泵安全检查、仿真扫描与输出。下列输出名是运行后生成的文件，并非实测数据：

```text
out/demo_vision.png
out/calibration.json
out/scan_result.json
out/nodes.csv
out/phase_diagram.gif
out/phase_diagram_final.png
out/evolution_trace.png
```

## 命令行

完整参数以 `python -m mcs_date <子命令> --help` 为准；不带子命令显示帮助并返回 0。

```powershell
python -m mcs_date demo --out out --quiet
python -m mcs_date learn --out out --history-root E:\MCS_QT_Data

# 默认目标 jetting；90/20 是可替换的操作员锚点，不是最优解
python -m mcs_date scan --out out --center-q1 90 --center-q2 20 --target-regime jetting --node-seconds 180 --max-nodes 140 --max-hours 4 --search-rounds 1 --history-root E:\MCS_QT_Data

# 四射线小规模试验：默认 14 节点、约 0.8 h（仿真）
python -m mcs_date pilot --out out --anchor-q1 90 --anchor-q2 20 --step 0.25 --steps-per-ray 3 --max-hours 1.5

# 可选 dripping 研究模式，仍然只是 ratio_window 先验仿真
python -m mcs_date scan --out out_dripping --target-regime dripping --max-nodes 60 --max-hours 4 --no-animate
python -m mcs_date animate --result out/scan_result.json --out out

python -m mcs_date vision --synthetic --max-frames 120 --out out_vision
python -m mcs_date vision --video input.mp4 --wall-top 0.30 --wall-bottom 0.70 --out out_video
python -m mcs_date vision --camera 0 --max-frames 120 --out out_camera

# 不带 --port 为仿真；加 --port 将实际写硬件，须先完成独立安全验收
python -m mcs_date pump --q1 80 --q2 20
```

`scan` 支持 `--quiet`，但没有真实相机/串口闭环开关；指定历史目录也不会把仿真转换为实测。`learn` 还支持 `--observations`，输入必须是确经人工复核、来源可追溯的五元数组 `(Q1,Q2,label,confidence,source)`；不提供虚构标签示例作为默认实验事实。标签规则见[标定模块](<src/mcs_date/experiment/calibration.py>)。

## 生产模块与当前能力

| 部分 | 实现 | 当前范围 |
| --- | --- | --- |
| 视觉 | [帧流水线](<src/mcs_date/vision/pipeline.py>)、[液滴检测](<src/mcs_date/vision/detector.py>)、[管道定位](<src/mcs_date/vision/channel_region.py>)、[跟踪](<src/mcs_date/vision/tracker.py>)、[统计](<src/mcs_date/vision/metrics.py>) | BGR 帧 → ROI 校正、检测、跟踪和统计；真实物理尺寸仍需标定 |
| 相机 | [相机数据模型](<src/mcs_date/vision/cameras/models.py>)、[CLI 采集入口](<src/mcs_date/__main__.py>) | OpenCV `VideoCapture` 独立读帧；没有海康/GenTL/大恒等工业 SDK 适配器 |
| 泵 | [协议](<src/mcs_date/pump/protocol.py>)、[串口客户端](<src/mcs_date/pump/client.py>)、[控制服务](<src/mcs_date/pump/service.py>)、[安全不变量](<src/mcs_date/pump/invariants.py>) | 仿真与 TS 串口后端，参数回读、运行状态检查和停机失败传播 |
| 实验 | [分类/成滴证据](<src/mcs_date/experiment/phase.py>)、[仿真器](<src/mcs_date/experiment/chip.py>)、[配置/结果](<src/mcs_date/experiment/models.py>)、[扫描策略](<src/mcs_date/experiment/strategy.py>)、[四射线试验](<src/mcs_date/experiment/pilot.py>)、[执行器](<src/mcs_date/experiment/runner.py>) | 有限预算扫描与区间夹逼；真实测量源仍需单独实现并验收 |
| 证据与输出 | [只读标定](<src/mcs_date/experiment/calibration.py>)、[JSON/CSV](<src/mcs_date/experiment/report.py>)、[动画](<src/mcs_date/experiment/animate.py>)、[离线演示](<src/mcs_date/demo.py>) | 来源审计、保守加载、仿真水印与节点演化展示 |

### 视觉不是流型闭环适配器

`VisionPipeline.process_frame(frame, timestamp=..., current_wall_lines=...)` 返回检测、跟踪、计数和管道状态。可传两条归一化管壁直线做手动校正；未定位管道时不输出有效检测。像素到微米标定缺失时，不应把尺寸预览当作定量证据。

独立 CLI 相机使用单调时钟；视频优先媒体时间戳，必要时依据有效 FPS 回退。`total_detections` 是**逐帧检测数之和**，不是去重后的成滴事件数。当前路径没有自动提供已校准交汇点、断裂位置 `Lb/h`、充分观察长度及唯一事件，不能直接驱动真实流型闭环。

### 泵安全：回读参数不等于物理流量

CH1 为连续相（油）总流量 Q1，CH2 为分散相（水）Q2，单位 µL/min。TS 串口采用 `1200 8E1`；真实串口需额外安装 `pip install -e ".[hardware]"`。

- 流量必须有限、为正，并满足 **`Q1 >= 2*Q2`**；这是当前不可放宽的安全不变量，不是可随意调整的间隙参数。
- 串口后端写入后检查设定参数回读与运行状态；`parameter_readback_q1/q2` 只表示设备参数读回，不是独立流量计测量，不能证明实际物理流量。
- 启动前及退出时执行停机/状态确认；不能确认两通道停止会抛出 `PumpSafetyError`，不会静默当作成功。关闭串口不等于停泵。
- 扫描结束或异常会**尝试**停泵并传播失败，但断线、设备故障或机械原因下，软件无法保证硬件已经停止。必须准备独立急停与人工确认；当前离线测试不构成硬件安全认证。

## 蜘蛛式扫描、预算与未知区域

默认从操作员锚点出发，交替扩展 Q1 站位（最多 7 个，站位倍率 1.8）；每个站位固定 Q1，沿 Q2 两侧以倍率 1.6 探索。已出现目标/非目标观测对时，使用几何二分（默认 3 轮）缩小区间；粗节点未命中狭窄目标区时只做有限插点搜索，不保证一定发现。

每次真实执行的节点计划为换参数 → 稳定等待 30 s → 观测 180 s；总预算计入**所有节点的全部启动等待与观测时长**，默认最多 140 个节点、4 小时，流量范围 Q1∈[2,200]、Q2∈[0.2,100]，同时受安全比例约束。离线虚拟时钟可以很快跑完，不代表硬件只需这些墙钟时间。API 真实执行器使用单调时钟和有界 I/O，另存 `actual_elapsed_s`、`clock_kind`、状态及停止调用结果；计划总时长与实际耗时分开解释。

- 默认边界名 `jetting_lower` / `jetting_upper`；可选 dripping 模式为 `droplet_lower` / `droplet_upper`。`lower` / `upper` 指**固定 Q1 下 Q2 的两侧**，不是未经核实的物理相变常数。
- 边界保存原始内/外节点及 Q2 夹逼区间、区间宽度和分辨率；几何中点只是摘要，不是精确边界。
- `unknown` 不伪装成目标外部，不凭空生成边界；细化遇到未知会保留较粗且不确定的区间。
- 节点/时间预算耗尽、安全域截断、未观测到外侧或未发现目标都会记录状态/未解析区间，不能据此宣称完整区域已经找全。
- `classify_regime` 对真实证据要求与显式 `simulation` / `heuristic` 路径不同。工程阈值须经实际标定，低 CV 不会自动成为流型标签。

## 四射线小规模试验（pilot）

面对“通常到不了最佳滴落区、实际多在射流”的设备，`pilot` 不复刻全局自适应找窗，而是按操作员认可的射流锚点做一轮可解释的小试验（用户 2026-11-11 方案）：

- 固定锚点 `anchor_q1/anchor_q2`，沿 `R = Q1/Q2` 与 `S = Q1+Q2` 各向两侧走有限步（默认 25%，3 步/射线），最后回测锚点检查漂移/滞回；默认 14 个节点、每节点 180 s 观测，计划约 0.8 h（仿真）。
- 安全域或流量上下限之外的射线截断记入 `ray_status[*].censored_by`，不会为画完整边界而越界或补造数据。
- 同一条射线上相邻节点若出现目标与束流观测，记录为 `ray_crossings`：仅表示两点之间的**夹逼区间**，不改写成连续边界，也不补二分点；四射线是探索设计，不是最优搜索。
- 逐节点记录尺寸 `D/h`、CV、去重成滴事件与断裂位置 `Lb/h`、观测范围及演化阶段（`near_anchor` → `size_change` → `breakup_migration` → `unbroken_in_observation_window`，仅作参考差值，不是因果证明）。
- 输出 `pilot_result.json`、`nodes.csv`、`pilot_plan.csv`（含每条射线计划点与是否执行）、二相图与 `evolution_trace.png`。四射线点数不足以支撑区域插值，因此图只画实测点与夹逼短线，不填充流型区域。

该命令仍只运行**离线仿真**；`--max-hours` 会拒绝超出预算的计划，避免默认配置悄悄膨胀。真实执行需先接入带时间同步的测量源，并按泵安全条款验收。

## 历史学习：目前没有确认的成滴窗口

`learn` 只读结构化记录，默认历史根目录可用 `--history-root` 或环境变量 `MCS_QT_DATA_DIR` 指定。缺失/损坏来源会记录警告，不能用内置“整理标签”补成实验事实。

当前本机历史数据重算结果（源数据会变化，复核时以本次输出为准）：

| 证据 | 当前数量 | 可以支持什么 |
| --- | --- | --- |
| campaign 尺寸候选 | 12 | 仅用于低置信度尺寸趋势拟合；物理尺度未经独立验证 |
| response library | 45 条流量记录，其中 41 条长度有效 | 核对覆盖与响应长度候选；不与 campaign 直径混为同一标定尺度 |
| survey | 194 条 unknown/pending；0 reviewed | 记录缺口，不生成流型或成滴标签 |
| 有效复核成滴/不成滴标签 | 0 / 0 | `window_identified=false`，`calibration_status=prior` |

当前 `ratio_low=2.2`、`ratio_high=4.185`、`ratio_thread=6.0` 的置信度均为 `prior`；尺寸模型置信度为 `low`。尺寸拟合不是流型分类器，也不证明存在 dripping 最优区。历史覆盖包含流量比与总流量两个维度，不能因比值相同就忽略总流量混杂。

输出审计信息包括 `observation_ledger`（来源路径/记录索引、源 SHA-256、流量比、总流量、复核状态、标签、是否可用于窗口及排除理由）、`source_provenance`（只读来源及 SHA-256）、`coverage`、`interval_constraints` 和 `warnings`。SQLite 以只读 immutable 方式打开，不写源库，也不合并未落盘 WAL；历史图片/视频不在此学习步骤解码。

只有明确复核的成滴观测与注明高/低侧的不成滴观测，才能约束相应比值区间；单独 `jetting` 不是不成滴证据，未注明侧的 `no_droplet` 不能自动绑定高侧。约束冲突时保留冲突与先验，不移动参数去迎合数据。更完整的依据与缺口见[参考文档](<docs/references/README.md>)。

## 结果、来源与动画

运行输出：

- `scan_result.json`：`schema_version=2`，配置、节点完整特征、成滴评估、边界区间、来源与执行元数据。保留每节点 `features.source`，仿真记录有 `data_origin=simulation`、模型类型/配置/随机种子及未验证物理边界标记。
- `calibration.json`：证据账本、来源哈希、覆盖、约束、置信度和警告。
- `nodes.csv`：节点级摘要；完整演化特征以 JSON 为准。
- `phase_diagram.gif` 与 `phase_diagram_final.png`：按测量顺序逐步展示；GIF 可抽样节点以限制帧数，当前节点显示尺寸、CV、`Lb/h` 等统计。
- `evolution_trace.png`：按**节点顺序**展示液滴尺寸 `D/h`、CV 与断裂距离 `Lb/h` 的散点；不同蜘蛛腿不连线，缺测留空，不是自然时间演化或单条单调流量轨迹。束流节点不报告液滴尺寸与 `Lb/h`（未断裂不是 0）。
- `pilot` 另写 `pilot_result.json`、`nodes.csv`、`pilot_plan.csv`；`pilot_plan.csv` 的计划点与实际观测按 `purpose` 对齐，未执行的点标 `not_run`。

旧 schema 1 可读，但会警告来源与计时语义不明，将来源降级为 `unspecified_legacy`、边界分辨率降级为不确定；不会把旧文件当作实测证据。

绘图必须保留 **SIMULATION / 非实测** 水印；来源不明显示 **SOURCE UNVERIFIED**，不能自行抹去。安全域外被遮罩，距已测节点过远的区域仍为未知；有限邻域的最近邻填色不是新的测量。边界仅画原始夹逼区间与中点标记，不跨站位连成“已确认连续边界”，也不宣称填满整个相图。

GIF 编码与基础渲染使用 OpenCV/numpy 和内置编码器；Pillow 仅为可选中文字体增强。图像只能解释与其元数据一致的仿真/观测范围。

## 文献与开发

[Cubaud & Mason (2008)](https://doi.org/10.1063/1.2911716) 明确讨论 jetting 与 dripping 中的成滴；不能用其术语支持“射流无滴”。本芯片几何、相别角色和物性不与文献自动等价，不直接套用其常数。完整引用与证据分层见[参考文档](<docs/references/README.md>)。

```powershell
pip install -e ".[dev]"
pytest
ruff check .
```

测试侧重离线分类、证据保守性、扫描预算/未知与夹逼、来源持久化、动画、合成帧和泵安全失败路径。通过离线测试不表示真实硬件已验收。

许可证：[MIT](<LICENSE>)。
