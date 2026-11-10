# 参考文献与实测依据

## Cubaud & Mason (2008)

> T. Cubaud and T. G. Mason, "Capillary threads and viscous droplets in square
> microchannels", *Physics of Fluids* **20**, 053302 (2008).
> DOI: [10.1063/1.2911716](https://doi.org/10.1063/1.2911716)

本仓库的流型判据（滴落 / 射流 / 稳定束流 / 栓塞 / 浸润）与边界建模参考了这篇论文：

- 方形截面十字通道，四个臂宽均为 `h`；中心相 L1 以 `Q1` 注入，两侧鞘相 L2 各以 `Q2/2` 注入；
- 两相毛细数 `Ca_i = μ_i J_i / σ`，表观速度 `J_i = Q_i / h²`；
- 稳定束流直径 `δ/h ≈ (Q1/(2Q2))^{1/2}`；
- `Ca1 ≈ 10^-1` 附近为滴落与射流/束流的转折，高鞘相流量趋向稳定束流，过高进入栓塞；
- 低界面张力体系在低 `Ca1` 下出现浸润/指进。

论文 PDF 由用户提供，仅用于本地阅读与参数建模，**未提交到本仓库**（见 `.gitignore`
中的 `docs/references/*.txt`）。本目录只保留引用信息与结论摘要，不转载原文。

## Kovalchuk et al. (2019)

> N. M. Kovalchuk et al., *Microfluidics and Nanofluidics* (2019).
> DOI: [10.1007/s10404-019-2269-z](https://doi.org/10.1007/s10404-019-2269-z)

来自 MCS_QT 的 `experiments/cross_junction_literature_20261014` 报告：固定分散相流量、
改变连续相流量时观察到**重入式 jetting → dripping → jetting**（总油/水比 3.33 为
jetting，4.0 与 10 为 dripping，11.1 又回到 jetting）。这支持本芯片“只有中间油水比
窗口能形成液滴、两侧分别进入射流/束流”的非单调结构。

## 本芯片的角色映射

文献中 L1 是较粘的中心相、L2 是鞘相；本芯片的 CH1（油）是两侧鞘相、CH2（水）是
中心相。因此 `ChipModelConfig` 中的边界参数是按本芯片几何与流体角色重写的待标定参数，
不能直接套用文献中的常数。

## 历史数据（只读，不随仓库分发）

`mcs_date.experiment.calibration` 读取本机 `MCS_QT_Data` 下的结构化记录（路径可用
`MCS_QT_DATA_DIR` 覆盖）：

- `experiments/experiment_20261016/campaign_analysis/fit_results.json`：12 段液滴尺寸；
- `models/response_library.sqlite3`：45 条换流量实验；
- `experiments/*/session_summary.json`：194 条 `regime_survey` 记录（全部未知/待复核，只统计缺口）。

这些原始数据体积很大且包含实验图像，**不进入本仓库**；仓库只保存由它们重算出的标定参数
（`calibration.json` 由 `python -m mcs_date learn` 生成）。
