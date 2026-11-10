# mcs-date

面向微流控实验数据（MCS）的**实验记录 / 数据版本管理**工具骨架。

当前仓库是初始化脚手架，用于承载：

- 实验记录的规范化命名与日期归档（如 `YYYYMMDD_<实验名>`）；
- 实验数据（标定、日志、导出结果）的索引与检索；
- 与 [MCS_QT](https://github.com/snow78408-del/MCS_QT) 采集端的数据交接约定。

## 目录结构

```
mcs-date/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── src/
│   └── mcs_date/
│       ├── __init__.py     # 包入口，暴露 __version__
│       └── __main__.py     # python -m mcs_date 命令行入口
└── tests/
    └── test_smoke.py
```

## 快速开始

```bash
# 可选：创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows

# 可编辑安装
pip install -e .

# 运行
python -m mcs_date
```

## 开发

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## 许可证

[MIT](LICENSE)
