# Repository Guidelines

## 项目结构与模块组织
- `train_router_mdeberta.py`, `train_router_mdeberta_routerbench.py`：核心训练入口，提供命令行参数。
- `evaluation_router.py`：加载已训练权重并输出各任务指标。
- `train_scripts/`：可复现实验的训练脚本（如 `router_train.sh`）。
- `eval_scripts/`：辅助生成/评估 LLM 输出的脚本。
- `datasets/` 与 `data/`：数据集与中间产物；大型文件请勿入库。
- `utils/`：通用工具（如 `meters.py`）。
- `src/`：笔记本与实验（如 `cluster_generate.ipynb`）。
- `figs/`：论文与 README 使用的图像。

## 构建、测试与开发命令
- 安装依赖：`python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- 快速训练（示例）：
  `python train_router_mdeberta.py --data_paths ./datasets/split2_model7_cluster/gsm8k-train.json --test_data_paths ./datasets/split2_model7/gsm8k-test.json --test_data_type multi_attempt --training_steps 50 --save_path ./logs/debug`
- 脚本训练：`bash train_scripts/router_train.sh`（先确认 `CUDA_VISIBLE_DEVICES` 与数据/保存路径）。
- 评估检查点：
  `python evaluation_router.py --trained_router_path ./logs/debug/best_training_model.pth`

## 代码风格与命名约定
- Python 3，遵循 PEP 8，4 空格缩进；函数/变量用 `snake_case`，类用 `CapWords`。
- 使用命令行参数替代硬编码路径；优先相对路径（如 `./logs/...`）。
- 笔记本放置于 `src/`；可复用逻辑沉淀到 `utils/` 或新模块。

## 测试指南
- 暂无单元测试；请用 `evaluation_router.py` 在代表性数据集上验证。
- 新增逻辑建议先进行 smoke run（小 `--training_steps`），并在 `./logs/` 记录指标。
- 保持数据集列表与 `--test_data_type` 一致；新增数据字段需补充说明。

## 提交与 Pull Request 规范
- 采用 Conventional Commits（如 `feat: ...`、`fix: ...`），与现有提交历史一致。
- PR 需包含：变更目的、关键改动、复现命令（训练/评估）、所用数据与路径说明（必要时附截图/表格）。
- 避免脚本中使用绝对路径；以参数化方式提供并记录默认值。

## 安全与配置提示
- GPU 设置：在训练前设置 `CUDA_VISIBLE_DEVICES`。
- 数据管理：勿提交大型数据/模型文件；必要时更新 `.gitignore`。
- 可复现性：固定随机种子（`--seed`）；用 `requirements.txt` 管理环境。
