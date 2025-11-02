# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RouterDC is a NeurIPS 2024 research project implementing a query-based router using dual contrastive learning to select the most suitable LLM for each input query. The router uses a DeBERTa-v3 encoder to embed queries and learns trainable LLM embeddings, optimizing selection through two contrastive losses: sample-to-LLM and sample-to-sample (cluster-based).

## Architecture

### Core Components

**RouterModule** (train_router_mdeberta.py:72-99)
- Backbone encoder: microsoft/mdeberta-v3-base (DeBERTaV2Model)
- Trainable LLM embeddings: nn.Embedding layer initialized with normal distribution (std=0.78)
- Similarity computation: Cosine similarity between query embeddings and LLM embeddings
- Forward pass uses first token ([CLS]) as query representation

**Dual Contrastive Losses**
- `compute_sample_llm_loss`: Sample-to-LLM contrastive loss using top-k positive and last-k negative LLMs
- `compute_cluster_loss`: Sample-to-sample contrastive loss grouping similar queries via cluster_id
- Loss weighting controlled by `--sample_loss_weight` and `--cluster_loss_weight`

**Dataset Types**
- `multi_attempt`: Sum of scores across attempts (code generation tasks like HumanEval, MBPP)
- `probability`: Binary 0/1 correctness (MMLU, ARC, etc.)

### Directory Structure

- `train_router_mdeberta.py`: Main training script for 7-model split2 datasets
- `train_router_mdeberta_routerbench.py`: Training script for RouterBench (11 models, supports cost-aware routing)
- `evaluation_router.py`: Standalone evaluation script loading trained checkpoints
- `datasets/`: Contains training/test splits with cluster annotations
  - `split2_model7_cluster/`: 7-model setup with cluster IDs for in-distribution tasks
  - `split2_model7/`: Test sets without cluster IDs
  - `routerbench_cluster/`: RouterBench datasets (CSV/PKL format)
- `train_scripts/`: Bash scripts with hyperparameter configurations
- `utils/meters.py`: AverageMeter for tracking metrics
- `src/cluster_generate.ipynb`: Notebook for assigning cluster IDs to training data

## Common Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training

**Quick debug run:**
```bash
python train_router_mdeberta.py \
  --data_paths ./datasets/split2_model7_cluster/gsm8k-train.json \
  --test_data_paths ./datasets/split2_model7/gsm8k-test.json \
  --test_data_type multi_attempt \
  --training_steps 50 \
  --batch_size 64 \
  --learning_rate 5e-5 \
  --eval_steps 50 \
  --save_path ./logs/debug
```

**Full training (recommended):**
```bash
# Adjust CUDA_VISIBLE_DEVICES in script first
bash train_scripts/router_train.sh
```

**Key training arguments:**
- `--data_paths`: List of training JSON files (must have cluster_id field)
- `--test_data_paths`: List of test JSON files
- `--test_data_type`: Corresponding evaluation types ("multi_attempt" or "probability")
- `--top_k`, `--last_k`: Number of positive/negative LLMs in sample-to-LLM loss (default: 3)
- `--cluster_loss_weight`: Weight for cluster contrastive loss (default: 1)
- `--sample_loss_weight`: Weight for sample-to-sample task-tag loss (default: 0)
- `--H`: Number of negative samples in cluster loss (default: 3)
- `--training_steps`: Total training iterations (default: 1000)
- `--eval_steps`: Evaluate every N steps (default: 50)
- `--final_eval`: Run comprehensive evaluation on all 8 datasets at end
- `--seed`: Random seed for reproducibility

### Evaluation

**Evaluate a trained checkpoint:**
```bash
python evaluation_router.py \
  --trained_router_path ./logs/debug/best_training_model.pth \
  --test_data_paths ./datasets/split2_model7/gsm8k-test.json ./datasets/split2_model7/mmlu_test.json \
  --test_data_type multi_attempt probability
```

Output format: Prints per-dataset accuracy followed by space-separated scores in fixed order (mmlu, gsm8k, cmmlu, arc, humaneval, MATH, mbpp, ceval).

### Dataset Format

**Training data (JSON):**
```json
{
  "question": "What is 2+2?",
  "scores": {"model1": 1.0, "model2": 0.0, ...},
  "cluster_id": 42
}
```

**RouterBench data (CSV/PKL):**
Must have `prompt` column and columns named after LLM identifiers plus corresponding `{llm}|total_cost` columns for cost-aware routing.

## Development Workflow

### Adding New Datasets
1. Generate LLM outputs using lm-evaluation-harness or bigcode-evaluation-harness (see eval_scripts/)
2. Merge scores with queries using convert_dataset_7_model.ipynb
3. Assign cluster IDs using src/cluster_generate.ipynb
4. Add paths to training script with appropriate `--test_data_type`

### Modifying Loss Functions
- Sample-to-LLM loss (train_router_mdeberta.py:101-125): Adjust `top_k`/`last_k` or masking logic
- Cluster loss (train_router_mdeberta.py:151-173): Modify negative sampling strategy in H parameter
- Loss weighting: Tune `--sample_loss_weight` and `--cluster_loss_weight` hyperparameters

### Checkpointing
- Best model saved to `{save_path}/best_training_model.pth` based on validation accuracy
- Training automatically evaluates at `--eval_steps` intervals
- Load checkpoints with `torch.load()` and apply to RouterModule instance

## Important Notes

- **GPU Configuration**: Set `CUDA_VISIBLE_DEVICES` before training (scripts default to specific GPUs)
- **Data Requirements**: Training datasets MUST include `cluster_id` field; test sets don't require it (defaults to 0)
- **Model Size**: Default 7 LLMs (node_size=7 for split2_model7, node_size=11 for RouterBench)
- **Similarity Function**: Cosine similarity (`--similarity_function cos`) is default and recommended
- **Reproducibility**: Use `--seed` parameter; setup_seed() sets all random states
- **Evaluation Metrics**:
  - `acc`: Exact match with oracle (best LLM)
  - `acc_predict`: Actual performance score when routing to predicted LLM
