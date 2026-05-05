# PAPformer

Adaptive Multi-Scale Patch Transformer for Time Series Forecasting.

> **[中文文档](#中文文档)**

## Overview

PAPformer introduces an adaptive patch tokenization mechanism for Transformer-based time series forecasting. Instead of using a fixed patch size, the model dynamically selects patch sizes per token via a gating network conditioned on local signal complexity (variance + finite-difference energy).

The full pipeline is **two-stage**:

1. **SMVMD** (Sliding-Window Variational Mode Decomposition) decomposes the raw time series into Intrinsic Mode Functions (IMFs).
2. **PAPformer** takes the decomposed IMFs as input features for multi-step forecasting.

## Repository Structure

```
PAPformer/
├── code/
│   ├── APTransformer.py    # PAPformer model, training, and evaluation
│   └── SMVMD.py            # Sliding-Window VMD signal decomposition
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: SMVMD Signal Decomposition

Decompose the raw signal into IMFs first. The output CSV contains one column per IMF plus the reconstructed signal, which will be used as input features for PAPformer.

```bash
python code/smvmd.py \
    --data_file /path/to/data.csv \
    --column 14 \
    --window_len 20600 \
    --overlap 0.4 \
    --K 9 \
    --alpha 2 \
    --save_csv smvmd_results.csv
```

### Step 2: Train PAPformer

Use the decomposed IMF data to train the forecasting model:

```bash
python code/APTransformer.py \
    --data_file /path/to/smvd_results.csv \
    --timestep 14 \
    --epochs 300 \
    --batch_size 256 \
    --lr 8e-5 \
    --d_model 128 \
    --nhead 4 \
    --num_layers 3 \
    --patch_sizes 2 3 5 7 11 14
```

## Model Architecture

**Adaptive Patch Tokenizer** extracts multi-scale patches (sizes 2, 3, 5, 7, 11, 14) and blends them per token using a gating network. The gating weights are regularized via:

- KL divergence against a complexity-aware prior
- Temporal smoothness penalty
- Entropy regularization

The blended patch tokens are passed through standard sinusoidal positional encoding, then a stack of Pre-LN Transformer encoder blocks with GELU-activated FFNs, and finally a forecast head.


---

## 中文文档

**[English](#papformer)**

### 概述

PAPformer 是一种面向时间序列预测的自适应多尺度 Patch Transformer 模型。不同于固定 patch 大小的方法，本模型通过门控网络根据局部信号复杂度（方差 + 有限差分能量）动态选择每个 token 的 patch 大小。

完整流程分为 **两个阶段**：

1. **SMVMD**（滑动窗口变分模态分解）将原始时间序列分解为多个本征模态函数（IMF）。
2. **PAPformer** 以分解后的 IMF 作为输入特征进行预测。

### 仓库结构

```
PAPformer/
├── code/
│   ├── APTransformer.py    # PAPformer 模型、训练与评估
│   └── smvmd.py            # 滑动窗口 VMD 信号分解
├── requirements.txt
└── README.md
```

### 安装

```bash
pip install -r requirements.txt
```

### 使用方法

#### 第一步：SMVMD 信号分解

先将原始信号分解为 IMF 分量。输出的 CSV 文件包含每个 IMF 一列以及重构信号，作为 PAPformer 的输入特征。

```bash
python code/smvmd.py \
    --data_file /path/to/data.csv \
    --column 14 \
    --window_len 20600 \
    --overlap 0.4 \
    --K 9 \
    --alpha 2 \
    --save_csv smvmd_results.csv
```

#### 第二步：训练 PAPformer

使用分解后的 IMF 数据训练预测模型：

```bash
python code/APTransformer.py \
    --data_file /path/to/smvd_results.csv \
    --timestep 14 \
    --epochs 300 \
    --batch_size 256 \
    --lr 8e-5 \
    --d_model 128 \
    --nhead 4 \
    --num_layers 3 \
    --patch_sizes 2 3 5 7 11 14
```

### 模型架构

**自适应 Patch 分词器** 提取多尺度 patch（大小 2, 3, 5, 7, 11, 14），并通过门控网络逐 token 进行混合。门控权重通过以下方式正则化：

- KL 散度（相对于复杂度感知先验）
- 时间平滑性惩罚
- 熵正则化

混合后的 patch token 经过标准正弦位置编码，然后送入多层 Pre-LN Transformer 编码器块（含 GELU 激活的 FFN），最后通过预测头输出结果。


