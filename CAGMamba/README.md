# CAGMamba: Context-Aware Gated Cross-Modal Mamba Network for Multimodal Sentiment Analysis

## Overview

This repository contains the official implementation of **CAGMamba**, a context-aware gated cross-modal Mamba framework for dialogue-based multimodal sentiment analysis.

CAGMamba introduces two key innovations:

1. **Context-Aware Sequence Construction**: Organizes contextual and current-utterance features into a temporally ordered binary sequence, enabling Mamba to explicitly model sentiment evolution across dialogue turns.

2. **Gated Cross-Modal Mamba Network (GCMN)**: A three-stream fusion module that integrates cross-modal and unimodal paths via learnable gating, combined with a Bi-directional Selective Scanning Module (BSSM) for capturing sequential dependencies from both temporal directions.

The model is trained with a three-branch multi-task objective over text, audio, and fused predictions.

<p align="center">
  <img src="figures/framework.pdf" width="90%" alt="CAGMamba Framework"/>
</p>

## Results

### CMU-MOSI

| Method | Acc-2 (Has0/Non0) | F1 (Has0/Non0) | Acc-7 | MAE | Corr |
|--------|-------------------|-----------------|-------|-----|------|
| MMML | 87.17/87.15 | 88.01/88.13 | 51.38 | 0.56 | 0.86 |
| MSAmba | 85.99/87.43 | 85.99/87.40 | 49.67 | 0.71 | 0.81 |
| **CAGMamba (Ours)** | **88.19/88.15** | **90.09/90.09** | **52.48** | **0.56** | **0.88** |

### CMU-MOSEI

| Method | Acc-2 (Has0/Non0) | F1 (Has0/Non0) | Acc-7 | MAE | Corr |
|--------|-------------------|-----------------|-------|-----|------|
| MMML | 86.03/86.12 | 87.81/87.65 | 54.37 | 0.53 | 0.81 |
| MSAmba | 85.78/86.86 | 85.99/86.93 | 54.21 | 0.51 | 0.80 |
| **CAGMamba (Ours)** | **87.08/87.14** | **88.72/88.57** | 54.56 | 0.51 | **0.82** |

### CH-SIMS

| Method | Acc-5 | Acc-3 | Acc-2 | F1 | MAE | Corr |
|--------|-------|-------|-------|-----|-----|------|
| MMML | 49.38 | 68.29 | 81.18 | 80.98 | 0.349 | 0.701 |
| MSAmba | 47.17 | 68.83 | 82.30 | 81.75 | 0.403 | 0.646 |
| **CAGMamba (Ours)** | 47.03 | 68.35 | **83.13** | **83.15** | 0.351 | **0.703** |

## Environment Setup

1. Create a new environment (Python 3.8+ recommended):
```bash
conda create -n cagmamba python=3.8
conda activate cagmamba
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Download Datasets

The three datasets (CMU-MOSI, CMU-MOSEI, and CH-SIMS) are available from: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk

### Data Directory Structure

Place the downloaded data in the following structure:

```
data/
├── mosi/
│   ├── raw/
│   └── label.csv
├── mosei/
│   ├── raw/
│   └── label.csv
└── sims/
    ├── raw/
    └── label.csv
```

### Audio Extraction

Before training, extract audio from the raw video files:

```bash
python extract_audio.py --dataset mosi    # or mosei, sims
```

## Project Structure

```
CAGMamba/
├── main_msamba.py                 # Main entry point
├── msamba_blocks.py               # Core BSSM and GCMN building blocks
├── msamba_mmml_model.py           # CAGMamba model (CMU-MOSI / CMU-MOSEI)
├── msamba_mmml_model_sims.py      # CAGMamba model (CH-SIMS)
├── msamba_train.py                # Training logic (CMU-MOSI / CMU-MOSEI)
├── msamba_train_sims.py           # Training logic (CH-SIMS)
├── extract_audio.py               # Audio extraction utility
├── run_audio.py                   # Audio-only training
├── utils/
│   ├── data_loader.py             # Data loading utilities
│   ├── en_model.py                # English dataset model wrapper
│   ├── en_train.py                # English dataset training
│   ├── ch_model.py                # Chinese dataset model wrapper
│   ├── ch_train.py                # Chinese dataset training
│   ├── context_model.py           # Context modeling utilities
│   ├── cross_attn_encoder.py      # Cross-attention encoder
│   ├── audio_model.py             # Audio feature model
│   ├── audio_loader.py            # Audio data loader
│   ├── audio_train.py             # Audio training
│   └── metricsTop.py              # Evaluation metrics
├── checkpoint/                    # Saved model checkpoints
├── data/                          # Dataset directory
└── requirements.txt
```

## Training

### Train with Text + Audio (Full Model)

```bash
# CMU-MOSI (default)
python main_msamba.py --dataset mosi --batch_size 128 --lr 5e-6 --gcmn_depth 1

# CMU-MOSEI
python main_msamba.py --dataset mosei --batch_size 128 --lr 5e-6 --gcmn_depth 1

# CH-SIMS
python main_msamba.py --dataset sims --batch_size 128 --lr 1e-5 --gcmn_depth 1
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `mosi` | Dataset: `mosi`, `mosei`, or `sims` |
| `--seed` | `1` | Random seed |
| `--batch_size` | `128` | Batch size |
| `--lr` | `5e-6` | Learning rate |
| `--gcmn_depth` | `1` | Number of GCMN layers |
| `--loss` | `MTA` | Multi-task loss branches (M: fusion, T: text, A: audio) |
| `--context` | `True` | Whether to use context modeling |
| `--early_stop` | `8` | Early stopping patience |

### Train with Audio Features Only

```bash
python run_audio.py --dataset mosi --lr 1e-4 --batch_size 16
```

## Model Architecture

CAGMamba consists of four stages:

1. **Feature Extraction**: RoBERTa-Large (text) + Data2Vec-Audio-Large (audio, frozen)
2. **Sequence Construction**: Projects features into shared space, stacks context→main in temporal order
3. **GCMN Fusion**: Three-stream gated fusion with BSSM — cross-modal path + two unimodal paths combined via learnable gating
4. **Multi-Task Prediction**: Three prediction heads (text, audio, fused) trained jointly

## Efficiency

| Method | Params | FLOPs | Acc-2 (MOSI) |
|--------|--------|-------|--------------|
| Transformer | 4.45M | 0.38G | 84.74/86.20 |
| MSAmba | 1.41M | 0.13G | 85.99/87.43 |
| **GCMN (Ours)** | **0.80M** | **0.10G** | **88.19/88.15** |

## Citation

If you find this work useful, please cite:

```bibtex
@article{cagmamba2025,
  title={CAGMamba: Context-Aware Gated Cross-Modal Mamba Network for Multimodal Sentiment Analysis},
  author={Anonymous},
  year={2025}
}
```

## Acknowledgments

This codebase builds upon [MMML](https://github.com/declare-lab/MSA-with-multi-task-multi-loss) and [Mamba](https://github.com/state-spaces/mamba). We thank the authors for their contributions.

## License

This project is released for academic research purposes.