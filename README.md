# Mamba-SFN

**A Lightweight Single-Layer Fusion Network for Multimodal Sentiment Analysis**

[![Paper](https://img.shields.io/badge/Paper-IJCNN%202026-blue)](https://doi.org/XXXX)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<p align="center">
  <img src="fig/architecture.png" width="700"/>
</p>

> Jiazheng Zhou, Xin Kang*, Kazuyuki Matsumoto, Linhuang Wang, Yupu Liu
>
> *Graduate School of Advanced Technology and Science, Tokushima University*

---

## Overview

Mamba-SFN is a resource-efficient multimodal sentiment analysis framework that replaces conventional multi-layer Transformer encoders with a **single-layer Mamba encoder** for temporal modeling, combined with a **single-layer bidirectional cross-attention module** for inter-modal interaction and a **bidirectional gating mechanism** for adaptive fusion.

### Highlights

- **24M-parameter** core fusion module (148M total including pretrained encoders)
- **2.79 GB** peak GPU memory on CMU-MOSI
- **4.37 ms/sample** inference latency
- Competitive or state-of-the-art performance on **CMU-MOSI**, **CMU-MOSEI**, and **CH-SIMS**

---

## Results

### CMU-MOSI & CMU-MOSEI

| Model | Year | ACC₂ (Has0/Non0) | F1 (Has0/Non0) | ACC₇ | MAE / Corr | Params |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MulT | 2019 | 81.10/– | 81.00/– | 39.10 | 0.889 / 0.686 | 111M |
| UniMSE | 2022 | 85.85/86.90 | 85.83/86.42 | 48.68 | 0.691 / 0.809 | – |
| MMML | 2024 | 85.91/88.16 | 85.85/88.15 | 48.25 | 0.643 / 0.838 | 453M |
| **Mamba-SFN** | **2026** | **86.15/88.57** | **86.09/88.56** | **49.56** | **0.631 / 0.841** | **148(24)M** |

### CH-SIMS

| Model | Year | ACC₂ | ACC₃ | ACC₅ | F1 | MAE | Corr |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ALMT | 2023 | 81.19 | 68.93 | 45.73 | **81.57** | 0.404 | 0.619 |
| 3WD-DRT | 2025 | 81.37 | 66.64 | 43.44 | 80.71 | 0.408 | 0.620 |
| **Mamba-SFN** | **2026** | **81.40** | **70.24** | **46.83** | 81.34 | **0.403** | **0.624** |

---

## Getting Started

### Environment

```bash
conda create -n mamba-sfn python=3.8
conda activate mamba-sfn
pip install -r requirements.txt
```

**Key dependencies:** PyTorch 2.1.0, Transformers 4.34.1, mamba-ssm

### Data Preparation

1. Download [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/), [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/), and [CH-SIMS](https://github.com/thuiar/MMSA) datasets.

2. Extract audio from video files:

```bash
# For MOSEI (includes video fixing for corrupted frames)
python extract_audio.py --dataset mosei

# For SIMS
python extract_audio.py --dataset sims
```

3. Organize the data directory as:

```
data/
├── MOSI/
│   ├── Raw/          # Original video files
│   └── wav/          # Extracted audio (auto-generated)
├── MOSEI/
│   ├── Raw/
│   └── wav/
└── SIMS/
    ├── Raw/
    └── wav/
```

### Pretrained Models

| Modality | English | Chinese |
|:---:|:---:|:---:|
| Text | [RoBERTa-base](https://huggingface.co/roberta-base) | [RoBERTa-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) |
| Audio | [Data2Vec](https://huggingface.co/facebook/data2vec-audio-base) | [HuBERT](https://huggingface.co/TencentGameMate/chinese-hubert-base) |

### Training

```bash
# CMU-MOSI
python run.py --dataset mosi --lr 6.5e-6 --batch_size 8 --model mamba --num_hidden_layers 1

# CMU-MOSEI
python run.py --dataset mosei --lr 6e-6 --batch_size 8 --model mamba --num_hidden_layers 1

# CH-SIMS
python run.py --dataset simi --lr 1e-6 --batch_size 8 --model mamba --num_hidden_layers 1
```

### Arguments

| Argument | Default | Description |
|:---|:---:|:---|
| `--seed` | 42 | Random seed |
| `--batch_size` | 8 | Batch size |
| `--lr` | 6.5e-6 | Learning rate |
| `--model` | mamba | Model type (`mamba`, `cc`, `SAGE`) |
| `--dataset` | mosi | Dataset (`mosi`, `mosei`, `simi`) |
| `--num_hidden_layers` | 1 | Number of cross-modality encoder layers |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhou2026mambasfn,
  title     = {Mamba-SFN: A Lightweight Single-Layer Fusion Network for Multimodal Sentiment Analysis},
  author    = {Zhou, Jiazheng and Kang, Xin and Matsumoto, Kazuyuki and Wang, Linhuang and Liu, Yupu},
  booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026}
}
```

## Acknowledgments

This work has been supported by the Project of Discretionary Budget of the Dean, Graduate School of Technology, Industrial and Social Sciences, Tokushima University, and the Collaboration Program between Tokushima University and National Taiwan University of Science and Technology.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
