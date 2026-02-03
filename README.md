# ML / DL Portfolio
From-scratch implementations and competition-grade ML systems.

This repository is a curated portfolio of the best projects I have built end-to-end. They are full implementations.

## Core Capabilities

- From-scratch deep learning with NumPy and math
  - Manual forward/backward passes for CNNs and RNNs (NumPy), including gradient debugging and stability fixes
- Modern architectures implemented with custom modules in PyTorch
  - Transformer encoder–decoder, Swin Transformer, SSD object detection pipeline
- Competition-grade ML pipelines
  - Out-of-fold training, teacher stacking, multi-seed ensembling, Optuna hyperparameter optimization, weighted probability blending, feature creation and selection

## Highlights

- **MALLORN Astronomical Classification Competition**
  - **29th out of 2389 entrants (893 teams)**
  - Built a feature-engineered, teacher-stacked, Optuna-tuned XGBoost pipeline for imbalanced classification on LSST-like multi-band lightcurves
- **Manual gradient implementations**
  - Implemented and debugged full backprop for convolutional layers, pooling, dense layers, and GRU gates (NumPy)
- **Systems-level ML engineering**
  - Detection pipeline with anchors, IoU matching, hard negative mining, decoding, and Non-maximum Suppression
  - Training mechanics: AMP, warmup + cosine scheduling, stochastic depth, Mixup/CutMix, gradient clipping

## Project Index

| Project | Domain | What it demonstrates | Key Result | Location |
|---|---|---|---|---|
| **MALLORN Astronomical Classification Competition** | Time Series + Tabular ML | Feature engineering, leakage-safe CV, OOF predictions, teacher stacking, Optuna on OOF F1, threshold tuning, multi-seed ensembling | 29th / 2389 entrants, F1 ≈ 0.66 private LB | [`MALLORN_Astronomical_Classification_Competition/`](./MALLORN_Astronomical_Classification) |
| **LeNet-5 on MNIST (NumPy)** | Computer Vision | CNN from scratch in NumPy + PyTorch replication | ~94% test accuracy (NumPy) (minimal training) | [`CNN_from_Scratch_with_NumPy/`](./CNN_from_Scratch_with_NumPy) |
| **Swin Transformer (Tiny)** | Computer Vision | Windowed attention, shifted windows, modern training stack | ~56% Top-1 on Tiny ImageNet | [`Swin_Transformer/`](./Swin_Transformer) |
| **SSD (ResNet-34) Detector** | Computer Vision | Anchors, IoU matching, hard-negative mining, NMS | Correct pedestrian detection on Penn-Fudan | [`Single_Shot_Multibox_Detection/`](./Single_Shot_Multibox_Detection) |
| **GRU Translator (NumPy)** | NLP | Full BPTT GRU encoder–decoder | Stable predictions in PyTorch replication | [`GRU_Encoder_Decoder_Numpy/`](./GRU_Encoder_Decoder_Numpy) |
| **Transformer Translator** | NLP | Transformer encoder–decoder + beam search | Working beam-search translations | [`Transformer_Encoder-Decoder/`](./Transformer_Encoder-Decoder) |


## Featured Project: MALLORN (Kaggle Competition)

**Task:** Detect tidal disruption events (TDEs) using only photometric lightcurves (no spectra).  
**Data:** Irregular multi-band time series (LSST filters `u,g,r,i,z,y`) with object metadata (`Z`, `Z_err`, `EBV`, and train-only `SpecType`).  
**Core constraints:** Strong class imbalance and split-structured data that can cause leakage with naive validation.

### Outcome

- **29th out of 2389 entrants (893 teams)**
- Strong final leaderboard performance under tight time constraints (about 2 weeks instead of the full 4 months)

## How to navigate this repo
Start with the project README in each folder for architecture diagrams, design decisions, and results.

Each project folder contains:
- `README.md` with implementation details and results
- Fully commented Jupyter notebooks that include the full code and walk through the entire project step by step.
- Requirements are in `requirements.txt`

## Contact

- LinkedIn: https://www.linkedin.com/in/kaizen-rowe-1a0998349/
- Email: rowekaizen@gmail.com