# CLIP-based Context-aware Academic Emotion Recognition

This repo is the official implementation for CLIP-based Context-aware Academic Emotion Recognition[[arixv](dkdjk)]. The paper has been accepted to ICCV 2025.

## Introduction

to be added.

## Weights Download

We provide the model weights trained by the method in this paper, which can be downloaded [here]().

## Visualizations

to be added.

## Environment

The code is developed and tested under the following environment:

to be update

- Python 3.8

- PyTorch 1.10.2

- CUDA 11.3

```bash
conda create -n dsta python=3.8
conda activate dsta
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Usage

### Training

### Evaluation

## Citations

If you find our paper useful in your research, please consider citing:

```bash
@InProceedings{Zhao_2025_ICCV,
    author    = {Zhao, Luming and Xuan, Jingwen and Lou, Jiamin and Yu, Yonghui and Yang, Wenwu},
    title     = {Context-Aware Academic Emotion Dataset and Benchmark},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025}
}
```

## Acknowledgment

Our codes are mainly based on [DFER-CLIP](https://github.com/zengqunzhao/DFER-CLIP/tree/main).Many thanks to the authors!
