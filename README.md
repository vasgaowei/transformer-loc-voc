# Weakly Supervised Object Localization using Transformer on Pascal VOC dataset. 

This repository contains training code and evaluation code for weakly supervised object localization on Pascal VOC dataset. The localization performance is measured using *CorLoc* metric

# Usage

First clone the repository locally:
```
https://github.com/vasgaowei/transformer-loc-voc.git
```
Then install Pytorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):


```

conda create -n pytorch1.7 python=3.6
conda activate pytorc1.7
conda install anaconda
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install timm==0.3.2
```
## Data preparation

### Pascal VOC dataset
