# Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models

This repository contains the official PyTorch implementation of StoryGen: https://arxiv.org/abs/2306.00973/

## Some Information
[Project Page](https://haoningwu3639.github.io/StoryGen_Webpage/)  $\cdot$ [PDF Download](https://arxiv.org/abs/2306.00973/)

## Requirements
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.12](https://pytorch.org/)
- xformers == 0.0.13
- diffusers == 0.13.1
- accelerate == 0.17.1
- transformers == 4.27.4

A suitable [conda](https://conda.io/) environment named `storygen` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate storygen
```

## Dataset Preparation
Coming soon...

## Training
Coming soon...


## Citation
If you use this code for your research or project, please cite:
 
	@article{liu2023intelligent,
      title={Intelligent Grimm -- Open-ended Visual Storytelling via Latent Diffusion Models}, 
      author={Chang Liu and Haoning Wu and Yujie Zhong and Xiaoyun Zhang and Weidi Xie},
      year={2023},
      journal={arXiv preprint arXiv:2306.00973},
	}

## Acknowledgements
Many thanks to the code bases from [diffusers](https://github.com/huggingface/diffusers).