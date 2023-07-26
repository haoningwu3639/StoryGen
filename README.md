# Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models

This repository contains the official PyTorch implementation of StoryGen: https://arxiv.org/abs/2306.00973/

## Some Information
[Project Page](https://haoningwu3639.github.io/StoryGen_Webpage/)  $\cdot$ [PDF Download](https://arxiv.org/abs/2306.00973/) $\cdot$ Dataset (Coming Soon)

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
Before training, please download pre-trained StableDiffusion-1.5 from [SDM](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) (including vae, scheduler, tokenizer and unet). Besides, please download pre-trained CLIP-vit-large from [CLIP](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) (pytorch_model.bin is required.) Then, all the pre-trained checkpoints should be placed into the corresponding location in the folder `./ckpt/stable-diffusion-v1-5/`

For Stage 1, train the StyleTransfer LoRA layers via:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train.py --training_stage 1
```
For Stage 2, train the Context Moudle via:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train.py --training_stage 2
```


## Inference
Coming soon...

## TODO
- [x] Model Code
- [x] Training Code
- [ ] (Soon) Inference Code
- [ ] (Soon) Dataset Processing Pipeline
- [ ] (Soon) Meta Data

## Citation
If you use this code for your research or project, please cite:
 
	@article{liu2023intelligent,
      title={Intelligent Grimm -- Open-ended Visual Storytelling via Latent Diffusion Models}, 
      author={Chang Liu and Haoning Wu and Yujie Zhong and Xiaoyun Zhang and Weidi Xie},
      year={2023},
      journal={arXiv preprint arXiv:2306.00973},
	}

## Acknowledgements
Many thanks to the code bases from [diffusers](https://github.com/huggingface/diffusers) and [Tune-A-Video](https://github.com/showlab/Tune-A-Video).

## Contact
If you have any question, please feel free to contact haoningwu3639@gmail.com or liuchang666@sjtu.edu.cn.