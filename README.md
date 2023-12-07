# Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models

This repository contains the official PyTorch implementation of StoryGen: https://arxiv.org/abs/2306.00973/


# We will update our latest code and dataset soon!

## Some Information
[Project Page](https://haoningwu3639.github.io/StoryGen_Webpage/)  $\cdot$ [PDF Download](https://arxiv.org/abs/2306.00973/) $\cdot$ [Dataset](https://drive.google.com/file/d/1rz57PZHNCDCxU3x2jx6zqxn6IfxNC23E/view?usp=sharing)

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

## Meta Data Preparation
We provide the metadata of our StorySalon dataset in `./data/metadata.json`. It includes the id, name, url, duration and the keyframe list after filtering of the videos.

To download these videos, we recommend to use [youtube-dl](https://github.com/yt-dlp/yt-dlp) via:
```
youtube-dl --write-auto-sub -o 'file\%(title)s.%(ext)s' -f 135 [url]
```

The keyframes extracted with the following data processing pipeline(step 1) can be filtered according to the keyframe list provided in the metadata to avoid manually selection.

The corresponding masks, story-level description and visual description can be extracted with the following data processing pipeline or downloaded from [here](https://drive.google.com/file/d/1rz57PZHNCDCxU3x2jx6zqxn6IfxNC23E/view?usp=sharing).

## Data Processing Pipeline
The data processing pipeline includes several necessary steps: 
- Extract the keyframes and their corresponding subtitles;
- Detect and remove duplicate frames;
- Segment text, people, and headshots in images; and remove frames that only contain real people;
- Inpaint the text, headshots and real hands in the frames according to the segmentation mask;
- (Optional) Use Caption model combined with subtitles to generate a description of each image.

The keyframes and their corresponding subtitles can be extracted via:
```
python ./data_process/extract.py
```

The duplicate frames can be detected and removed via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/dup_remove.py
```

The text, people and headshots can be segmented, and the frames that only contain real people are then removed via:
```
python ./data_process/yolov7/human_ocr_mask.py
```

The text, headshots and real hands in the frames can be inpainted with [SDM-Inpainting](https://github.com/CompVis/stable-diffusion), according to the segmentation mask via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/SDM/inpaint.py
```

Besides, we also provide the code to get story-level paired image-text samples.
We can align the subtitles with visual frames by using Dynamic Time Warping(DTW) algorithm via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/align.py
```

(Optional) You can use [ChatCaptioner](https://github.com/Vision-CAIR/ChatCaptioner/tree/main/ChatCaptioner) to obtain the caption of each image via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/ChatCaptioner/main_caption.py
```

For a more detailed introduction to the data processing pipeline, please refer to `./data_process/README.md` and our paper.

## Training
Before training, please download pre-trained StableDiffusion-1.5 from [SDM](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) (including vae, scheduler, tokenizer and unet). Besides, please download pre-trained CLIP-vit-large from [CLIP](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) (pytorch_model.bin is required.) Then, all the pre-trained checkpoints should be placed into the corresponding location in the folder `./ckpt/stable-diffusion-v1-5/`

For Stage 1, train the StyleTransfer LoRA layers via:
```
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py --training_stage 1
```
For Stage 2, train the Context Moudle via:
```
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py --training_stage 2
```
If you have multiple GPUs to accelerate the training process, you can use:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train.py --training_stage 1
```

## Inference
```
python inference.py --ref_prompt 'Once upon a time, there is a white cat.' \
                  --prompt 'One day, the white cat is running in the rain.'
```

## TODO
- [x] Model & Training & Inference Code
- [x] Dataset Processing Pipeline
- [x] Meta Data
- [ ] (Soon) Code Update
- [ ] (Soon) Data Update
- [ ] (TBD) Release Checkpoints

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
