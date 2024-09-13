# Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models (CVPR 2024)

This repository contains the official PyTorch implementation of StoryGen: https://arxiv.org/abs/2306.00973/

<div align="center">
   <img src="./teaser.png">
</div>

## Some Information
[Project Page](https://haoningwu3639.github.io/StoryGen_Webpage/)  $\cdot$ [Paper](https://arxiv.org/abs/2306.00973/) $\cdot$ [Dataset](https://huggingface.co/datasets/haoningwu/StorySalon) $\cdot$ [Checkpoint](https://huggingface.co/haoningwu/StoryGen)

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
#### Data from YouTube
We provide the metadata of our StorySalon dataset in `./data/metadata.json`. It includes the id, name, url, duration and the keyframe list after filtering the videos.

To download these videos, we recommend to use [youtube-dl](https://github.com/yt-dlp/yt-dlp) via:
```
youtube-dl --write-auto-sub -o 'file\%(title)s.%(ext)s' -f 135 [url]
```

The keyframes extracted with the following data processing pipeline (step 1) can be filtered according to the keyframe list provided in the metadata to avoid manual selection.

The corresponding masks, story-level description and visual description can be extracted with the following data processing pipeline or downloaded from [here](https://huggingface.co/datasets/haoningwu/StorySalon).

#### Data from Open-source Libraries
For the open-source PDF data, you can directly download the frames, corresponding masks, description and narrative from [StorySalon](https://huggingface.co/datasets/haoningwu/StorySalon).

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

(Optional) You can use [TextBind](https://github.com/SihengLi99/TextBind) or [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4) to obtain the caption of each image via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/TextBind/main_caption.py
CUDA_VISIBLE_DEVICES=0 python ./data_process/MiniGPT-v2/main_caption.py
```

(Discarded) Previous method: You can also use [ChatCaptioner](https://github.com/Vision-CAIR/ChatCaptioner/tree/main/ChatCaptioner) to obtain the caption of each image via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/ChatCaptioner/main_caption.py
```

For a more detailed introduction to the data processing pipeline, please refer to `./data_process/README.md` and our paper.

## Training
Before training, please download pre-trained StableDiffusion-1.5 from [SDM](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) (including vae, scheduler, tokenizer and unet). Then, all the pre-trained checkpoints should be placed into the corresponding location in the folder `./ckpt/stable-diffusion-v1-5/`

For Stage 1, pre-train the self-attention layers in SDM for StyleTransfer via:
```
CUDA_VISIBLE_DEVICES=0 accelerate launch train_StorySalon_stage1.py
```

For Stage 2, train the Visual-Language Context Module via:

```
CUDA_VISIBLE_DEVICES=0 accelerate launch train_StorySalon_stage2.py
```

For replicating the experiments on MS-COCO, train via:

```
CUDA_VISIBLE_DEVICES=0 accelerate launch train_COCO.py
```

If you have multiple GPUs to accelerate the training process, you can use:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train_StorySalon_stage2.py
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py
```

## About the Testset
Regarding the test set in our StorySalon dataset, some researchers have reported that they are unable to obtain corresponding test data through our data processing pipeline due to certain YouTube videos being taken down.

To address this issue, we have provided pre-processed test datasets for download via [Google Drive](https://drive.google.com/file/d/1o4ZzFyc4rTEFnud9FGhlTVftkw9QoJOp/view) or [HuggingFace](https://huggingface.co/datasets/haoningwu/StorySalon/blob/main/testset.zip), which can be used for testing and benchmarking purposes.

In consideration of potential copyright risks associated with YouTube videos, we have not directly included the original video content.

Please feel free to contact us if you have any questions regarding this part.

## TODO
- [x] Model & Training & Inference Code
- [x] Dataset Processing Pipeline
- [x] Meta Data
- [x] Code Update
- [x] Release Checkpoints
- [x] Data Update
- [x] Share links for testset

## License
The code and checkpoints in this repository are under MIT license.
The open-source books in the StorySalon dataset come from multiple online open-source libraries (please refer to the Appendix of our paper for more details), and they are all under CC-BY 4.0 license.
It should be noted that, for the data extracted from the video data, we only provide YouTube URLs and data processing pipelines, if you wish to use them for commercial purposes, we recommend that you obey the relevant regulations of YouTube.

## Citation
If you use this code for your research or project, please cite:

      @inproceedings{liu2024intelligent,
            title     = {Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models},
            author    = {Liu, Chang and Wu, Haoning and Zhong, Yujie and Zhang, Xiaoyun and Wang, Yanfeng and Xie, Weidi},
            booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
            pages     = {6190--6200},
            year      = {2024}
      }


## Acknowledgements
Many thanks to the code bases from [diffusers](https://github.com/huggingface/diffusers) and [SimpleSDM](https://github.com/haoningwu3639/SimpleSDM).

## Contact
If you have any questions, please feel free to contact haoningwu3639@gmail.com or liuchang666@sjtu.edu.cn.
