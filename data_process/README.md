# Data Processing Pipeline for StoryGen

## Data Processing Pipeline
The data processing pipeline includes several necessary steps: 
- Extract the keyframes and their corresponding subtitles;
- Detect and remove duplicate frames;
- Segment text, people, and headshots in images; and remove frames that only contain real people;
- Inpaint the text, headshots and real hands in the frames according to the segmentation mask;
- (Optional) Use Caption model combined with subtitles to generate a description of each image.

We will introduce these steps in the following parts.

## Extract the keyframes and their corresponding subtitles

The keyframes and their corresponding subtitles can be extracted via:
```
python ./data_process/extract.py
```

Before running this code, make sure you have downloaded all the videos and their corresponding subtitle files, and put them into the same folder.

The path to the folder should be written in the [video_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/extract.py#L11).
And [save_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/extract.py#L12) is the path to store the extracted keyframes and their corresponding subtitles.

## Detect and remove duplicate frames

The duplicate frames can be detected and removed via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/dup_remove.py
```

[save_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/dup_remove.py#L7) is the path to the extracted keyframes and their corresponding subtitles.

## Segment text, people, and headshots in images; and remove frames that only contain real people

The text, people and headshots can be segmented, and the frames that only contain real people are then removed via:
```
python ./data_process/yolov7/human_ocr_mask.py
```

[weights](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/yolov7/human_ocr_mask.py#L76) is the path to the saved yolov7 checkpoints.
[root_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/yolov7/human_ocr_mask.py#L77) is the path to the extracted keyframes and their corresponding subtitles.
[mask_save_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/yolov7/human_ocr_mask.py#L78) is the path to store the corresponding masks.

## Inpaint the text, headshots and real hands in the frames according to the segmentation mask
With [SDM-Inpainting](https://github.com/CompVis/stable-diffusion), the text, headshots and real hands in the frames can be inpainted according to the segmentation mask via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/SDM/inpaint.py
```
Please git clone the SDM-Inpainting repository and place it to `./SDM`, download the required checkpoints, and use our script `./SDM/inpaint.py` to perform inpainting.

## Align the subtitles with visual frames

We also provide the code to get story-level paired image-text samples.
We can align the subtitles with visual frames by using Dynamic Time Warping(DTW) algorithm via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/align.py
```
[image_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/align.py#L17) is the path to the extracted keyframes and their corresponding subtitles.
[txt_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/align.py#L18) is the path to store the extracted corresponding relationship between keyframes and subtitles.

## (Optional) Use Caption model combined with subtitles to generate a description of each image
You can use [TextBind](https://github.com/SihengLi99/TextBind) to obtain the caption of each image via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/TextBind/main_caption.py
```
Please git clone the TextBind repository and place it to `./TextBind`, download the required checkpoints, and use our script `./TextBind/main_caption.py` to generate description.

You can also use [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4) to obtain the caption of each image via: 
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/MiniGPT-v2/main_caption.py
```
Please git clone the MiniGPT-v2 repository and place it to `./MiniGPT-v2`, download the required checkpoints, and use our script `./MiniGPT-v2/main_caption.py` to generate description.

(Discarded) Previous Method: You can use [ChatCaptioner](https://github.com/Vision-CAIR/ChatCaptioner/tree/main/ChatCaptioner) to obtain the caption of each image via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/ChatCaptioner/main_caption.py
```
Please git clone the ChatCaptioner repository and place it to `./ChatCaptioner`, download the required checkpoints, and use our script `./ChatCaptioner/main_caption.py` to generate description.
