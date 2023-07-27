# Data Processing Pipeline for Intelligent Grimm

## Data Processing Pipeline
The data processing pipeline includes three necessary steps: 
- Extract the keyframes and their corresponding subtitles
- Remove duplicate frames
- Segment and remove person frames and headshots.

We will introduce these steps in the following parts.

## Extract the keyframes and their corresponding subtitles

The keyframes and their corresponding subtitles can be extracted via:
```
python ./data_process/extract.py
```

Before running this code, make sure you have downloaded all the videos and their corresponding subtitle files, and put them into the same folder.

The path to the folder should be written in the [video_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/extract.py#L11).
And [save_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/extract.py#L12) is the path to store the extracted keyframes and their corresponding subtitles.

## Remove duplicate frames

The duplicate frames can be removed via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/dup_remove.py
```

[save_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/dup_remove.py#L7) is the path to the extracted keyframes and their corresponding subtitles.

## Segment and remove person frames and headshots

The person frames and headshots can be segmented and removed via:
```
python ./data_process/yolov7/human_ocr_mask.py
```

[weights](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/yolov7/human_ocr_mask.py#L76) is the path to the saved yolov7 checkpoints.
[root_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/yolov7/human_ocr_mask.py#L77) is the path to the extracted keyframes and their corresponding subtitles.
[mask_save_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/yolov7/human_ocr_mask.py#L78) is the path to store the corresponding masks.

## Align the subtitles with visual frames

We also provide the code to get story-level paired image-text samples.
We can align the subtitles with visual frames by using Dynamic Time Warping (DTW) algorithm via:
```
CUDA_VISIBLE_DEVICES=0 python ./data_process/align.py
```

[image_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/align.py#L17) is the path to the extracted keyframes and their corresponding subtitles.
[txt_path](https://github.com/haoningwu3639/StoryGen/blob/f30602498a37a3df1036e1c3a3097d7cd2a1920d/data_process/align.py#L18) is the path to store the extracted corresponding relationship between keyframes and subtitles.