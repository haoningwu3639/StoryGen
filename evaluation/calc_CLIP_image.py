import os
import numpy as np
import clip
import torch
import torch.utils.data
import torch.utils.checkpoint

from tqdm.auto import tqdm
from PIL import Image


def calc_probs(model, preprocess, image, gt):
    image_input = preprocess(image).unsqueeze(0)
    gt_input = preprocess(gt).unsqueeze(0)
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    gt_input = gt_input.to(device)
    model = model.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        gt_features = model.encode_image(gt_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, gt_features.T).item()
        
    return clip_score

model, preprocess = clip.load('ViT-B/32')
image_dir = './Test_results/StoryGen/'
gt_dir = '../StorySalon/Testset_GT_resize/'

scores = []
images = sorted(os.listdir(image_dir))
for image in images:
    name = image.split('_')
    
    img = os.path.join(image_dir, image)
    img = Image.open(img)
    gt = os.path.join(gt_dir, image)
    gt = Image.open(gt)

    score = calc_probs(model, preprocess, img, gt)
    scores.append(score)
    
    print(np.mean(scores))