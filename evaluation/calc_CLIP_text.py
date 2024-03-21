import os
import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint

from PIL import Image
import clip

def calc_probs(model, preprocess, image, text):
    
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text], truncate = True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score

model, preprocess = clip.load('ViT-B/32')

image_dir = './Test_results/StoryGen/'
video_text_dir = '../StorySalon-Pro/Text/Caption/Video/'
pdf_text_dir = '../StorySalon-Pro/Text/Caption/eBooks/'

images = sorted(os.listdir(image_dir))

pdf_candidates = ['African', 'Bloom', 'Book', 'Digital', 'Literacy', 'StoryWeaver']

scores = []
for image in images:
    name = image.split('_')

    if len(name) == 3:
        name = name[0] + "/" + "_".join(name[0:])
        text_dir = video_text_dir
    elif len(name) == 2:
        index = name[0]
        name = name[0] + "/" + "_".join(name[0:])        
        for candidate in pdf_candidates:
            if os.path.exists(os.path.join(pdf_text_dir, candidate, index)):
                text_dir = os.path.join(pdf_text_dir, candidate)
    name = name.replace(".jpg", ".txt")
    
    print(name)
    
    with open(os.path.join(text_dir, name), 'r') as f:
        text = f.read()
    print(text)
    
    img = os.path.join(image_dir, image)
    img = Image.open(img)

    score = calc_probs(model, preprocess, img, text)
    scores.append(score)
    
    print(np.mean(scores))