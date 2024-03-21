import os
import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint
from PIL import Image
from transformers import AutoProcessor, AutoModel

def calc_probs(processor, model, prompt, images):
    # preprocess
    image_inputs = processor(images=images, padding=True, truncation=True, max_length=77, return_tensors="pt").to('cuda')
    text_inputs = processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to('cuda')

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # probs = torch.softmax(scores, dim=-1)
    
    # return probs.cpu().tolist()
    return scores.cpu().item()

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to("cuda")

image_dir = './Test_results/StoryGen/'
video_text_dir = '../StorySalon-Pro/Text/Caption/Video/'
pdf_text_dir = '../StorySalon-Pro/Text/Caption/StoryBooks/'

images = sorted(os.listdir(image_dir))

pdf_candidates = ['African', 'Bloom', 'Book', 'Digital', 'Literacy', 'StoryWeaver']

scores = []
for image in images:
    name = image.split('_')

    if len(name) == 2:
        index = name[0]
        name = name[0] + "/" + "_".join(name[0:])        
        for candidate in pdf_candidates:
            if os.path.exists(os.path.join(pdf_text_dir, candidate, index)):
                text_dir = os.path.join(pdf_text_dir, candidate)
    else:
        name = name[-4] + "/" + "_".join(name[-3:])
        text_dir = video_text_dir
    name = name.replace(".jpg", ".txt")
    print(name)
    
    with open(os.path.join(text_dir, name), 'r') as f:
        text = f.read()
    print(text)
    
    img = os.path.join(image_dir, image)
    img = Image.open(img)

    score = calc_probs(processor, model, text, img)
    scores.append(score)
    print(np.mean(scores))