import os
import re
import torch
from PIL import Image
from torchvision import transforms

save_path = 'image'

transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
model.cuda()

def image_dedup_reg(image_path):

    pre_feature = torch.zeros(1,768).to(device)
    pre_i_path = ''    

    # sort according to the numerical order instead of alphabetic order
    fns = lambda s: sum(((s,int(n))for s,n in re.findall('(\D+)(\d+)','a%s0'%s)),())  
    l = len(sorted(os.listdir(image_path),key=fns))
    i = 0

    for image_name in sorted(os.listdir(image_path),key=fns):
        i_path = image_path + "/" + image_name

        if os.path.splitext(os.path.basename(i_path))[1] == '.jpg':
            image = transform(Image.open(i_path)).unsqueeze(0).to(device)

            image_feature = model(image)
            similarity = torch.cosine_similarity(image_feature, pre_feature , dim=1)
            print(i_path+":  "+str(similarity.item()))

            if similarity.item() >= 0.75:
                if os.path.isfile(pre_i_path):
                    os.remove(pre_i_path) 
            
            pre_feature = image_feature
            pre_i_path = i_path

            i+=1

    torch.cuda.empty_cache()    

if __name__ == '__main__':

    for video_no in sorted(os.listdir(save_path)):
        image_path = save_path + "/" + video_no
        try:
            with torch.no_grad():
                image_dedup_reg(image_path)
            print('Finished Video: ' + video_no)
        except:
            print('Failed Video: ' + video_no)
               
# CUDA_VISIBLE_DEVICES=0 python dup_remove.py