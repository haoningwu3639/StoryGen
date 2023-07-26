import re
import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# This Simple Dataset is just for testing the pipeline works well.
class SimpleDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.image_dir = os.path.join(self.root, 'image')
        self.mask_dir = os.path.join(self.root, 'mask')
        self.text_dir = os.path.join(self.root, 'text')
        
        folders = sorted(os.listdir(self.image_dir))
        self.image_list = [os.path.join(self.image_dir, file) for file in folders]
        self.mask_list = [os.path.join(self.mask_dir, file) for file in folders]
        self.text_list = [os.path.join(self.text_dir, file.replace('.png', '.txt')) for file in folders]
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.image_list[index]
        mask = self.mask_list[index]
        text = self.text_list[index]
    
        image =  Image.open(image).convert('RGB')
        mask = Image.open(mask).convert('RGB')
        image = image.resize((512, 512))
        ref_image = image.resize((224, 224))
        mask = mask.resize((512, 512))
        
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        ref_image = transforms.ToTensor()(ref_image)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        ref_image = torch.from_numpy(np.ascontiguousarray(ref_image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()[[0], :, :] # 1 channel is enough
        # normalize
        image = image * 2. - 1.
        ref_image = ref_image * 2. - 1.
        
        with open(text, "r") as f:
            prompt = f.read()

        return {"image": image, "ref_image": ref_image, "mask": mask, "prompt": prompt}


class StorySalonDataset(Dataset):
    def __init__(self, root, dataset_name):
        self.root = root
        self.dataset_name = dataset_name
        self.image_dir = os.path.join(self.root,'image')
        self.mask_dir = os.path.join(self.root, 'mask')
        self.text_dir = os.path.join(self.root, 'text')
        
        folders = sorted(os.listdir(self.image_dir)) # 00001
        self.image_folders = [os.path.join(self.image_dir, folder) for folder in folders]
        self.mask_folders = [os.path.join(self.mask_dir, folder) for folder in folders]
        self.text_folders = [os.path.join(self.text_dir, folder) for folder in folders]

        self.image_list = []
        self.mask_list = []
        self.text_list = []
        
        fns = lambda s: sum(((s,int(n))for s, n in re.findall('(\D+)(\d+)','a%s0'%s)),()) 
        
        for video in self.image_folders: # video: image_folder, /dataset/image/00001
            images = sorted(os.listdir(video), key=fns)
            for i in range(len(images) - 1):
                self.image_list.append([os.path.join(video, images[i]), os.path.join(video, images[i+1])])

        for video in self.mask_folders: # video: mask_folder, /dataset/mask/00001
            masks = sorted(os.listdir(video), key=fns)
            for i in range(len(masks) - 1):
                self.mask_list.append([os.path.join(video, masks[i]), os.path.join(video, masks[i+1])])

        for video in self.text_folders: # video: text_folder, /dataset/text/00001
            texts = sorted(os.listdir(video), key=fns)
            for i in range(len(texts) - 1):
                self.text_list.append([os.path.join(video, texts[i]), os.path.join(video, texts[i+1])])

        cnt = int(len(self.image_list) * 0.9)
        if self.dataset_name == 'train':
            self.image_list = self.image_list[:cnt]
            self.mask_list = self.mask_list[:cnt]
            self.text_list = self.text_list[:cnt]
        elif self.dataset_name == 'test':
            self.image_list = self.image_list[cnt:]
            self.mask_list = self.mask_list[cnt:]
            self.text_list = self.text_list[cnt:]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        ref_image = self.image_list[index][0]
        image = self.image_list[index][1]
        mask = self.mask_list[index][1]
        text = self.text_list[index][1]
        
        ref_image = Image.open(ref_image).convert('RGB')
        image =  Image.open(image).convert('RGB')
        mask = Image.open(mask).convert('RGB')
        
        ref_image = ref_image.resize((224, 224))
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        ref_image = transforms.ToTensor()(ref_image)
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        ref_image = torch.from_numpy(np.ascontiguousarray(ref_image)).float()
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()[[0], :, :]
        
        with open(text, "r") as f:
            prompt = f.read()
        
        # normalize
        ref_image = ref_image * 2. - 1.
        image = image * 2. - 1.

        return {"ref_image": ref_image, "image": image, "mask": mask, "prompt": prompt}


if __name__ == '__main__':
    train_dataset = SimpleDataset(root="./data/")
    print(train_dataset.__len__())
    train_data = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    for i, data in enumerate(train_data):
        print(i)
        print(data['image'].shape)
        print(data['prompt'])