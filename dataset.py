import os
import cv2
import torch
import re
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms

# This Simple Dataset is just for testing the pipeline works well.
# Note: You should write a DataLoader suitable for your own Dataset!!!
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

        self.train_image_list = []
        self.train_mask_list = []
        self.train_text_list = []
        self.test_image_list = []
        self.test_mask_list = []
        self.test_text_list = []

        self.PDF_test_set = []
        self.video_test_set = []
        for line in open(os.path.join(self.root, 'PDF_test_set.txt')).readlines(): 
            self.PDF_test_set.append(line[:-1])
        for line in open(os.path.join(self.root, 'video_test_set.txt')).readlines():
            self.video_test_set.append(line[:-1])

        keys = ['African', 'Bloom', 'Book', 'Digital', 'Literacy', 'StoryWeaver']
        for key in keys:
            self.PDF_image_dir = os.path.join(self.root, 'Image_inpainted', key)
            self.PDF_mask_dir = os.path.join(self.root, 'Mask', key)
            self.PDF_text_dir = os.path.join(self.root, 'Text', 'Caption', key)
            PDF_folders = sorted(os.listdir(self.PDF_image_dir)) # 000575

            self.train_image_folders = [os.path.join(self.PDF_image_dir, folder) for folder in PDF_folders if folder not in self.PDF_test_set]
            self.train_mask_folders = [os.path.join(self.PDF_mask_dir, folder) for folder in PDF_folders if folder not in self.PDF_test_set]
            self.train_text_folders = [os.path.join(self.PDF_text_dir, folder) for folder in PDF_folders if folder not in self.PDF_test_set]
            self.test_image_folders = [os.path.join(self.PDF_image_dir, folder) for folder in PDF_folders if folder in self.PDF_test_set]
            self.test_mask_folders = [os.path.join(self.PDF_mask_dir, folder) for folder in PDF_folders if folder in self.PDF_test_set]
            self.test_text_folders = [os.path.join(self.PDF_text_dir, folder) for folder in PDF_folders if folder in self.PDF_test_set]

            for video in self.train_image_folders: # video: image_folder, /dataset/image/00001
                images = sorted(os.listdir(video))
                if len(images) <= 3:
                    print(video)
                    continue
                else:
                    for i in range(len(images) - 3):
                        self.train_image_list.append([os.path.join(video, images[i]), os.path.join(video, images[i+1]), os.path.join(video, images[i+2]), os.path.join(video, images[i+3])])
            
            for video in self.train_mask_folders: # video: mask_folder, /dataset/mask/00001
                masks = sorted(os.listdir(video))
                if len(masks) <= 3:
                    continue
                else:
                    for i in range(len(masks) - 3):
                        self.train_mask_list.append([os.path.join(video, masks[i]), os.path.join(video, masks[i+1]), os.path.join(video, masks[i+2]), os.path.join(video, masks[i+3])])
            
            for video in self.train_text_folders: # video: image_folder, /dataset/image/00001
                texts = sorted(os.listdir(video))
                if len(texts) <= 3:
                    continue
                else:
                    for i in range(len(texts) - 3):
                        self.train_text_list.append([os.path.join(video, texts[i]), os.path.join(video, texts[i+1]), os.path.join(video, texts[i+2]), os.path.join(video, texts[i+3])])

            for video in self.test_image_folders: # video: image_folder, /dataset/image/00001
                images = sorted(os.listdir(video))   
                if len(images) <= 3:
                    print(video)
                    continue
                else:
                    for i in range(len(images) - 3):
                        self.test_image_list.append([os.path.join(video, images[i]), os.path.join(video, images[i+1]), os.path.join(video, images[i+2]), os.path.join(video, images[i+3])])
            
            for video in self.test_mask_folders: # video: mask_folder, /dataset/mask/00001
                masks = sorted(os.listdir(video))
                if len(masks) <= 3:
                    continue
                else:
                    for i in range(len(masks) - 3):
                        self.test_mask_list.append([os.path.join(video, masks[i]), os.path.join(video, masks[i+1]), os.path.join(video, masks[i+2]), os.path.join(video, masks[i+3])])
            
            for video in self.test_text_folders: # video: image_folder, /dataset/image/00001
                texts = sorted(os.listdir(video))
                if len(texts) <= 3:
                    continue
                else:
                    for i in range(len(texts) - 3):
                        self.test_text_list.append([os.path.join(video, texts[i]), os.path.join(video, texts[i+1]), os.path.join(video, texts[i+2]), os.path.join(video, texts[i+3])])
        
        self.video_image_dir = os.path.join("./StorySalon/", 'image_inpainted_finally_checked')
        self.video_mask_dir = os.path.join("./StorySalon/", 'mask')
        self.video_text_dir = os.path.join(self.root, 'Text', 'Caption', 'Video')
        video_folders = sorted(os.listdir(self.video_image_dir)) # 00001
        self.train_image_folders = [os.path.join(self.video_image_dir, folder) for folder in video_folders if folder not in self.video_test_set]
        self.train_mask_folders = [os.path.join(self.video_mask_dir, folder) for folder in video_folders if folder not in self.video_test_set]
        self.train_text_folders = [os.path.join(self.video_text_dir, folder) for folder in video_folders if folder not in self.video_test_set]
        self.test_image_folders = [os.path.join(self.video_image_dir, folder) for folder in video_folders if folder in self.video_test_set]
        self.test_mask_folders = [os.path.join(self.video_mask_dir, folder) for folder in video_folders if folder in self.video_test_set]
        self.test_text_folders = [os.path.join(self.video_text_dir, folder) for folder in video_folders if folder in self.video_test_set]
        
        fns = lambda s: sum(((s,int(n))for s, n in re.findall('(\D+)(\d+)','a%s0'%s)),()) 
        
        for video in self.train_image_folders: # video: image_folder, /dataset/image/00001
            images = sorted(os.listdir(video), key=fns)
            if len(images) <= 3:
                print(video) # All stories shorter than 4 are in the train set.
                continue
            else:    
                for i in range(len(images) - 3):
                    self.train_image_list.append([os.path.join(video, images[i]), os.path.join(video, images[i+1]), os.path.join(video, images[i+2]), os.path.join(video, images[i+3])])

        for video in self.train_mask_folders: # video: mask_folder, /dataset/mask/00001
            masks = sorted(os.listdir(video), key=fns)
            if len(masks) <= 3:
                continue
            else:
                for i in range(len(masks) - 3):
                    self.train_mask_list.append([os.path.join(video, masks[i]), os.path.join(video, masks[i+1]), os.path.join(video, masks[i+2]), os.path.join(video, masks[i+3])])

        for video in self.train_text_folders: # video: image_folder, /dataset/image/00001
            texts = sorted(os.listdir(video), key=fns)
            if len(texts) <= 3:
                continue
            else:
                for i in range(len(texts) - 3):
                    self.train_text_list.append([os.path.join(video, texts[i]), os.path.join(video, texts[i+1]), os.path.join(video, texts[i+2]), os.path.join(video, texts[i+3])])
        
        for video in self.test_image_folders: # video: image_folder, /dataset/image/00001
            images = sorted(os.listdir(video), key=fns)
            if len(images) <= 3:
                print(video)
                continue
            else:    
                for i in range(len(images) - 3):
                    self.test_image_list.append([os.path.join(video, images[i]), os.path.join(video, images[i+1]), os.path.join(video, images[i+2]), os.path.join(video, images[i+3])])

        for video in self.test_mask_folders: # video: mask_folder, /dataset/mask/00001
            masks = sorted(os.listdir(video), key=fns)
            if len(masks) <= 3:
                continue
            else:
                for i in range(len(masks) - 3):
                    self.test_mask_list.append([os.path.join(video, masks[i]), os.path.join(video, masks[i+1]), os.path.join(video, masks[i+2]), os.path.join(video, masks[i+3])])

        for video in self.test_text_folders: # video: image_folder, /dataset/image/00001
            texts = sorted(os.listdir(video), key=fns)
            if len(texts) <= 3:
                continue
            else:
                for i in range(len(texts) - 3):
                    self.test_text_list.append([os.path.join(video, texts[i]), os.path.join(video, texts[i+1]), os.path.join(video, texts[i+2]), os.path.join(video, texts[i+3])])
        
        # In-house data
        # self.pdf_image_dir = os.path.join("/data/home/haoningwu/Dataset/StorySalon/", 'StoryBook_finally_checked', 'image_inpainted_finally_checked')
        # self.pdf_mask_dir = os.path.join("/data/home/haoningwu/Dataset/StorySalon/", 'StoryBook_finally_checked', 'mask')
        # self.pdf_text_dir = os.path.join(self.root, 'Text', 'Caption_new', 'eBook')
        # pdf_folders = sorted(os.listdir(self.pdf_image_dir)) # 00001
        # self.pdf_image_folders = [os.path.join(self.pdf_image_dir, folder) for folder in pdf_folders]
        # self.pdf_mask_folders = [os.path.join(self.pdf_mask_dir, folder) for folder in pdf_folders]
        # self.pdf_text_folders = [os.path.join(self.pdf_text_dir, folder) for folder in pdf_folders]
        # fns = lambda s: sum(((s,int(n))for s, n in re.findall('(\D+)(\d+)','a%s0'%s)),()) 
        
        # for video in self.pdf_image_folders: # video: image_folder, /dataset/image/00001
        #     images = sorted(os.listdir(video), key=fns)    
        #     if len(images) <= 3:
        #         print(video)
        #         continue
        #     else:    
        #         for i in range(len(images) - 3):
        #             self.train_image_list.append([os.path.join(video, images[i]), os.path.join(video, images[i+1]), os.path.join(video, images[i+2]), os.path.join(video, images[i+3])])

        # for video in self.pdf_mask_folders: # video: mask_folder, /dataset/mask/00001
        #     masks = sorted(os.listdir(video), key=fns)
        #     if len(masks) <= 3:
        #         continue
        #     else:
        #         for i in range(len(masks) - 3):
        #             self.train_mask_list.append([os.path.join(video, masks[i]), os.path.join(video, masks[i+1]), os.path.join(video, masks[i+2]), os.path.join(video, masks[i+3])])

        # for video in self.pdf_text_folders: # video: image_folder, /dataset/image/00001
        #     texts = sorted(os.listdir(video), key=fns)
        #     if len(texts) <= 3:
        #         continue
        #     else:
        #         for i in range(len(texts) - 3):
        #             self.train_text_list.append([os.path.join(video, texts[i]), os.path.join(video, texts[i+1]), os.path.join(video, texts[i+2]), os.path.join(video, texts[i+3])])
        
        if self.dataset_name == 'train':
            self.image_list = self.train_image_list
            self.mask_list = self.train_mask_list
            self.text_list = self.train_text_list
        elif self.dataset_name == 'test':
            self.image_list = self.test_image_list
            self.mask_list = self.test_mask_list
            self.text_list = self.test_text_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        ref_image_ids = self.image_list[index][0:3]
        image = self.image_list[index][3]
        mask = self.mask_list[index][3]
        
        ref_texts = self.text_list[index][0:3]
        text = self.text_list[index][3]
        
        ref_images_0 = []
        for id in ref_image_ids:
            ref_images_0.append(Image.open(id).convert('RGB'))
        image =  Image.open(image).convert('RGB')
        mask = Image.open(mask).convert('RGB')
        
        ref_images_1 = []
        for ref_image in ref_images_0:
            ref_images_1.append(ref_image.resize((512, 512))) 
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        
        ref_images_2 = []
        for ref_image in ref_images_1:
            ref_images_2.append(np.ascontiguousarray(transforms.ToTensor()(ref_image))) 
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        ref_images = torch.from_numpy(np.ascontiguousarray(ref_images_2)).float()
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        
        ref_prompts = []
        for ref_text in ref_texts:
            with open(ref_text, "r") as f:
                ref_prompts.append(f.read())
        with open(text, "r") as f:
            prompt = f.read()

        # Unconditional generation for classifier-free guidance
        if self.dataset_name == 'train':
            p = random.uniform(0, 1)
            if p < 0.05:
                prompt = ''
            p = random.uniform(0, 1)
            if p < 0.1:
                ref_prompts = ['','','']
                ref_images = ref_images * 0.
        
        # normalize
        for ref_image in ref_images:
            ref_image = ref_image * 2. - 1.
        # ref_images = ref_images * 2. - 1.
        image = image * 2. - 1.

        return {"ref_image": ref_images, "image": image, "mask": mask, "ref_prompt": ref_prompts, "prompt": prompt}


class COCOMultiSegDataset(Dataset):
    def __init__(self, root):
        self.seg_path = os.path.join(root, 'annotations/instances_train2017.json')
        self.cap_path = os.path.join(root, 'annotations/captions_train2017.json')
        self.image_path = os.path.join(root, 'train2017')

        with open(self.seg_path, 'r') as f:
            seg_data = json.load(f)
        with open(self.cap_path, 'r') as f:
            cap_data = json.load(f)

        self.image_list = seg_data['images']
        self.annotation_list = seg_data['annotations']
        self.category_list = seg_data['categories']
        self.caption_list = cap_data['annotations']


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_id = self.image_list[index]['id']

        image_info = next(image for image in self.image_list if image['id'] == image_id)
        image_path =os.path.join(self.image_path, image_info['file_name'])
        image = np.ascontiguousarray(Image.open(image_path).convert('RGB'))

        masks = [ann for ann in self.annotation_list if ann['image_id'] == image_id]

        captions= [item['caption'] for item in self.caption_list if item['image_id'] == image_id]
        tmp_ref_captions = []

        tmp_ref_images = [] # len(ref_captions) = len(ref_images)

        for annotation in masks:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            segmentation = annotation['segmentation']
            mask_cat = [item['name'] for item in self.category_list if item['id'] == annotation['category_id']]
            tmp_ref_captions.append(mask_cat[0])
            
            for segment in segmentation:
                if len(segment)>1:
                    poly = np.array(segment)
                    if poly.shape != ():
                        poly = poly.reshape((len(poly) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], color=255)

            tmp_ref_images.append(cv2.bitwise_and(image, image, mask=mask))
            
        while len(tmp_ref_images) < 3:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            tmp_ref_images.append(cv2.bitwise_and(image, image, mask=mask))
            tmp_ref_captions.append('') 
            
        if len(tmp_ref_images) > 3:
            ref_images = tmp_ref_images[0:2]
            ref_captions = tmp_ref_captions[0:3]
            for i in range(3,len(tmp_ref_images)):
                tmp_ref_images[2]+=tmp_ref_images[i]
            ref_images.append(tmp_ref_images[2])
        else:
            ref_images = tmp_ref_images
            ref_captions = tmp_ref_captions                
        
        ref_images_0 = []
        for id in ref_images:
            ref_images_0.append(Image.fromarray(id).convert('RGB'))        
        image = Image.fromarray(image).convert('RGB')
        
        ref_images_1 = []
        for ref_image in ref_images_0:
            ref_images_1.append(ref_image.resize((512, 512))) 
        image = image.resize((512, 512))
        
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=(-30, 30), translate=(0.2, 0.2), scale=(0.8, 1.3)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        ref_images_2 = []
        for ref_image in ref_images_1:
            ref_images_2.append(np.ascontiguousarray(transform(ref_image))) 
        image = transforms.ToTensor()(image)

        ref_images = torch.from_numpy(np.ascontiguousarray(ref_images_2)).float()
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        
        for ref_image in ref_images:
            ref_image = ref_image * 2. - 1.
        # ref_images = ref_images * 2. - 1.
        image = image * 2. - 1.

        if len(captions)>0:
            text = captions[random.randint(0, len(captions)-1)]
        else:
            text = ''
            
        # Unconditional generation for classifier-free guidance
        p = random.uniform(0, 1)
        if p < 0.05:
            text = ''
        p = random.uniform(0, 1)
        if p < 0.1:
            ref_captions = ['','','']
            ref_images = ref_images * 0.

        return {"image": image, 'prompt': text, 'ref_image': ref_images, 'ref_prompt': ref_captions}


class COCOValMultiSegDataset(Dataset):
    def __init__(self, root):
        self.seg_path = os.path.join(root, 'annotations/instances_val2017.json')
        with open(self.seg_path, 'r') as f:
            seg_data = json.load(f)

        self.annotation_list = seg_data['annotations']
        self.category_list = seg_data['categories']
        
        self.image_path = os.path.join(root, 'val2017')
        self.text_path = os.path.join("./COCOVal", 'Caption')
        
        self.image_list = sorted(os.listdir(self.image_path))
        self.caption_list = sorted(os.listdir(self.text_path))

        # self.image_list = self.image_list[3600:3950]


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_id = image_name.split('.')[0]
        text_path = os.path.join(self.text_path, (image_id + '.txt'))

        image_path =os.path.join(self.image_path, image_name)
        image = np.ascontiguousarray(Image.open(image_path).convert('RGB'))

        masks = [ann for ann in self.annotation_list if ann['image_id'] == int(image_id.lstrip('0'))]

        tmp_ref_captions = []

        tmp_ref_images = [] # len(ref_captions) = len(ref_images)

        for annotation in masks:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            segmentation = annotation['segmentation']
            mask_cat = [item['name'] for item in self.category_list if item['id'] == annotation['category_id']]
            tmp_ref_captions.append(mask_cat[0])
            
            for segment in segmentation:
                if len(segment)>1:
                    poly = np.array(segment)
                    if poly.shape != ():
                        poly = poly.reshape((len(poly) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], color=255)

            tmp_ref_images.append(cv2.bitwise_and(image, image, mask=mask))
            
        while len(tmp_ref_images) < 3:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            tmp_ref_images.append(cv2.bitwise_and(image, image, mask=mask))
            tmp_ref_captions.append('') 
            
        if len(tmp_ref_images) > 3:
            ref_images = tmp_ref_images[0:2]
            ref_captions = tmp_ref_captions[0:3]
            for i in range(3,len(tmp_ref_images)):
                tmp_ref_images[2]+=tmp_ref_images[i]
            ref_images.append(tmp_ref_images[2])
        else:
            ref_images = tmp_ref_images
            ref_captions = tmp_ref_captions                
        
        ref_images_0 = []
        for id in ref_images:
            ref_images_0.append(Image.fromarray(id).convert('RGB'))        
        image = Image.fromarray(image).convert('RGB')
        
        ref_images_1 = []
        for ref_image in ref_images_0:
            ref_images_1.append(ref_image.resize((512, 512))) 
        image = image.resize((512, 512))
        
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        ref_images_2 = []
        for ref_image in ref_images_1:
            ref_images_2.append(np.ascontiguousarray(transform(ref_image))) 
        image = transforms.ToTensor()(image)

        ref_images = torch.from_numpy(np.ascontiguousarray(ref_images_2)).float()
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        
        for ref_image in ref_images:
            ref_image = ref_image * 2. - 1.
        # ref_images = ref_images * 2. - 1.
        image = image * 2. - 1.

        with open(text_path, "r") as f:
            text = f.read()

        return {"image": image, 'prompt': text, 'ref_image': ref_images, 'ref_prompt': ref_captions, 'image_path': image_path}


if __name__ == '__main__':
    train_dataset = COCOMultiSegDataset(root="./COCO2017/")
    
    print(train_dataset.__len__())

    train_data = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)
    # B C H W
    for i, data in enumerate(train_data):
        print(i)
        print(data["prompt"])
        print(data["ref_prompt"])
        
        print(data["ref_image"].shape)
        print(data["image"].shape)
        if i > 9:
            break