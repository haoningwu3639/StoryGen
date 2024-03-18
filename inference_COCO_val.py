import os
import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline

from dataset import COCOValMultiSegDataset
from transformers import AutoProcessor, AutoModel

logger = get_logger(__name__)

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
        
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist()

def test():
    
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to("cuda")
    
    pretrained_model_path = './checkpoint_COCO/'
    logdir = './inference_COCO'
    num_inference_steps = 40
    guidance_scale = 7.0
    image_guidance_scale = 3.5
    num_sample_per_prompt = 10
    stage = "multi-image-condition"
    mixed_precision = "fp16"

    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    accelerator = Accelerator(mixed_precision=mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    val_dataset = COCOValMultiSegDataset(root="./COCO2017/")
    val_data = DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=False)
    
    print(val_dataset.__len__())

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    
    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed" f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("StoryGen-COCO")

    vae.eval()
    text_encoder.eval()
    unet.eval()
    
    for i, batch in tqdm(enumerate(val_data)):
        print(i)
        ref_image = batch['ref_image']
        prompt = batch['prompt']
        ref_prompt = batch['ref_prompt']
        
        image_name = batch['image_path'][0].split('/')[-1]
        image_path = os.path.join(logdir, image_name)

        if os.path.exists(image_path):
            continue
        
        sample_seeds = torch.randint(0, 100000, (num_sample_per_prompt,))
        sample_seeds = sorted(sample_seeds.numpy().tolist())
    
        generator = []
        for seed in sample_seeds:
            generator_temp = torch.Generator(device=accelerator.device)
            generator_temp.manual_seed(seed)
            generator.append(generator_temp)
        with torch.no_grad():
            output = pipeline(
                stage = stage,
                prompt = prompt,
                image_prompt = ref_image,
                prev_prompt = ref_prompt,
                height = 512,
                width = 512,
                generator = generator,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                image_guidance_scale = image_guidance_scale,
                num_images_per_prompt=num_sample_per_prompt,
            ).images
        
        images = []
        for i, image in enumerate(output):
            images.append(image[0])
        scores = calc_probs(processor, model, prompt, images)
        index = np.argmax(scores)
        images[index].save(image_path)
        
        break

if __name__ == "__main__":
    test()

# CUDA_VISIBLE_DEVICES=0 accelerate launch inference_COCO_val.py