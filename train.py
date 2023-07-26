import argparse
import os
import cv2
from typing import Optional, Dict

from omegaconf import OmegaConf

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.cuda.amp import autocast

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from utils.util import get_function_args
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StoryGenPipeline
from dataset import SimpleDataset, StorySalonDataset

logger = get_logger(__name__)

class SampleLogger:
    def __init__(
        self,
        image_logdir: str,
        num_samples_per_prompt: int = 1,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.0,
    ) -> None:
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_sample_per_prompt = num_samples_per_prompt
        self.image_logdir = image_logdir
        os.makedirs(self.image_logdir, exist_ok=True)
        
    def log_sample_images(
        self, batch, visual_projection, image_encoder, cross_frame_attn, pipeline: StoryGenPipeline, device: torch.device, step: int
    ):
        sample_seeds = torch.randint(0, 100000, (self.num_sample_per_prompt,))
        sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds
        self.prompts = batch["prompt"]
        for idx, prompt in enumerate(tqdm(self.prompts, desc="Generating sample images")):
            ref_image = batch["ref_image"][idx, :, :, :].unsqueeze(0)
            image = batch["image"][idx, :, :, :].unsqueeze(0)
            ref_image = ref_image.to(device=device)
            image = image.to(device=device)
            
            ref_img_feature = image_encoder(ref_image).last_hidden_state
            projected_ref_img_feature = visual_projection(ref_img_feature)
            cross_frame_feature = projected_ref_img_feature
                
            generator = []
            for seed in self.sample_seeds:
                generator_temp = torch.Generator(device=device)
                generator_temp.manual_seed(seed)
                generator.append(generator_temp)
                
            sequence = pipeline(
                cond=cross_frame_feature, # past frame,
                prompt=prompt, # current frame
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_images_per_prompt=self.num_sample_per_prompt,
                cross_frame_attn=cross_frame_attn,
            ).images

            image = (image + 1.) / 2. # for visualization
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            ref_image = (ref_image + 1.) / 2. # for visualization
            ref_image = ref_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()    
            cv2.imwrite(os.path.join(self.image_logdir, f"{step}_{idx}_{seed}_gt.png"), image[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(self.image_logdir, f"{step}_{idx}_{seed}_ref.png"), ref_image[:, :, ::-1] * 255)
            with open(os.path.join(self.image_logdir, f"{step}_{idx}_{seed}" + '.txt'), 'a') as f:
                f.write(batch['prompt'][idx])
            for i, img in enumerate(sequence):
                img[0].save(os.path.join(self.image_logdir, f"{step}_{idx}_{sample_seeds[i]}_output.png"))
            
def train(
    pretrained_model_path: str,
    logdir: str,
    image_logdir: str,
    train_steps: int = 5000,
    validation_steps: int = 1000,
    validation_sample_logger: Optional[Dict] = None,
    gradient_accumulation_steps: int = 10, # important hyper-parameter
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    train_batch_size: int = 1,
    val_batch_size: int = 1,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_warmup_steps: int = 0,
    use_8bit_adam: bool = True,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    checkpointing_steps: int = 10000,
    cross_frame_attn: bool = False,
):
    
    args = get_function_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    # Load models and create wrapper
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    if not cross_frame_attn: # Stage 1
        # Initialize the TextEncoder and ImageEncoder from pre-trained CLIP-large
        text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="CLIP")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="CLIP")
        # Train from scratch
        unet = UNet2DConditionModel.from_config(pretrained_model_path, subfolder="unet")
        # Load StableDiffusion Unet to initialize our StoryGen.
        # Note: LoRA layers have already been initialized to zero in attention.py
        unet.load_SDM_state_dict(torch.load("./ckpt/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin", map_location="cpu"))
    else: # Stage 2
        text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    
    pipeline = StoryGenPipeline(
        vae=vae,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    if not cross_frame_attn: # Stage 1, train LoRA layers for Style Transfer Module
        for name, module in unet.named_modules():
            if name.endswith("attn2"):
                for n, m in module.named_modules():
                    if n =="add_v_proj" or n =="add_k_proj" or n =="add_v_proj" or n =="add_out_proj":
                        for params in m.parameters():
                            params.requires_grad = True
                            
    else: # Stage 2, train Context Module
        trainable_modules = ("attn1_cross", "attn2_cross", "ff_cross")
        for name, module in unet.named_modules():
            if name.endswith(trainable_modules):
                for params in module.parameters():
                    params.requires_grad = True
                
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataset = SimpleDataset(root="./data/")
    val_dataset = SimpleDataset(root="./data/")
    # Actually, you should use StorySalon Dataset to train the StoryGen model.
    # train_dataset = StorySalonDataset(root="./StorySalon/", dataset_name='train')
    # val_dataset = StorySalonDataset(root="./StorySalon/", dataset_name='test')
    
    print(train_dataset.__len__())
    print(val_dataset.__len__())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

    pipeline, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipeline, unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # For mixed precision training we cast the text_encoder, image_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    # Extract the visual projection layer of CLIP Image Encoder
    visual_projection = image_encoder.visual_projection
    
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("StoryGen")
    step = 0

    if validation_sample_logger is not None and accelerator.is_main_process:
        validation_sample_logger = SampleLogger(**validation_sample_logger, image_logdir=image_logdir, num_samples_per_prompt=val_batch_size)

    progress_bar = tqdm(range(step, train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            # If the process is stuck somewhere, try to comment out this line
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)
    val_data_yielder = make_data_yielder(val_dataloader)

    while step < train_steps:
        batch = next(train_data_yielder)
        vae.eval()
        text_encoder.eval()
        image_encoder.eval()
        unet.train()
        
        ref_image = batch["ref_image"].to(dtype=weight_dtype)
        image = batch["image"].to(dtype=weight_dtype)
        prompt = batch["prompt"]
        mask = batch["mask"].to(dtype=weight_dtype)
        mask = mask.repeat(1, 4, 1, 1) # 1 channels to 4 channels
        mask = F.interpolate(mask, scale_factor = 1 / 8., mode="bilinear", align_corners=False)
        
        b, c, h, w = image.shape
        prompt_ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        ref_img_feature = image_encoder(ref_image).last_hidden_state
        projected_ref_img_feature = visual_projection(ref_img_feature)
        
        latents = vae.encode(image).latent_dist.sample()
        latents = latents * 0.18215
        
        noise = torch.randn_like(latents) # [-1, 1]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise according to the noise magnitude at each timestep (this is the forward diffusion process)
        noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)
        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(prompt_ids.to(accelerator.device)).last_hidden_state # B * 77 * 768
        cross_frame_feature = projected_ref_img_feature
        
        # Predict the noise residual
        target = noise
        model_pred = unet(noisy_latent, timesteps, cross_frame_attn, cross_frame_feature, encoder_hidden_states).sample
        # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = F.mse_loss(model_pred.float() * (1. - mask), target.float() * (1. - mask), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1
            if accelerator.is_main_process:
                if validation_sample_logger is not None and step % validation_steps == 0:
                    unet.eval()
                    val_batch = next(val_data_yielder)
                    with autocast():
                        validation_sample_logger.log_sample_images(
                            batch=val_batch,
                            visual_projection=visual_projection, 
                            image_encoder=image_encoder,
                            cross_frame_attn=cross_frame_attn,
                            pipeline=pipeline,
                            device=accelerator.device,
                            step=step,
                        )
                if step % checkpointing_steps == 0:
                    pipeline_save = StoryGenPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        image_encoder=image_encoder,
                        tokenizer=tokenizer,
                        unet=accelerator.unwrap_model(unet),
                        scheduler=scheduler,
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_{step}")
                    pipeline_save.save_pretrained(checkpoint_save_path)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)
    accelerator.end_training()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_stage', default=1, type=int)
    args = parser.parse_args()
    if args.training_stage == 1:
        config = './config/stage1_config.yml'
    elif args.training_stage == 2:
        config = './config/stage2_config.yml'
    else:
        raise ValueError("Wrong Trainig Stage Hyperparameter!")
    
    train(**OmegaConf.load(config))