import os
import cv2
import random
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
from transformers import AutoTokenizer, CLIPTextModel

from utils.util import get_time_string, get_function_args
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline
from dataset import StorySalonDataset

logger = get_logger(__name__)

class SampleLogger:
    def __init__(
        self,
        logdir: str,
        subdir: str = "sample",
        stage: str = 'no',
        num_samples_per_prompt: int = 1,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.0,
        image_guidance_scale: float = 3.5,
    ) -> None:
        self.stage = stage
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_sample_per_prompt = num_samples_per_prompt
        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir)
        
    def log_sample_images(
        self, batch, pipeline: StableDiffusionPipeline, device: torch.device, step: int
    ):
        sample_seeds = torch.randint(0, 100000, (self.num_sample_per_prompt,))
        sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds
        self.prompts = batch["prompt"]
        self.prev_prompts = batch["ref_prompt"]

        for idx, prompt in enumerate(tqdm(self.prompts, desc="Generating sample images")):
            image = batch["image"][idx, :, :, :].unsqueeze(0)
            ref_images = batch["ref_image"][idx, :, :, :, :].unsqueeze(0)
            image = image.to(device=device)
            ref_images = ref_images.to(device=device)
            generator = []
            for seed in self.sample_seeds:
                generator_temp = torch.Generator(device=device)
                generator_temp.manual_seed(seed)
                generator.append(generator_temp) 
            sequence = pipeline(
                stage = self.stage,
                prompt = prompt,
                image_prompt = ref_images,
                prev_prompt = self.prev_prompts,
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                image_guidance_scale = self.image_guidance_scale,
                num_images_per_prompt=self.num_sample_per_prompt,
            ).images

            image = (image + 1.) / 2. # for visualization
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}_{seed}.png"), image[:, :, ::-1] * 255)
            v_refs = []
            ref_images = ref_images.squeeze(0)
            for ref_image in ref_images:
                # v_ref = (ref_image + 1.) / 2. # for visualization
                v_ref = ref_image.permute(1, 2, 0).detach().cpu().numpy()
                v_refs.append(v_ref)
            for i in range(len(v_refs)):
                cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}_{seed}_ref_{i}.png"), v_refs[i][:, :, ::-1] * 255)
                
            with open(os.path.join(self.logdir, f"{step}_{idx}_{seed}" + '.txt'), 'a') as f:
                f.write(batch['prompt'][idx])
                f.write('\n')
                f.write('\n')
                for prev_prompt in self.prev_prompts:
                    f.write(prev_prompt[0])
                    f.write('\n')
            for i, img in enumerate(sequence):
                img[0].save(os.path.join(self.logdir, f"{step}_{idx}_{sample_seeds[i]}_output.png"))
            
def train(
    pretrained_model_path: str,
    logdir: str,
    train_steps: int = 300,
    validation_steps: int = 1000,
    validation_sample_logger: Optional[Dict] = None,
    gradient_accumulation_steps: int = 30, # important hyper-parameter
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    train_batch_size: int = 4,
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
    checkpointing_steps: int = 2000,
):
    
    args = get_function_args()
    time_string = get_time_string()
    logdir += f"_{time_string}"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_config(pretrained_model_path, subfolder="unet")
    pretrained_sdm = torch.load('./ckpt/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin', map_location='cpu')
    unet.load_SDM_state_dict(pretrained_sdm)
    # unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    pipeline = StableDiffusionPipeline(
        vae=vae,
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
    unet.requires_grad_(False)

    trainable_modules = ("attn1")
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

    train_dataset = StorySalonDataset(root="./StorySalon/", dataset_name='train')
    val_dataset = StorySalonDataset(root="./StorySalon/", dataset_name='test')
    
    print(train_dataset.__len__())
    print(val_dataset.__len__())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("StoryGen-Single")
    step = 0

    if validation_sample_logger is not None and accelerator.is_main_process:
        validation_sample_logger = SampleLogger(**validation_sample_logger, logdir=logdir)

    progress_bar = tqdm(range(step, train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)
    val_data_yielder = make_data_yielder(val_dataloader)

    while step < train_steps:
        batch = next(train_data_yielder)
        
        vae.eval()
        text_encoder.eval()
        unet.train()
        
        image = batch["image"].to(dtype=weight_dtype)
        prompt = batch["prompt"]
        prompt_ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        mask = batch["mask"].to(dtype=weight_dtype)        
        mask = mask[:, [0], :, :].repeat(1, 4, 1, 1) # 3 channels to 4 channels
        mask = F.interpolate(mask, scale_factor = 1 / 8., mode="bilinear", align_corners=False)
        b, c, h, w = image.shape
        
        latents = vae.encode(image).latent_dist.sample()
        latents = latents * 0.18215
        
        # Sample noise that we'll add
        noise = torch.randn_like(latents) # [-1, 1]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=latents.device)
        timesteps = timesteps.long()
        
        # Add noise according to the noise magnitude at each timestep (this is the forward diffusion process)
        noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)
        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(prompt_ids.to(accelerator.device))[0] # B * 77 * 768
        
        # Predict the noise residual
        model_pred = unet(noisy_latent, timesteps, encoder_hidden_states=encoder_hidden_states, image_hidden_states=None, return_dict=False)[0]
        
        # loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        loss = F.mse_loss(model_pred.float() * (1. - mask), noise.float() * (1 - mask), reduction="mean")

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
                            batch = val_batch,
                            pipeline=pipeline,
                            device=accelerator.device,
                            step=step,
                        )
                if step % checkpointing_steps == 0:
                    pipeline_save = StableDiffusionPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
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
    config = "./config/stage1_config.yml"
    train(**OmegaConf.load(config))

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_StorySalon.py
