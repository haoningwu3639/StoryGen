import argparse, os
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB").resize([864, 480]))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L").resize([864, 480]))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)
    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default='../../StorySalon/', type=str)
    parser.add_argument("--outdir", default='../../StorySalon/image_inpainted', type=str)
    parser.add_argument("--steps", type=int, default=25)
    opt = parser.parse_args()

    image_dir = os.path.join(opt.indir, 'image')
    mask_dir = os.path.join(opt.indir, 'mask')
    folders = sorted(os.listdir(image_dir))

    for folder in folders:
        if not os.path.exists(os.path.join(opt.outdir, folder)):
            print(os.path.join(opt.outdir, folder))
            os.mkdir(os.path.join(opt.outdir, folder))

    image_folders = [os.path.join(image_dir, folder) for folder in folders]
    mask_folders = [os.path.join(mask_dir, folder) for folder in folders]
    image_list = []
    mask_list = []

    for video in image_folders: # video: image_folder, /StorySalon/image/00001
        images = sorted(os.listdir(video))
        for image in images:
            image_list.append(os.path.join(video, image))

    for video in mask_folders: # video: mask_folder, /StorySalon/mask/00001
        masks = sorted(os.listdir(video))
        for mask in masks:
            mask_list.append(os.path.join(video, mask))

    outpath_list = [x.replace("image", "image_inpainted") for x in image_list]
    print(f"Found {len(mask_list)} inputs.")

    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask, outpath in tqdm(zip(image_list, mask_list, outpath_list)):
                batch = make_batch(image, mask, device=device)
                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1] - 1,) + c.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps, conditioning=c, batch_size=c.shape[0], shape=shape, verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                inpainted = (1 - mask) * image + mask * predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255                
                inpainted = Image.fromarray(inpainted.astype(np.uint8))
                inpainted = inpainted.resize([854, 480])
                inpainted.save(outpath)