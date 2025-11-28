import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
from diffusers import DDIMScheduler, AutoencoderKLTemporalDecoder
from transformers import CLIPTextModel, CLIPTokenizer

from backbones.video_diffusion_sd.models.unet_3d_condition import (
    UNetPseudo3DConditionModel,
)
from backbones.video_diffusion_sd.pipelines.stable_diffusion import (
    SpatioTemporalStableDiffusionPipeline,
)
from backbones.video_diffusion_sd.pnp_utils import (
    register_spatial_attention_pnp,
    latent_adain,
)
from src.util import save_folder, save_videos_grid, load_ddim_latents_at_t, seed_everything
from utils import logger, get_exp_dir

def main(
    pretrained_model_path: str,
    content_inv_path: str,
    style_inv_path: str,
    mask_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
    #
    time_steps: int = 50,
    seed: Optional[int] = 33,
    **kwargs,
):
    if seed is not None:
        seed_everything(seed)
    
    # Load model
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    ).requires_grad_(False)

    # use 3d vae for more stable results
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae"
    ).requires_grad_(False)
    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_path, "unet")
    ).requires_grad_(False)

    # set device
    text_encoder = text_encoder.to(weight_dtype).cuda()
    vae = vae.to(weight_dtype).cuda()
    unet = unet.to(weight_dtype).cuda()

    # custom pipe
    pipe = SpatioTemporalStableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(
            pretrained_model_path, subfolder="scheduler"
        ),
    )

    # Load inversion noises
    content_inv_noise = (
        load_ddim_latents_at_t(time_steps, ddim_latents_path=content_inv_path)
        .to(weight_dtype)
        .cuda()
    )
    style_inv_noise = (
        load_ddim_latents_at_t(time_steps, ddim_latents_path=style_inv_path)
        .to(weight_dtype)
        .cuda()
    )
    
    # Check if frame counts match, if not, adjust style_inv_noise
    # Shape of latents: (b, c, f, h, w)
    content_frames = content_inv_noise.shape[2]
    style_frames = style_inv_noise.shape[2]
    
    if content_frames != style_frames:
        logger.info(f"Warning: Content inversion has {content_frames} frames, Style inversion has {style_frames} frames, adjusting...")
        if style_frames == 1:
            # If style has only 1 frame, repeat it to match content frames
            style_inv_noise = style_inv_noise.repeat(1, 1, content_frames, 1, 1)
            logger.info(f"Repeated style frames to match content frames: {content_frames}")
            
        elif content_frames > style_frames:
            # If content has more frames, repeat style frames
            repeat_times = content_frames // style_frames
            remainder = content_frames % style_frames
            style_inv_noise_repeated = style_inv_noise.repeat(1, 1, repeat_times, 1, 1)
            if remainder > 0:
                style_inv_noise_remainder = style_inv_noise[:, :, -remainder:, :, :]
                style_inv_noise = torch.cat([style_inv_noise_repeated, style_inv_noise_remainder], dim=2)
            else:
                style_inv_noise = style_inv_noise_repeated
            logger.info(f"Repeated style frames to match content frames: {content_frames}")
        else:
            # If style has more frames, truncate to match content frames
            style_inv_noise = style_inv_noise[:, :, :content_frames, :, :]
            logger.info(f"Truncated style frames to match content frames: {content_frames}")
    
    # Init Pnp, modify attention forward
    register_spatial_attention_pnp(pipe)

    # Check if using batch processing
    content_name = content_inv_path.split("/")[-2]
    batches_dir = os.path.join(os.path.dirname(os.path.dirname(content_inv_path)), content_name, "frames", "batches")
    batch_info_file = os.path.join(batches_dir, "batch_info.json")
    use_batches = os.path.exists(batch_info_file)
    
    if use_batches:
        # Batch processing mode
        logger.info("Detected batch processing, will perform style transfer for each batch separately...")
        with open(batch_info_file, "r") as f:
            batch_data = json.load(f)
        
        batch_info = batch_data['batches']
        video_fps = batch_data.get('video_fps', 30.0)
        logger.info(f"Total {len(batch_info)} batches to process, original video fps: {video_fps:.2f}")
        
        # Process each batch
        all_samples = []
        for batch_idx, batch in enumerate(batch_info):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batch_info)}: frames {batch['start_frame']}-{batch['end_frame']-1}")
            
            # Load inversion results for this batch
            batch_inv_path = os.path.join(content_inv_path, f"batch_{batch_idx:03d}")
            content_inv_noise = (
                load_ddim_latents_at_t(time_steps, ddim_latents_path=batch_inv_path)
                .to(weight_dtype)
                .cuda()
            )
            style_inv_noise = (
                load_ddim_latents_at_t(time_steps, ddim_latents_path=style_inv_path)
                .to(weight_dtype)
                .cuda()
            )
            
            # Adjust style frames to match content frames
            content_frames = content_inv_noise.shape[2]
            style_frames = style_inv_noise.shape[2]
            if content_frames != style_frames:
                if style_frames == 1:
                    style_inv_noise = style_inv_noise.repeat(1, 1, content_frames, 1, 1)
                elif content_frames > style_frames:
                    repeat_times = content_frames // style_frames
                    remainder = content_frames % style_frames
                    style_inv_noise_repeated = style_inv_noise.repeat(1, 1, repeat_times, 1, 1)
                    if remainder > 0:
                        style_inv_noise_remainder = style_inv_noise[:, :, -remainder:, :, :]
                        style_inv_noise = torch.cat([style_inv_noise_repeated, style_inv_noise_remainder], dim=2)
                    else:
                        style_inv_noise = style_inv_noise_repeated
                else:
                    style_inv_noise = style_inv_noise[:, :, :content_frames, :, :]
            
            # Init latent-shift
            inv_latents_at_t = latent_adain(content_inv_noise, style_inv_noise)
            
            # Perform style transfer for each batch
            batch_sample = pipe.video_style_transfer(
                "",
                latents=inv_latents_at_t,
                num_inference_steps=time_steps,
                content_inv_path=batch_inv_path,
                style_inv_path=style_inv_path,
                mask_path=mask_path,
                output_type="tensor",
            ).images
            
            if isinstance(batch_sample, np.ndarray):
                batch_sample = torch.from_numpy(batch_sample)
            batch_sample = torch.clamp(batch_sample, 0.0, 1.0)
            batch_sample = batch_sample.permute(0, 4, 1, 2, 3).contiguous()
            
            # If first batch, take all; otherwise skip overlapping frames
            overlap = batch_data.get('overlap', 2)
            if batch_idx == 0:
                all_samples.append(batch_sample)
            else:
                # Skip first overlap frames
                if batch_sample.shape[2] > overlap:
                    all_samples.append(batch_sample[:, :, overlap:, :, :])
                else:
                    all_samples.append(batch_sample)
        
        # Merge results from all batches
        logger.info("Merging results from all batches...")
        sample = torch.cat(all_samples, dim=2)  # Concatenate along frame dimension
        logger.info(f"Total frames after merging: {sample.shape[2]}")
    else:
        # Single batch mode
        logger.info("Processing single batch mode...")
        # Init latent-shift
        inv_latents_at_t = latent_adain(content_inv_noise, style_inv_noise)
        
        # video_style_transfer
        sample = pipe.video_style_transfer(
            "",
            latents=inv_latents_at_t,
            num_inference_steps=time_steps,
            content_inv_path=content_inv_path,
            style_inv_path=style_inv_path,
            mask_path=mask_path,
            output_type="tensor",  # Make sure output is tensor
        ).images
        # If sample is a numpy array, convert to tensor
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        sample = torch.clamp(sample, 0.0, 1.0)
        sample = sample.permute(0, 4, 1, 2, 3).contiguous()
        
        # Try to read original video's fps
        fps_file = os.path.join(os.path.dirname(os.path.dirname(content_inv_path)), content_name, "frames", "video_fps.txt")
        video_fps = 30.0  # Default fps
        if os.path.exists(fps_file):
            try:
                with open(fps_file, "r") as f:
                    video_fps = float(f.read().strip())
                logger.info(f"Using original video fps: {video_fps:.2f}")
            except:
                logger.warning(f"Failed to read fps file, using default fps: {video_fps}")
        else:
            logger.warning(f"Fps file not found, using default fps: {video_fps}")
    
    # save
    output_path = os.path.join(
        output_path,
        "sd",
        f'{str(content_inv_path).split("/")[-2]}_{str(style_inv_path).split("/")[-2]}',
    )
    os.makedirs(output_path, exist_ok=True)
    
    # Save each frame as an image (optional, for debugging)
    frames_dir = os.path.join(output_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    save_folder(sample, frames_dir)
    
    # Compose video (using original video's fps to maintain original duration)
    output_video_path = os.path.join(output_path, "output_video.mp4")
    logger.info(f"Composing final video using original video fps ({video_fps:.2f})...")
    save_videos_grid(sample, output_video_path, fps=int(video_fps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path", type=str, default="stable-diffusion-v1-5"
    )
    # parser.add_argument(
    #     "--pretrained_model_path", type=str, 
    #     default="stabilityai/stable-diffusion-2-1-base"
    # )
    parser.add_argument(
        "--content_inv_path",
        type=str,
        default=get_exp_dir() + "results/contents-inv/sd/mallard-fly/inversion",
    )
    parser.add_argument(
        "--style_inv_path", type=str, default=get_exp_dir() + "results/styles-inv/sd/0/inversion"
    )
    parser.add_argument(
        "--mask_path", type=str, default=None, required=False,
        help="Optional mask path. If not provided, mask will not be used."
    )
    # parser.add_argument("--mask_path", type=str, default="results/masks/sd/mallard-fly")
    parser.add_argument("--output_path", type=str, default=get_exp_dir() + "results/stylizations")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
