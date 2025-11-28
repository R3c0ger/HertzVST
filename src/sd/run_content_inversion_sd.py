import argparse
import json
import os
import shutil
from typing import Optional

import torch
from diffusers import AutoencoderKLTemporalDecoder
from diffusers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from backbones.video_diffusion_sd.models.unet_3d_condition import (
    UNetPseudo3DConditionModel,
)
from backbones.video_diffusion_sd.pipelines.stable_diffusion import (
    SpatioTemporalStableDiffusionPipeline,
)
from inversion_tools.ddim_inversion import content_inversion_reconstruction
from src.util import seed_everything, extract_video_frames
from utils import logger, get_exp_dir

# The decord package should be imported after torch
import decord
decord.bridge.set_bridge("torch")


def main(
    pretrained_model_path: str,
    content_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
    #
    num_frames: int = 16,
    height: int = 512,
    width: int = 512,
    time_steps: int = 50,
    #
    ft_indices: int = None,
    ft_timesteps: int = None,
    is_opt: bool = True,
    seed: Optional[int] = 33,
    **kwargs,
):
    if seed is not None:
        seed_everything(seed)

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

    # inversion scheduler
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler"
    )
    ddim_inv_scheduler.set_timesteps(time_steps)

    # make dir
    original_output_path = output_path
    content_name = content_path.split("/")[-1]
    actual_num_frames = num_frames  # default to using the passed num_frames
    
    if content_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
        # If it's a video file, first split it into frames
        content_name = content_name.rsplit(".", 1)[0]  # Remove extension
        frames_dir = os.path.join(original_output_path, "sd", content_name, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        logger.info(f"Splitting video into frames: {content_path}")
        
        # First check total video frames
        vr_temp = decord.VideoReader(content_path, width=width, height=height)
        total_video_frames = len(vr_temp)
        try:
            video_fps = vr_temp.get_avg_fps()
        except:
            video_fps = 30.0
        
        # Extract all frames (no sampling)        
        extracted_frames, video_fps = extract_video_frames(
            content_path, 
            frames_dir, 
            image_size=(width, height),
            max_frames=None  # Extract all frames
        )
        actual_num_frames = extracted_frames
        logger.info(f"Extracted {extracted_frames} frames to {frames_dir}, video fps: {video_fps:.2f}")
        
        # If there are too many frames, use batch processing
        batch_size = 16  # Process 16 frames per batch
        if extracted_frames > batch_size:
            logger.info(f"Video has {extracted_frames} frames, exceeding single batch frame count {batch_size}, using batch processing...")
            
            # Calculate how many batches are needed (can have overlap for smooth transitions)
            overlap = 2  # Number of overlapping frames between batches
            step_size = batch_size - overlap  # Starting frame interval for each batch
            
            num_batches = (extracted_frames - overlap + step_size - 1) // step_size
            logger.info(f"Dividing into {num_batches} batches, each with {batch_size} frames, overlapping {overlap} frames")
            
            # Create batches directory
            batches_dir = os.path.join(frames_dir, "batches")
            os.makedirs(batches_dir, exist_ok=True)
            
            # Create directory for each batch and copy frames
            batch_info = []
            for batch_idx in range(num_batches):
                start_frame = batch_idx * step_size
                end_frame = min(start_frame + batch_size, extracted_frames)
                actual_batch_frames = end_frame - start_frame
                
                batch_dir = os.path.join(batches_dir, f"batch_{batch_idx:03d}")
                os.makedirs(batch_dir, exist_ok=True)
                
                # Copy frames to batch directory
                for i in range(actual_batch_frames):
                    src_frame = os.path.join(frames_dir, f"%05d.png" % (start_frame + i))
                    dst_frame = os.path.join(batch_dir, f"%05d.png" % i)
                    if os.path.exists(src_frame):
                        shutil.copy2(src_frame, dst_frame)
                
                batch_info.append({
                    'batch_dir': batch_dir,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'num_frames': actual_batch_frames
                })
                logger.info(f"Batch {batch_idx}: Frames {start_frame}-{end_frame-1} ({actual_batch_frames} frames) saved to {batch_dir}")
            
            # Save batch info
            batch_info_file = os.path.join(batches_dir, "batch_info.json")
            with open(batch_info_file, "w") as f:
                json.dump({
                    'total_frames': extracted_frames,
                    'batch_size': batch_size,
                    'overlap': overlap,
                    'step_size': step_size,
                    'num_batches': num_batches,
                    'batches': batch_info,
                    'video_fps': video_fps
                }, f, indent=2)
            logger.info(f"Batch info saved to: {batch_info_file}")
        
        # Save fps info for later use
        fps_file = os.path.join(frames_dir, "video_fps.txt")
        with open(fps_file, "w") as f:
            f.write(str(video_fps))
        # Use frames directory as content_path
        content_path = frames_dir
    
    output_path = os.path.join(original_output_path, "sd", content_name)
    inversion_path = os.path.join(output_path, "inversion")
    reconstruction_path = os.path.join(output_path, "reconstruction")
    ft_path = os.path.join(output_path, "features")
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    os.makedirs(ft_path, exist_ok=True)
    
    # Check for batch processing
    batches_dir = os.path.join(content_path, "batches")
    batch_info_file = os.path.join(batches_dir, "batch_info.json")
    use_batches = os.path.exists(batch_info_file)
    
    if use_batches:
        # Batch processing mode
        with open(batch_info_file, "r") as f:
            batch_data = json.load(f)
        
        batch_info = batch_data['batches']
        logger.info(f"Detected batch processing, performing inversion on {len(batch_info)} batches...")
        
        # Perform inversion on each batch
        for batch_idx, batch in enumerate(batch_info):
            logger.info(
                f"\nProcessing batch {batch_idx + 1}/{len(batch_info)}: "
                f"Frames {batch['start_frame']}-{batch['end_frame']-1}"
            )
            
            batch_content_path = batch['batch_dir']
            batch_inversion_path = os.path.join(inversion_path, f"batch_{batch_idx:03d}")
            batch_reconstruction_path = os.path.join(reconstruction_path, f"batch_{batch_idx:03d}")
            batch_ft_path = os.path.join(ft_path, f"batch_{batch_idx:03d}")
            os.makedirs(batch_inversion_path, exist_ok=True)
            os.makedirs(batch_reconstruction_path, exist_ok=True)
            os.makedirs(batch_ft_path, exist_ok=True)
            
            with torch.no_grad():
                content_inversion_reconstruction(
                    pipe,
                    ddim_inv_scheduler,
                    batch_content_path,
                    batch_inversion_path,
                    batch_reconstruction_path,
                    batch['num_frames'],  # Use actual number of frames in the batch
                    height,
                    width,
                    time_steps,
                    weight_dtype,
                    ft_indices=[ft_indices],
                    ft_timesteps=[ft_timesteps],
                    ft_path=batch_ft_path,
                    is_opt=is_opt,
                )
        logger.info("All batches inversion completed!")
    else:
        # Single batch processing mode
        with torch.no_grad():
            content_inversion_reconstruction(
                pipe,
                ddim_inv_scheduler,
                content_path,
                inversion_path,
                reconstruction_path,
                actual_num_frames,  # Use actual number of frames
                height,
                width,
                time_steps,
                weight_dtype,
                ft_indices=[ft_indices],
                ft_timesteps=[ft_timesteps],
                ft_path=ft_path,
                is_opt=is_opt,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path", type=str, default="stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--content_path", type=str, default="examples/contents/mallard-fly"
    )
    parser.add_argument("--output_path", type=str, default=get_exp_dir() + "results/contents-inv")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--time_steps", type=int, default=50)
    #
    parser.add_argument("--ft_indices", type=int, default=2)
    parser.add_argument("--ft_timesteps", type=int, default=301)
    parser.add_argument("--is_opt", action="store_true", help="use Easy-Inv")
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
