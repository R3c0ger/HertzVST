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
from src.util import seed_everything, extract_video_frames, load_ddim_latents_at_t
from utils import logger

# The decord package should be imported after torch
import decord
decord.bridge.set_bridge("torch")


def merge_inversion_chunks(chunk_info, inversion_path, ft_path):
    """
    Merge inversion results from multiple chunks.
    
    Args:
        chunk_info: List of chunk information
        inversion_path: Inversion results save path
        ft_path: Feature save path
    """    
    # Get total frames and time steps
    total_frames = chunk_info[-1]['end_frame']
    # Assume 50 steps (can be inferred from the first file)
    time_steps = 50
    
    # Merge latents for each time step
    logger.info(f"Merging inversion results for {time_steps} time steps...")
    for t in range(1, time_steps + 1):
        merged_latents = []
        overlap = 2  # Overlapping frames
        
        for chunk_idx, chunk in enumerate(chunk_info):
            chunk_inversion_path = os.path.join(inversion_path, f"chunk_{chunk_idx:03d}")
            chunk_latents = load_ddim_latents_at_t(t, chunk_inversion_path)
            
            # If it's the first chunk, take all
            if chunk_idx == 0:
                merged_latents.append(chunk_latents)
            else:
                # For subsequent chunks, skip the overlapping frames
                # latents shape: (b, c, f, h, w)
                if chunk_latents.shape[2] > overlap:
                    merged_latents.append(chunk_latents[:, :, overlap:, :, :])
                else:
                    merged_latents.append(chunk_latents)
        
        # Merge all latents
        if len(merged_latents) > 1:
            merged_latent = torch.cat(merged_latents, dim=2)  # Concatenate along frame dimension
        else:
            merged_latent = merged_latents[0]
        
        # Save merged latents
        merged_latent_path = os.path.join(inversion_path, f"ddim_latents_{t}.pt")
        torch.save(merged_latent, merged_latent_path)
    
    logger.info(f"Inversion results merged, total frames: {merged_latent.shape[2]}")


def main(
    pretrained_model_path: str,
    content_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
    #
    height: int = 512,
    width: int = 512,
    time_steps: int = 50,
    max_frames: int = 30,  # Maximum frames to process to avoid OOM
    overlap_frames: int = 2,
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
    actual_num_frames = 16  # Use the provided num_frames by default
    if content_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
        # If it's a video file, split it into frames first
        content_name = content_name.rsplit(".", 1)[0]  # Remove extension
        frames_dir = os.path.join(original_output_path, "sd", content_name, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        logger.info(f"Splitting video into frames: {content_path}")
        extracted_frames, video_fps = extract_video_frames(
            content_path, 
            frames_dir, 
            image_size=(width, height),
            max_frames=None  # Extract all frames to preserve original duration
        )
        logger.info(f"Extracted {extracted_frames} frames to {frames_dir}, video fps: {video_fps:.2f}")
        
        # Save fps info for later use
        fps_file = os.path.join(frames_dir, "video_fps.txt")
        with open(fps_file, "w") as f:
            f.write(str(video_fps))
        
        # If extracted frames exceed max_frames, use chunked processing
        if extracted_frames > max_frames:
            logger.info(f"Video has {extracted_frames} frames, exceeding max processing limit {max_frames}; will use chunked processing")
            # Calculate number of chunks needed (each chunk has 16 frames, with overlap)
            chunk_size = max_frames
            # Overlapping frames between chunks for smooth transition
            step_size = chunk_size - overlap_frames  # Frame interval between chunk starts
            
            num_chunks = (extracted_frames - overlap_frames + step_size - 1) // step_size
            logger.info(
                f"Will split into {num_chunks} chunks, "
                f"each with {chunk_size} frames and {overlap_frames} overlapping frames"
            )
            
            # Create chunks directory
            chunks_dir = os.path.join(frames_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            
            # Create directory and copy frames for each chunk
            chunk_info = []
            for chunk_idx in range(num_chunks):
                start_frame = chunk_idx * step_size
                end_frame = min(start_frame + chunk_size, extracted_frames)
                actual_chunk_frames = end_frame - start_frame
                
                chunk_dir = os.path.join(chunks_dir, f"chunk_{chunk_idx:03d}")
                os.makedirs(chunk_dir, exist_ok=True)
                
                # Copy frames to chunk directory
                for i in range(actual_chunk_frames):
                    src_frame = os.path.join(frames_dir, f"%05d.png" % (start_frame + i))
                    dst_frame = os.path.join(chunk_dir, f"%05d.png" % i)
                    if os.path.exists(src_frame):
                        shutil.copy2(src_frame, dst_frame)
                
                chunk_info.append({
                    'chunk_dir': chunk_dir,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'num_frames': actual_chunk_frames
                })
                logger.info(f"Chunk {chunk_idx}: frames {start_frame}-{end_frame-1} ({actual_chunk_frames} frames)")
            
            # Save chunk info
            import json
            chunk_info_file = os.path.join(chunks_dir, "chunk_info.json")
            with open(chunk_info_file, "w") as f:
                json.dump({
                    'total_frames': extracted_frames,
                    'chunk_size': chunk_size,
                    'overlap': overlap_frames,
                    'step_size': step_size,
                    'num_chunks': num_chunks,
                    'chunks': chunk_info
                }, f, indent=2)
            
            # Use the first chunk as content_path (all chunks will be processed later)
            content_path = chunk_info[0]['chunk_dir']
            actual_num_frames = chunk_info[0]['num_frames']
            # Mark that chunked processing is needed
            use_chunks = True
        else:
            # Use the actual number of extracted frames
            actual_num_frames = extracted_frames
            # Use frame directory as content_path
            content_path = frames_dir
            use_chunks = False
    
    output_path = os.path.join(original_output_path, "sd", content_name)
    inversion_path = os.path.join(output_path, "inversion")
    reconstruction_path = os.path.join(output_path, "reconstruction")
    ft_path = os.path.join(output_path, "features")
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    os.makedirs(ft_path, exist_ok=True)
    
    # If using chunked processing
    if use_chunks:
        chunks_dir = os.path.join(original_output_path, "sd", content_name, "frames", "chunks")
        chunk_info_file = os.path.join(chunks_dir, "chunk_info.json")
        with open(chunk_info_file, "r") as f:
            chunk_data = json.load(f)
        
        chunk_info = chunk_data['chunks']
        logger.info(f"Starting processing of {len(chunk_info)} chunks...")
        
        # Create separate inversion directories for each chunk
        for chunk_idx, chunk in enumerate(chunk_info):
            logger.info(f"\nProcessing chunk {chunk_idx + 1}/{len(chunk_info)}: frames {chunk['start_frame']}-{chunk['end_frame']-1}")
            chunk_inversion_path = os.path.join(inversion_path, f"chunk_{chunk_idx:03d}")
            chunk_reconstruction_path = os.path.join(reconstruction_path, f"chunk_{chunk_idx:03d}")
            chunk_ft_path = os.path.join(ft_path, f"chunk_{chunk_idx:03d}")
            os.makedirs(chunk_inversion_path, exist_ok=True)
            os.makedirs(chunk_reconstruction_path, exist_ok=True)
            os.makedirs(chunk_ft_path, exist_ok=True)
            
            with torch.no_grad():
                content_inversion_reconstruction(
                    pipe,
                    ddim_inv_scheduler,
                    chunk['chunk_dir'],
                    chunk_inversion_path,
                    chunk_reconstruction_path,
                    height,
                    width,
                    time_steps,
                    weight_dtype,
                    ft_indices=[ft_indices],
                    ft_timesteps=[ft_timesteps],
                    ft_path=chunk_ft_path,
                    is_opt=is_opt,
                )
            
            # Clear memory after processing each chunk
            torch.cuda.empty_cache()
        
        # Merge inversion results from all chunks
        logger.info(f"\nMerging inversion results from {len(chunk_info)} chunks...")
        merge_inversion_chunks(chunk_info, inversion_path, ft_path)
        logger.info("Content inversion completed (chunked processing)")
    else:
        # Single-chunk processing (original logic)
        with torch.no_grad():
            content_inversion_reconstruction(
                pipe,
                ddim_inv_scheduler,
                content_path,
                inversion_path,
                reconstruction_path,
                height,
                width,
                time_steps,
                weight_dtype,
                ft_indices=[ft_indices],
                ft_timesteps=[ft_timesteps],
                ft_path=ft_path,
                is_opt=is_opt,
            )
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        logger.info(f"Content inversion completed, actual number of frames processed: {actual_num_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path", type=str, default="stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--content_path", type=str, default="examples/contents/mallard-fly"
    )
    parser.add_argument("--output_path", type=str, default="results/contents-inv")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--max_frames", type=int, default=30, 
        help="Maximum number of frames to process to avoid OOM (default: 30; adjust based on GPU memory)")
    parser.add_argument("--overlap_frames", type=int, default=2, 
        help="Number of overlapping frames between chunks for smooth transition")
    #
    parser.add_argument("--ft_indices", type=int, default=2)
    parser.add_argument("--ft_timesteps", type=int, default=301)
    parser.add_argument("--is_opt", action="store_true", help="use Easy-Inv")
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
