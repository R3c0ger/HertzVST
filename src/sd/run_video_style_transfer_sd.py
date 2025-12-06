import argparse
import os
import json
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from utils import logger


def process_chunk(
    chunk_idx: int,
    chunk: dict,
    content_inv_path: str,
    style_inv_path: str,
    style_inv_noise: torch.Tensor,
    pretrained_model_path: str,
    mask_path: str,
    time_steps: int,
    weight_dtype: torch.dtype,
    device_id: int = 0,
    overlap: int = 2,
    # Plan A and Plan B parameters
    use_multi_scale_flow: bool = False,
    flow_scales: list = None,
    flow_fusion_method: str = 'weighted_average',
    use_temporal_attention: bool = False,
    temporal_attention_channels: int = 320,
    temporal_attention_heads: int = 8,
    temporal_attention_dropout: float = 0.0,
    temporal_attention_steps: tuple = (20, 30),
):
    if flow_scales is None:
        flow_scales = [1.0, 0.5, 0.25]
    """
    Process a single video chunk.

    Args:
        chunk_idx: Index of the current chunk.
        chunk: Chunk metadata.
        content_inv_path: Path to content inversion results.
        style_inv_path: Path to style inversion results.
        style_inv_noise: Pre-loaded style inversion noise tensor.
        pretrained_model_path: Path to the pre-trained model.
        mask_path: Path to optional mask.
        time_steps: Number of inference steps.
        weight_dtype: Data type for model weights.
        device_id: GPU device ID.
        overlap: Number of overlapping frames between chunks.

    Returns:
        (chunk_idx, chunk_sample): Chunk index and processed result.
    """
    logger.info(f"[GPU {device_id}] Starting processing chunk {chunk_idx + 1}: frames {chunk['start_frame']}-{chunk['end_frame']-1}")
    
    # Set GPU for this thread
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    # Load inversion results for this chunk
    chunk_inv_path = os.path.join(content_inv_path, f"chunk_{chunk_idx:03d}")
    content_inv_noise = (
        load_ddim_latents_at_t(time_steps, ddim_latents_path=chunk_inv_path)
        .to(weight_dtype)
        .to(device)
    )
    
    # Move style_inv_noise to corresponding GPU
    style_inv_noise_gpu = style_inv_noise.to(device)
    
    # Adjust style frame count to match content
    content_frames = content_inv_noise.shape[2]
    style_frames = style_inv_noise_gpu.shape[2]
    if content_frames != style_frames:
        if style_frames == 1:
            style_inv_noise_gpu = style_inv_noise_gpu.repeat(1, 1, content_frames, 1, 1)
        elif content_frames > style_frames:
            repeat_times = content_frames // style_frames
            remainder = content_frames % style_frames
            style_inv_noise_repeated = style_inv_noise_gpu.repeat(1, 1, repeat_times, 1, 1)
            if remainder > 0:
                style_inv_noise_remainder = style_inv_noise_gpu[:, :, -remainder:, :, :]
                style_inv_noise_gpu = torch.cat([style_inv_noise_repeated, style_inv_noise_remainder], dim=2)
            else:
                style_inv_noise_gpu = style_inv_noise_repeated
        else:
            style_inv_noise_gpu = style_inv_noise_gpu[:, :, :content_frames, :, :]
    
    # Init latent-shift
    inv_latents_at_t = latent_adain(content_inv_noise, style_inv_noise_gpu)
    
    # Create independent pipeline per GPU to avoid threading conflicts
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    ).requires_grad_(False).to(weight_dtype).to(device)
    
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae"
    ).requires_grad_(False).to(weight_dtype).to(device)
    
    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_path, "unet")
    ).requires_grad_(False).to(weight_dtype).to(device)
    
    pipe = SpatioTemporalStableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(
            pretrained_model_path, subfolder="scheduler"
        ),
    )
    register_spatial_attention_pnp(pipe)
    
    # Perform style transfer on this chunk
    chunk_sample = pipe.video_style_transfer(
        "",
        latents=inv_latents_at_t,
        num_inference_steps=time_steps,
        content_inv_path=chunk_inv_path,
        style_inv_path=style_inv_path,
        mask_path=mask_path,
        output_type="tensor",
        # Plan A and Plan B parameters
        use_multi_scale_flow=use_multi_scale_flow,
        flow_scales=flow_scales,
        flow_fusion_method=flow_fusion_method,
        use_temporal_attention=use_temporal_attention,
        temporal_attention_channels=temporal_attention_channels,
        temporal_attention_heads=temporal_attention_heads,
        temporal_attention_dropout=temporal_attention_dropout,
        temporal_attention_steps=temporal_attention_steps,
    ).images
    
    if isinstance(chunk_sample, np.ndarray):
        chunk_sample = torch.from_numpy(chunk_sample)
    chunk_sample = torch.clamp(chunk_sample, 0.0, 1.0)
    chunk_sample = chunk_sample.permute(0, 4, 1, 2, 3).contiguous()
    
    # Clean up GPU memory
    del content_inv_noise, style_inv_noise_gpu, inv_latents_at_t
    del pipe, vae, unet, text_encoder
    torch.cuda.empty_cache()
    
    logger.info(f"[GPU {device_id}] Finished processing chunk {chunk_idx + 1}")
    return chunk_idx, chunk_sample


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
    #
    # Plan A: Multi-scale Optical Flow Fusion parameters
    use_multi_scale_flow: bool = False,
    flow_scales: list = None,
    flow_fusion_method: str = 'weighted_average',
    #
    # Plan B: Temporal Attention Enhancement parameters
    use_temporal_attention: bool = False,
    temporal_attention_channels: int = 320,
    temporal_attention_heads: int = 8,
    temporal_attention_dropout: float = 0.0,
    temporal_attention_steps: tuple = (20, 30),
    #
    **kwargs,
):
    if flow_scales is None:
        flow_scales = [1.0, 0.5, 0.25]
    if seed is not None:
        seed_everything(seed)
    
    # Load model
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    ).requires_grad_(False)

    # Use 3D VAE for more stable results
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae"
    ).requires_grad_(False)
    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_path, "unet")
    ).requires_grad_(False)

    # Set device
    text_encoder = text_encoder.to(weight_dtype).cuda()
    vae = vae.to(weight_dtype).cuda()
    unet = unet.to(weight_dtype).cuda()

    # Custom pipeline
    pipe = SpatioTemporalStableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(
            pretrained_model_path, subfolder="scheduler"
        ),
    )

    # Check if chunked processing is used
    content_name = content_inv_path.split("/")[-2]
    chunks_dir = os.path.join("results/contents-inv/sd", content_name, "frames", "chunks")
    chunk_info_file = os.path.join(chunks_dir, "chunk_info.json")
    use_chunks = os.path.exists(chunk_info_file)
    
    # Initialize attention registration (required in both modes)
    register_spatial_attention_pnp(pipe)
    
    if use_chunks:
        # Chunked processing mode
        logger.info("Detected chunked processing; performing style transfer on each chunk separately...")
        with open(chunk_info_file, "r") as f:
            chunk_data = json.load(f)
        
        chunk_info = chunk_data['chunks']
        num_chunks = len(chunk_info)
        logger.info(f"Found {num_chunks} chunks to process")
        
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s)")
        
        # Check if parallel processing is enabled (via argument; default enabled)
        use_parallel = kwargs.get('use_parallel', True)
        max_workers = kwargs.get('max_workers', min(num_gpus, num_chunks)) if use_parallel else 1
        
        if use_parallel and num_gpus > 1:
            logger.info(f"Parallel processing enabled with {max_workers} GPUs")
        elif use_parallel and num_gpus == 1:
            logger.info("Only 1 GPU detected; using sequential processing (parallel mode requires multiple GPUs)")
            use_parallel = False
        else:
            logger.info("Using sequential processing mode")
        
        # Pre-load style_inv_noise (shared across all chunks)
        style_inv_noise = (
            load_ddim_latents_at_t(time_steps, ddim_latents_path=style_inv_path)
            .to(weight_dtype)
            .cuda()
        )
        
        if use_parallel and num_gpus > 1:
            # Parallel processing mode (multi-GPU)
            all_samples_dict = {}
            overlap = 2
            
            # Use ThreadPoolExecutor for parallel GPU tasks (not affected by GIL)
            logger.info(f"Starting parallel processing with {max_workers} GPU(s)...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {}
                for chunk_idx, chunk in enumerate(chunk_info):
                    # Assign GPU in round-robin fashion
                    device_id = chunk_idx % num_gpus
                    future = executor.submit(
                        process_chunk,
                        chunk_idx,
                        chunk,
                        content_inv_path,
                        style_inv_path,
                        style_inv_noise,
                        pretrained_model_path,
                        mask_path,
                        time_steps,
                        weight_dtype,
                        device_id,
                        overlap,
                        # Plan A and Plan B parameters
                        use_multi_scale_flow,
                        flow_scales,
                        flow_fusion_method,
                        use_temporal_attention,
                        temporal_attention_channels,
                        temporal_attention_heads,
                        temporal_attention_dropout,
                        temporal_attention_steps,
                    )
                    futures[future] = chunk_idx
                
                # Collect results
                completed = 0
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        _, chunk_sample = future.result()
                        all_samples_dict[chunk_idx] = chunk_sample
                        completed += 1
                        logger.info(f"Progress: {completed}/{num_chunks} chunks completed")
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
                        import traceback
                        traceback.print_exc()
                        raise e
            
            # Merge chunks in order
            all_samples = []
            for chunk_idx in range(num_chunks):
                chunk_sample = all_samples_dict[chunk_idx]
                if chunk_idx == 0:
                    all_samples.append(chunk_sample)
                else:
                    # Skip first `overlap` frames
                    if chunk_sample.shape[2] > overlap:
                        all_samples.append(chunk_sample[:, :, overlap:, :, :])
                    else:
                        all_samples.append(chunk_sample)
        else:
            # Sequential processing mode (original logic, single GPU or parallel disabled)
            all_samples = []
            overlap = 2
            for chunk_idx, chunk in enumerate(chunk_info):
                logger.info(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}: frames {chunk['start_frame']}-{chunk['end_frame']-1}")
                
                # Load inversion results for this chunk
                chunk_inv_path = os.path.join(content_inv_path, f"chunk_{chunk_idx:03d}")
                content_inv_noise = (
                    load_ddim_latents_at_t(time_steps, ddim_latents_path=chunk_inv_path)
                    .to(weight_dtype)
                    .cuda()
                )
                
                # Adjust style frame count to match content
                content_frames = content_inv_noise.shape[2]
                style_frames = style_inv_noise.shape[2]
                if content_frames != style_frames:
                    if style_frames == 1:
                        style_inv_noise_local = style_inv_noise.repeat(1, 1, content_frames, 1, 1)
                    elif content_frames > style_frames:
                        repeat_times = content_frames // style_frames
                        remainder = content_frames % style_frames
                        style_inv_noise_repeated = style_inv_noise.repeat(1, 1, repeat_times, 1, 1)
                        if remainder > 0:
                            style_inv_noise_remainder = style_inv_noise[:, :, -remainder:, :, :]
                            style_inv_noise_local = torch.cat([style_inv_noise_repeated, style_inv_noise_remainder], dim=2)
                        else:
                            style_inv_noise_local = style_inv_noise_repeated
                    else:
                        style_inv_noise_local = style_inv_noise[:, :, :content_frames, :, :]
                else:
                    style_inv_noise_local = style_inv_noise
                
                # Init latent-shift
                inv_latents_at_t = latent_adain(content_inv_noise, style_inv_noise_local)
                
                # Perform style transfer on this chunk
                chunk_sample = pipe.video_style_transfer(
                    "",
                    latents=inv_latents_at_t,
                    num_inference_steps=time_steps,
                    content_inv_path=chunk_inv_path,
                    style_inv_path=style_inv_path,
                    mask_path=mask_path,
                    output_type="tensor",
                    # Plan A and Plan B parameters
                    use_multi_scale_flow=use_multi_scale_flow,
                    flow_scales=flow_scales,
                    flow_fusion_method=flow_fusion_method,
                    use_temporal_attention=use_temporal_attention,
                    temporal_attention_channels=temporal_attention_channels,
                    temporal_attention_heads=temporal_attention_heads,
                    temporal_attention_dropout=temporal_attention_dropout,
                    temporal_attention_steps=temporal_attention_steps,
                ).images
                
                if isinstance(chunk_sample, np.ndarray):
                    chunk_sample = torch.from_numpy(chunk_sample)
                chunk_sample = torch.clamp(chunk_sample, 0.0, 1.0)
                chunk_sample = chunk_sample.permute(0, 4, 1, 2, 3).contiguous()
                
                # For the first chunk, take all frames; otherwise skip overlapping frames
                if chunk_idx == 0:
                    all_samples.append(chunk_sample)
                else:
                    if chunk_sample.shape[2] > overlap:
                        all_samples.append(chunk_sample[:, :, overlap:, :, :])
                    else:
                        all_samples.append(chunk_sample)
                
                # Clean up memory
                del content_inv_noise, style_inv_noise_local, inv_latents_at_t
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        logger.info(f"\nMerging results from {len(all_samples)} chunks...")
        sample = torch.cat(all_samples, dim=2)  # Concatenate along frame dimension
        logger.info(f"Merging complete; total frames: {sample.shape[2]}")
        
        # Clean up style_inv_noise
        del style_inv_noise
        torch.cuda.empty_cache()
    else:
        # Single-chunk processing mode (original logic)
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
        
        # Check frame count compatibility and adjust style_inv_noise if needed
        # Latent shape: (b, c, f, h, w)
        content_frames = content_inv_noise.shape[2]
        style_frames = style_inv_noise.shape[2]
        
        if content_frames != style_frames:
            logger.warning(f"Warning: Content inversion has {content_frames} frames, style inversion has {style_frames} frames; adjusting...")
            if style_frames == 1:
                # If style has only 1 frame, repeat to match content
                style_inv_noise = style_inv_noise.repeat(1, 1, content_frames, 1, 1)
                logger.info(f"Extended style frame count to {content_frames}")
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
                logger.info(f"Extended style frame count to {content_frames}")
            else:
                # If style has more frames, truncate to match content
                style_inv_noise = style_inv_noise[:, :, :content_frames, :, :]
                logger.info(f"Truncated style frame count to {content_frames}")
        
        # Init latent-shift
        inv_latents_at_t = latent_adain(content_inv_noise, style_inv_noise)

        # Video style transfer
        sample = pipe.video_style_transfer(
            "",
            latents=inv_latents_at_t,
            num_inference_steps=time_steps,
            content_inv_path=content_inv_path,
            style_inv_path=style_inv_path,
            mask_path=mask_path,
            output_type="tensor",  # Ensure tensor output
            # Plan A and Plan B parameters
            use_multi_scale_flow=use_multi_scale_flow,
            flow_scales=flow_scales,
            flow_fusion_method=flow_fusion_method,
            use_temporal_attention=use_temporal_attention,
            temporal_attention_channels=temporal_attention_channels,
            temporal_attention_heads=temporal_attention_heads,
            temporal_attention_dropout=temporal_attention_dropout,
            temporal_attention_steps=temporal_attention_steps,
        ).images
        # Convert numpy array to tensor if necessary
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        # Clamp values to [0, 1]
        sample = torch.clamp(sample, 0.0, 1.0)
        sample = sample.permute(0, 4, 1, 2, 3).contiguous()

    # Save results
    output_path = os.path.join(
        output_path,
        "sd",
        f'{content_inv_path.split("/")[-2]}_{style_inv_path.split("/")[-2]}',
    )
    os.makedirs(output_path, exist_ok=True)
    
    # Save individual frames (optional, for debugging)
    frames_dir = os.path.join(output_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    save_folder(sample, frames_dir)
    
    # Attempt to read original video FPS
    fps_file = os.path.join("results/contents-inv/sd", content_name, "frames", "video_fps.txt")
    fps_file_sampled = os.path.join("results/contents-inv/sd", content_name, "frames", "sampled", "video_fps.txt")
    video_fps = 30.0  # Default FPS
    
    # Prefer original FPS file (especially important for chunked processing)
    if os.path.exists(fps_file):
        try:
            with open(fps_file, "r") as f:
                video_fps = float(f.read().strip())
            logger.info(f"Using original video FPS: {video_fps:.2f}")
        except Exception as e:
            logger.error(f"Failed to read FPS file ({fps_file}): {e}; trying sampled FPS file")
            if os.path.exists(fps_file_sampled):
                try:
                    with open(fps_file_sampled, "r") as f:
                        video_fps = float(f.read().strip())
                    logger.info(f"Using sampled video FPS: {video_fps:.2f}")
                except Exception as e2:
                    logger.error(f"Failed to read sampled FPS file ({fps_file_sampled}): {e2}; using default FPS: {video_fps}")
            else:
                logger.warning(f"Sampled FPS file not found; using default FPS: {video_fps}")
    elif os.path.exists(fps_file_sampled):
        try:
            with open(fps_file_sampled, "r") as f:
                video_fps = float(f.read().strip())
            logger.warning(f"Using sampled video FPS: {video_fps:.2f}")
        except Exception as e:
            logger.error(f"Failed to read sampled FPS file ({fps_file_sampled}): {e}; using default FPS: {video_fps}")
    else:
        logger.warning(f"No FPS file found (tried {fps_file} and {fps_file_sampled}); using default FPS: {video_fps}")
    
    # Log video duration info
    actual_frames = sample.shape[2]
    calculated_duration = actual_frames / video_fps
    logger.info(f"Generated video info: {actual_frames} frames, fps={video_fps:.2f}, estimated duration={calculated_duration:.2f}s")
    
    # Save final video
    output_video_path = os.path.join(output_path, "output_video_hjh.mp4")
    logger.info(f"Saving video to: {output_video_path}")
    save_videos_grid(sample, output_video_path, fps=video_fps)
    logger.info("Video saved successfully")


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
        default="results/contents-inv/sd/mallard-fly/inversion",
    )
    parser.add_argument(
        "--style_inv_path", type=str, default="results/styles-inv/sd/0/inversion"
    )
    parser.add_argument(
        "--mask_path", type=str, default=None, required=False,
        help="Optional mask path. If not provided, mask will not be used."
    )
    # parser.add_argument("--mask_path", type=str, default="results/masks/sd/mallard-fly")
    parser.add_argument("--output_path", type=str, default="results/stylizations")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--use_parallel", action="store_true", default=True, help="Enable parallel processing (requires multiple GPUs); enabled by default")
    parser.add_argument("--no_parallel", action="store_false", dest="use_parallel", help="Disable parallel processing")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers (default: use all available GPUs)")
    #
    # Plan A: Multi-scale Optical Flow Fusion parameters
    parser.add_argument("--use_multi_scale_flow", action="store_true", default=False, help="Enable Plan A: Multi-scale optical flow fusion")
    parser.add_argument("--flow_scales", type=float, nargs='+', default=[1.0, 0.5, 0.25], help="List of scales for multi-scale optical flow")
    parser.add_argument("--flow_fusion_method", type=str, default='weighted_average', choices=['weighted_average', 'max_confidence'], help="Optical flow fusion method")
    #
    # Plan B: Temporal Attention Enhancement parameters
    parser.add_argument("--use_temporal_attention", action="store_true", default=False, help="Enable Plan B: Temporal Attention enhancement")
    parser.add_argument("--temporal_attention_channels", type=int, default=320, help="Number of channels for Temporal Attention")
    parser.add_argument("--temporal_attention_heads", type=int, default=8, help="Number of attention heads for Temporal Attention")
    parser.add_argument("--temporal_attention_dropout", type=float, default=0.0, help="Dropout rate for Temporal Attention")
    parser.add_argument("--temporal_attention_steps", type=int, nargs=2, default=[20, 30], help="Range of denoising steps [start, end] to apply Temporal Attention")
    args = parser.parse_args()
    args_dict = vars(args)
    # Convert temporal_attention_steps to tuple
    if isinstance(args_dict.get('temporal_attention_steps'), list):
        args_dict['temporal_attention_steps'] = tuple(args_dict['temporal_attention_steps'])
    main(**args_dict)