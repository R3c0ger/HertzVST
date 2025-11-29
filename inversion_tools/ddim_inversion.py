import json
import os
from typing import Union

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torchvision import transforms
from tqdm import tqdm

from src.util import save_videos_grid, load_video_frames, load_ddim_latents_at_t
from utils import logger

# The decord package should be imported after torch
import decord
decord.bridge.set_bridge("torch")


def merge_inversion_chunks(
    inversion_root: str,
    meta_info_path: str,
    time_steps: int,
) -> None:
    """
    Merge inversion latents from multiple overlapping chunks 
    into a single coherent latent sequence for each DDIM timestep.

    Args:
        inversion_root (str): Root directory containing chunk_*/ subdirectories with inversion latents.
        meta_info_path (str): Path to meta_info.json containing chunk metadata.
        time_steps (int): Total number of DDIM timesteps used during inversion.
    """
    # Load chunk metadata
    with open(meta_info_path, 'r') as f:
        meta = json.load(f)
    total_frames = meta["total_frames"]
    chunks = meta["chunks"]  # List of {"start": int, "end": int, "num_frames": int}
    overlap = meta.get("overlap_frames", 0)

    logger.info(f"Merging inversion latents from {len(chunks)} chunks...")

    for t in tqdm(range(1, time_steps + 1), desc="Merging timesteps"):
        merged_latents = []
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_inversion_path = os.path.join(inversion_root, f"chunk_{chunk_idx:03d}")
            chunk_latents = load_ddim_latents_at_t(t, chunk_inversion_path)
            
            # If it's the first chunk, take all frames
            if chunk_idx == 0:
                merged_latents.append(chunk_latents)
            else:
                # For subsequent chunks, skip overlapping frames
                if chunk_latents.shape[2] > overlap:
                    merged_latents.append(chunk_latents[:, :, overlap:, :, :])
                else:
                    merged_latents.append(chunk_latents)
        
        # Concatenate all latents along the frame dimension
        if len(merged_latents) > 1:
            merged_latent = torch.cat(merged_latents, dim=2)
        else:
            merged_latent = merged_latents[0]
        
        # Save the merged latents for this timestep
        merged_latent_path = os.path.join(inversion_root, f"ddim_latents_{t}.pt")
        torch.save(merged_latent, merged_latent_path)

    logger.info(f"Merged inversion latents saved to: {inversion_root}")


def content_inversion_reconstruction(
    pipe,
    ddim_inv_scheduler,
    content_path: str,
    inversion_path: str,
    reconstruction_path: str,
    height: int,
    width: int,
    time_steps: int,
    weight_dtype,
    ft_indices=None,
    ft_timesteps=None,
    ft_path=None,
    is_opt=True,
    max_frames_per_chunk: int = 30,  # for long video inversion
    overlap_frames: int = 2,  # for smooth transition
):
    # Step 1: load video frames and fps
    if content_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
        vr = decord.VideoReader(content_path, width=width, height=height)
        total_frames = len(vr)
        fps = float(vr.get_avg_fps())
        frame_loader = lambda idx: vr.get_batch(idx)
    else:
        frame_loader, total_frames, fps \
            = load_video_frames(content_path, image_size=(width, height))
    logger.info(f"Total frames in video: {total_frames}, FPS: {fps}")

    # Step 2: chunk processing for long videos
    if total_frames <= max_frames_per_chunk:
        chunks = [(0, total_frames)]
    else:
        step = max_frames_per_chunk - overlap_frames
        chunks = []
        start = 0
        while start < total_frames:
            end = min(start + max_frames_per_chunk, total_frames)
            chunks.append((start, end))
            if end == total_frames:
                break
            start += step
    logger.info(f"Split the video in {len(chunks)} chunks: \n{chunks}")

    # Save meta info
    meta_info = {
        "total_frames": total_frames,
        "fps": fps,
        "max_frames_per_chunk": max_frames_per_chunk,
        "overlap_frames": overlap_frames,
        "chunks": [{"start": s, "end": e, "num_frames": e - s} for s, e in chunks]
    }
    with open(os.path.join(reconstruction_path, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    # Step 3: process each chunk
    logger.info(f"Start processing chunks...")
    for idx, (start, end) in enumerate(chunks):
        logger.info(f"Processing chunk {idx + 1}/{len(chunks)}: frames {start}-{end-1}")

        # create dirs for each chunk
        chunk_inv_path = os.path.join(inversion_path, f"chunk_{idx:03d}")
        chunk_rec_path = os.path.join(reconstruction_path, f"chunk_{idx:03d}")
        chunk_ft_path = os.path.join(ft_path, f"chunk_{idx:03d}") if ft_path else None
        os.makedirs(chunk_inv_path, exist_ok=True)
        os.makedirs(chunk_rec_path, exist_ok=True)
        if chunk_ft_path:
            os.makedirs(chunk_ft_path, exist_ok=True)

        # Load frames for the current chunk
        indices = list(range(start, end))
        raw_frames = frame_loader(indices)  # (N, H, W, C) for decord; (N, C, H, W) for folder

        # Unify to (N, H, W, C)
        if not content_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
            raw_frames = raw_frames.permute(0, 2, 3, 1)  # (N, C, H, W) → (N, H, W, C)

        pixel_values = (raw_frames / 127.5 - 1.0).unsqueeze(0)  # (1, N, H, W, C)
        pixel_values = rearrange(pixel_values, "b f h w c -> (b f) c h w").to(weight_dtype).cuda()

        # VAE encode
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=len(indices))
            latents = latents * pipe.vae.config.scaling_factor

            # Inversion
            logger.info(f"Starting inversion for chunk {idx + 1}/{len(chunks)}")
            ddim_inv_latent = ddim_inversion(
                pipe,
                ddim_inv_scheduler,
                video_latent=latents,
                num_inv_steps=time_steps,
                prompt="",
                inversion_path=chunk_inv_path,
                ft_indices=ft_indices,
                ft_timesteps=ft_timesteps,
                ft_path=chunk_ft_path,
                is_opt=is_opt,
            )[-1].to(weight_dtype)

            # Reconstruction
            logger.info(f"Starting reconstruction for chunk {idx + 1}/{len(chunks)}")
            sample = pipe.reconstruction(
                "", latents=ddim_inv_latent, video_length=len(indices), guidance_scale=1.0
            ).images
            sample = sample.permute(0, 4, 1, 2, 3).contiguous()
            save_videos_grid(sample, os.path.join(chunk_rec_path, "recon.mp4"))

        # Free up GPU memory
        torch.cuda.empty_cache()
    
    logger.info(f"All chunks processed. Start merging chunks...")
    merge_inversion_chunks(
        inversion_root=inversion_path,
        meta_info_path=os.path.join(reconstruction_path, "meta_info.json"),
        time_steps=time_steps,
    )

    return total_frames, fps, chunks


def merge_style_inversion_chunks(
    inversion_root: str,
    chunks: tuple,
    time_steps: int,
) -> None:
    logger.info(f"Merging {len(chunks)} style chunks...")

    for t in tqdm(range(1, time_steps + 1), desc="Merging style timesteps"):
        merged_latents = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_inversion_path = os.path.join(inversion_root, f"chunk_{chunk_idx:03d}")
            chunk_latents = load_ddim_latents_at_t(t, chunk_inversion_path)
            
            # If it's the first chunk, take all frames
            if chunk_idx == 0:
                merged_latents.append(chunk_latents)
            else:
                # For subsequent chunks, skip overlapping frames
                if chunk_latents.shape[2] > 2:
                    merged_latents.append(chunk_latents[:, :, 2:, :, :])
                else:
                    merged_latents.append(chunk_latents)
        
        # Concatenate all latents along the frame dimension
        if len(merged_latents) > 1:
            merged_latent = torch.cat(merged_latents, dim=2)
        else:
            merged_latent = merged_latents[0]

        # Save
        output_path = os.path.join(inversion_root, f"ddim_latents_{t}.pt")
        torch.save(merged_latent, output_path)

    logger.info(f"Style inversion merged and saved to: {inversion_root}")


def style_inversion_reconstruction(
    pipe,
    ddim_inv_scheduler,
    style_path,
    inversion_path,
    reconstruction_path,
    height,
    width,
    time_steps,
    weight_dtype,
    is_opt=True,
    chunks=None,
):
    # Load and preprocess style image
    style_image = Image.open(style_path).convert("RGB").resize((width, height))
    style_tensor = transforms.ToTensor()(style_image)  # (C, H, W)
    style_tensor = 2.0 * style_tensor - 1.0  # [-1, 1]

    # Process each chunk
    for idx, (start, end) in enumerate(chunks):
        chunk_len = end - start
        logger.info(f"Processing style chunk {idx + 1}/{len(chunks)}: frames {start}-{end-1}")

        # Repeat the style tensor to chunk length
        pixel_values = style_tensor.unsqueeze(0).repeat(chunk_len, 1, 1, 1)  # (F, C, H, W)
        pixel_values = pixel_values.to(weight_dtype).cuda()

        # VAE encode
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=chunk_len)
            latents = latents * pipe.vae.config.scaling_factor

            # Inversion
            chunk_inv_path = os.path.join(inversion_path, f"chunk_{idx:03d}")
            os.makedirs(chunk_inv_path, exist_ok=True)

            ddim_inv_latent = ddim_inversion(
                pipe,
                ddim_inv_scheduler,
                video_latent=latents,
                num_inv_steps=time_steps,
                prompt="",
                inversion_path=chunk_inv_path,
                is_opt=is_opt,
            )[-1].to(weight_dtype)

            # Reconstruction (for this chunk only)
            chunk_rec_path = os.path.join(reconstruction_path, f"chunk_{idx:03d}")
            os.makedirs(chunk_rec_path, exist_ok=True)
            sample = pipe.reconstruction(
                "", latents=ddim_inv_latent, video_length=chunk_len, guidance_scale=1.0
            ).images
            sample = sample.permute(0, 4, 1, 2, 3).contiguous()
            save_videos_grid(sample, os.path.join(chunk_rec_path, "recon.mp4"))

        torch.cuda.empty_cache()

    # Merge style chunks (simple concatenation, no blending)
    logger.info("Merging style inversion chunks...")
    merge_style_inversion_chunks(
        inversion_root=inversion_path,
        chunks=chunks,
        time_steps=time_steps,
    )


# *********************************************************************************************


@torch.no_grad()
def ddim_inversion(
    pipeline,
    ddim_scheduler,
    video_latent,
    num_inv_steps,
    prompt="",
    inversion_path=None,
    ft_indices=None,
    ft_timesteps=None,
    ft_path=None,
    is_opt=False,
):
    if is_opt:
        ddim_latents = ddim_loop_plus(
            pipeline,
            ddim_scheduler,
            video_latent,
            num_inv_steps,
            prompt,
            inversion_path,
            ft_indices=ft_indices,
            ft_timesteps=ft_timesteps,
            ft_path=ft_path,
        )
    else:
        ddim_latents = ddim_loop(
            pipeline,
            ddim_scheduler,
            video_latent,
            num_inv_steps,
            prompt,
            inversion_path,
            ft_indices=ft_indices,
            ft_timesteps=ft_timesteps,
            ft_path=ft_path,
        )
    return ddim_latents


@torch.no_grad()
def ddim_loop(
    pipeline,
    ddim_scheduler,
    latent,
    num_inv_steps,
    prompt,
    inversion_path,
    ft_indices=None,
    ft_timesteps=None,
    ft_path=None,
):
    context = init_prompt(pipeline, prompt)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    if inversion_path is not None:
        torch.save(
            latent.detach().clone(),
            os.path.join(inversion_path, f"ddim_latents_{0}.pt"),
        )
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(
            pipeline,
            latent,
            t,
            cond_embeddings,
            ft_indices=ft_indices,
            ft_timesteps=ft_timesteps,
            ft_path=ft_path,
        )
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        # save latent
        if inversion_path is not None:
            torch.save(
                latent.detach().clone(),
                os.path.join(inversion_path, f"ddim_latents_{i + 1}.pt"),
            )
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_loop_plus(
    pipeline,
    ddim_scheduler,
    latent,
    num_inv_steps,
    prompt,
    inversion_path,
    ft_indices=None,
    ft_timesteps=None,
    ft_path=None,
):
    context = init_prompt(pipeline, prompt)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    if inversion_path is not None:
        torch.save(
            latent.detach().clone(),
            os.path.join(inversion_path, f"ddim_latents_{0}.pt"),
        )
    or_latent_idx = 0.5
    inject_steps = 0.05
    inject_len = 0.2
    num_inference_steps = 50
    num_fix_itr = 0
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(
            pipeline,
            latent,
            t,
            cond_embeddings,
            ft_indices=ft_indices,
            ft_timesteps=ft_timesteps,
            ft_path=ft_path,
        )
        noise_pred = noise_pred.requires_grad_(True)

        last_noise = noise_pred
        if (
            (inject_steps + inject_len) * num_inference_steps
            > i
            > inject_steps * num_inference_steps
        ):
            # print("add!")
            if i > 0:
                latent = or_latent_idx * latent + (1 - or_latent_idx) * last_latent
        for fix_itr in range(num_fix_itr):
            if fix_itr == 0:
                print("fix!")
            if fix_itr > 0:
                latents_tmp = next_step(
                    (noise_pred + last_noise) / 2, t, latent, ddim_scheduler
                )
            else:
                latents_tmp = next_step(noise_pred, t, latent, ddim_scheduler)
            last_noise = noise_pred
            noise_pred = get_noise_pred_single(
                latents_tmp, t, cond_embeddings, pipeline.unet
            )

        last_latent = latent
        latent = next_step(noise_pred, t, latent, ddim_scheduler)

        # save latent
        if inversion_path is not None:
            torch.save(
                latent.detach().clone(),
                os.path.join(inversion_path, f"ddim_latents_{i + 1}.pt"),
            )
        all_latent.append(latent)
        continue
    return all_latent


@torch.no_grad()
def init_prompt(pipeline, prompt):
    uncond_input = pipeline.tokenizer(
        [""],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipeline.text_encoder(
        uncond_input.input_ids.to(pipeline.device)
    )[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: int,
    sample: Union[torch.FloatTensor, np.ndarray],
    ddim_scheduler,
):
    timestep, next_timestep = (
        min(
            timestep
            - ddim_scheduler.config.num_train_timesteps
            // ddim_scheduler.num_inference_steps,
            999,
        ),
        timestep,
    )
    alpha_prod_t = (
        ddim_scheduler.alphas_cumprod[timestep]
        if timestep >= 0
        else ddim_scheduler.final_alpha_cumprod
    )
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t

    next_original_sample = (
        sample - beta_prod_t**0.5 * model_output
    ) / alpha_prod_t**0.5
    pred_epsilon = model_output

    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * pred_epsilon
    next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction

    return next_sample


def get_noise_pred_single(
    pipeline, latents, t, context, ft_indices=None, ft_timesteps=None, ft_path=None
):
    noise_pred = pipeline.unet(
        latents,
        t,
        encoder_hidden_states=context,
        ft_indices=ft_indices,
        ft_timesteps=ft_timesteps,
        ft_path=ft_path,
    )["sample"]
    return noise_pred
