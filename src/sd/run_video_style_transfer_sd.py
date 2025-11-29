import argparse
import json
import os
from typing import Optional, List

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


def merge_style_transfer_chunks(
    chunk_outputs: List[torch.Tensor],
    chunks: List[dict],
    overlap_frames: int
) -> torch.Tensor:
    """
    合并风格迁移的分块结果，处理重叠帧
    """
    logger.info(f"Merging {len(chunk_outputs)} style transfer chunks with {overlap_frames} overlap frames")
    
    merged_frames = []
    total_processed = 0
    
    for i, (chunk, output) in enumerate(zip(chunks, chunk_outputs)):
        chunk_frames = output.shape[2]  # 获取分块的帧数
        chunk_output = output.cpu()
        
        if i == 0:
            # 第一个分块：取所有帧
            merged_frames.append(chunk_output)
            total_processed = chunk_frames
        else:
            # 后续分块：跳过重叠帧
            if overlap_frames > 0 and chunk_frames > overlap_frames:
                # 使用重叠区域的加权平均来平滑过渡
                if i < len(chunk_outputs) - 1:
                    # 中间分块：取非重叠部分
                    frames_to_take = chunk_output[:, :, overlap_frames:, :, :]
                    merged_frames.append(frames_to_take)
                    total_processed += (chunk_frames - overlap_frames)
                else:
                    # 最后一个分块：取所有帧（包括重叠）
                    merged_frames.append(chunk_output)
                    total_processed += chunk_frames
            else:
                # 无重叠或帧数不足
                merged_frames.append(chunk_output)
                total_processed += chunk_frames
    
    # 沿帧维度拼接
    if len(merged_frames) > 1:
        merged_output = torch.cat(merged_frames, dim=2)
    else:
        merged_output = merged_frames[0]
    
    logger.info(f"Merged output shape: {merged_output.shape}")
    return merged_output


def load_mask_chunk(mask_path: str, chunk_info: dict, total_frames: int) -> Optional[torch.Tensor]:
    """
    加载对应分块的mask
    """
    if mask_path is None or not os.path.exists(mask_path):
        return None
    
    # 如果mask是单张图片，扩展到整个分块
    if mask_path.endswith(('.png', '.jpg', '.jpeg')):
        from PIL import Image
        import torchvision.transforms as transforms
        
        mask_image = Image.open(mask_path).convert('L')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        mask_tensor = transform(mask_image)
        # 扩展到分块的所有帧
        mask_chunk = mask_tensor.unsqueeze(0).unsqueeze(0).repeat(
            1, 1, chunk_info['num_frames'], 1, 1
        )
        return mask_chunk
    
    # 如果mask已经是分块格式
    chunk_mask_path = os.path.join(mask_path, f"chunk_{chunk_info['index']:03d}.pt")
    if os.path.exists(chunk_mask_path):
        return torch.load(chunk_mask_path)
    
    # 如果mask是完整的视频mask，需要分割
    full_mask_path = os.path.join(mask_path, "propagated_masks.pt")
    if os.path.exists(full_mask_path):
        full_mask = torch.load(full_mask_path)
        start, end = chunk_info['start'], chunk_info['end']
        return full_mask[:, :, start:end, :, :]
    
    return None


def main(
    pretrained_model_path: str,
    content_inv_path: str,
    style_inv_path: str,
    mask_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
    time_steps: int = 50,
    seed: Optional[int] = 33,
    overlap_frames: int = 2,
    fps: int = 8,
    **kwargs,
):
    if seed is not None:
        seed_everything(seed)
    
    # 加载模型
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

    # 读取内容分块元信息
    content_meta_path = os.path.join(content_inv_path, "../reconstruction/meta_info.json")
    if not os.path.exists(content_meta_path):
        logger.warning("No chunk meta info found, using single chunk processing")
        # 回退到单块处理
        return process_single_chunk(
            pipe, content_inv_path, style_inv_path, mask_path, 
            output_path, weight_dtype, time_steps
        )
    
    with open(content_meta_path, 'r') as f:
        content_meta = json.load(f)
    
    chunks = content_meta["chunks"]
    total_frames = content_meta["total_frames"]
    overlap_frames = content_meta.get("overlap_frames", overlap_frames)
    
    logger.info(f"Processing style transfer in {len(chunks)} chunks (total {total_frames} frames)")

    # 为每个分块添加索引信息
    for i, chunk in enumerate(chunks):
        chunk['index'] = i

    # 处理每个分块
    chunk_outputs = []
    for chunk in chunks:
        chunk_idx = chunk['index']
        start, end = chunk['start'], chunk['end']
        
        logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}: frames {start}-{end-1}")

        # 加载当前分块的内容latent
        chunk_content_path = os.path.join(content_inv_path, f"chunk_{chunk_idx:03d}")
        content_inv_noise = load_ddim_latents_at_t(time_steps, chunk_content_path)
        
        # 加载对应分块的风格latent
        chunk_style_path = os.path.join(style_inv_path, f"chunk_{chunk_idx:03d}")
        style_inv_noise = load_ddim_latents_at_t(time_steps, chunk_style_path)
        
        # 加载对应分块的mask
        chunk_mask = load_mask_chunk(mask_path, chunk, total_frames)
        chunk_mask_path = None
        if chunk_mask is not None:
            # 临时保存分块mask
            chunk_mask_dir = os.path.join(output_path, "temp_masks")
            os.makedirs(chunk_mask_dir, exist_ok=True)
            chunk_mask_path = os.path.join(chunk_mask_dir, f"chunk_{chunk_idx:03d}.pt")
            torch.save(chunk_mask, chunk_mask_path)
        
        # 风格迁移
        inv_latents_at_t = latent_adain(
            content_inv_noise.to(weight_dtype).cuda(),
            style_inv_noise.to(weight_dtype).cuda()
        )
        
        # 注册attention（每个分块都需要）
        register_spatial_attention_pnp(pipe)
        
        # 分块风格迁移
        sample = pipe.video_style_transfer(
            "",
            latents=inv_latents_at_t,
            num_inference_steps=time_steps,
            content_inv_path=chunk_content_path,
            style_inv_path=chunk_style_path,
            mask_path=chunk_mask_path,
        ).images
        
        sample = sample.permute(0, 4, 1, 2, 3).contiguous()
        chunk_outputs.append(sample)
        
        # 清理临时mask文件
        if chunk_mask_path and os.path.exists(chunk_mask_path):
            os.remove(chunk_mask_path)
        
        # 释放GPU内存
        torch.cuda.empty_cache()
        logger.info(f"Chunk {chunk_idx + 1} completed, output shape: {sample.shape}")
    
    # 合并所有分块的结果
    merged_output = merge_style_transfer_chunks(chunk_outputs, chunks, overlap_frames)
    
    # 保存最终结果
    final_output_path = os.path.join(
        output_path,
        "sd",
        f'{os.path.basename(os.path.dirname(content_inv_path))}_{os.path.basename(os.path.dirname(style_inv_path))}',
    )
    os.makedirs(final_output_path, exist_ok=True)
    
    # 保存每帧图片
    frames_dir = os.path.join(final_output_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    save_folder(merged_output, frames_dir)
    
    # 保存最终视频
    output_video_path = os.path.join(final_output_path, "output_video.mp4")
    logger.info(f"Saving final video to {output_video_path}")
    save_videos_grid(merged_output, output_video_path, fps=fps)
    
    logger.info("Style transfer completed successfully!")


def process_single_chunk(
    pipe,
    content_inv_path: str,
    style_inv_path: str,
    mask_path: str,
    output_path: str,
    weight_dtype: torch.dtype,
    time_steps: int,
):
    """单块处理（回退方案）"""
    logger.info("Using single chunk processing")
    
    # 加载完整的latent
    content_inv_noise = load_ddim_latents_at_t(time_steps, content_inv_path)
    style_inv_noise = load_ddim_latents_at_t(time_steps, style_inv_path)
    
    # 风格迁移
    inv_latents_at_t = latent_adain(
        content_inv_noise.to(weight_dtype).cuda(),
        style_inv_noise.to(weight_dtype).cuda()
    )
    
    # 注册attention
    register_spatial_attention_pnp(pipe)
    
    # 风格迁移
    sample = pipe.video_style_transfer(
        "",
        latents=inv_latents_at_t,
        num_inference_steps=time_steps,
        content_inv_path=content_inv_path,
        style_inv_path=style_inv_path,
        mask_path=mask_path,
    ).images
    
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    
    # 保存结果
    final_output_path = os.path.join(
        output_path,
        "sd",
        f'{os.path.basename(os.path.dirname(content_inv_path))}_{os.path.basename(os.path.dirname(style_inv_path))}',
    )
    os.makedirs(final_output_path, exist_ok=True)
    
    frames_dir = os.path.join(final_output_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    save_folder(sample, frames_dir)
    
    output_video_path = os.path.join(final_output_path, "output_video.mp4")
    save_videos_grid(sample, output_video_path)
    
    logger.info("Single chunk style transfer completed!")


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
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=get_exp_dir() + "results/stylizations")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--max_frames_per_chunk", type=int, default=30)
    parser.add_argument("--overlap_frames", type=int, default=2)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
