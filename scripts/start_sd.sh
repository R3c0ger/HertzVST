export PYTHONPATH=$(pwd)
# Set Hugging Face mirror to solve network connection issues
export HF_ENDPOINT=https://hf-mirror.com

# step1: Perform inversion for content video.
CUDA_VISIBLE_DEVICES=0 python src/sd/run_content_inversion_sd.py \
                        --content_path examples/contents/bird.mp4 \
                        --output_path results/contents-inv \
                        --is_opt \
                        --max_frames 30
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# step2: Perform inversion for style image.
CUDA_VISIBLE_DEVICES=0 python src/sd/run_style_inversion_sd.py \
                        --style_path examples/styles/2.png \
                        --output_path results/styles-inv \
                        --content_name 01
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# step3: Perform mask propagation. [Optional; you can also customize the masks and skip this step.]
# CUDA_VISIBLE_DEVICES=0 python src/mask_propagation.py \
#                        --feature_path results/contents-inv/sd/bird/features/inversion_feature_map_2_block_301_step.pt \
#                        --backbone 'sd' \
#                        --mask_path 'examples/masks/bird.png' \
#                        --output_path 'results/masks'

# step4: Perform localized video style transfer with Plan A and/or Plan B
# You can enable Plan A (Multi-scale Optical Flow Fusion) and/or Plan B (Temporal Attention Enhancement) as needed.

# Example 1: Enable only Plan A (Multi-scale Optical Flow Fusion)
# CUDA_VISIBLE_DEVICES=0 python src/sd/run_video_style_transfer_sd.py \
#                         --content_inv_path results/contents-inv/sd/bird/inversion \
#                         --style_inv_path results/styles-inv/sd/0/inversion \
#                         --output_path results/stylizations \
#                         --use_multi_scale_flow \
#                         --flow_scales 1.0 0.5 0.25 \
#                         --flow_fusion_method weighted_average

# Example 2: Enable only Plan B (Temporal Attention Enhancement)
# CUDA_VISIBLE_DEVICES=0 python src/sd/run_video_style_transfer_sd.py \
#                         --content_inv_path results/contents-inv/sd/bird/inversion \
#                         --style_inv_path results/styles-inv/sd/1/inversion \
#                         --output_path results/stylizations \
#                         --use_temporal_attention \
#                         --temporal_attention_channels 320 \
#                         --temporal_attention_heads 8 \
#                         --temporal_attention_steps 20 30

# Example 3: Enable both Plan A and Plan B
# CUDA_VISIBLE_DEVICES=0 python src/sd/run_video_style_transfer_sd.py \
#                         --content_inv_path results/contents-inv/sd/bird/inversion \
#                         --style_inv_path results/styles-inv/sd/1/inversion \
#                         --output_path results/stylizations \
#                         --use_multi_scale_flow \
#                         --flow_scales 1.0 0.5 0.25 \
#                         --flow_fusion_method weighted_average \
#                         --use_temporal_attention \
#                         --temporal_attention_channels 320 \
#                         --temporal_attention_heads 8 \
#                         --temporal_attention_steps 20 30

# Example 4: Disable both (use original method)
CUDA_VISIBLE_DEVICES=0 python src/sd/run_video_style_transfer_sd.py \
                        --content_inv_path results/contents-inv/sd/bird/inversion \
                        --style_inv_path results/styles-inv/sd/2/inversion \
                        --output_path results/stylizations

# --mask_path results/masks/sd/bird