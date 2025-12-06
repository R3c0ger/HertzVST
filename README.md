<div style="text-align: center;">
<h1>
HertzVST - Diffusion-based Training-Free Framework for Video Style Transfer
</h1>
</div>

## Installation

Clone the repository:
```
git clone https://github.com/R3c0ger/HertzVST.git
cd HertzVST
```

Installation with the `requirement.txt`:
```
conda create -n HertzVST python=3.10
conda activate HertzVST
pip install -r requirements.txt
```

## Project Structure

**Warning:** The `stable-diffusion-v1-5/` directory contains the Stable Diffusion model files, which are required for the framework. When running scripts for the first time, it will download the model here. **The size of this directory is over 40GB, and you should ensure you have sufficient disk space.**


```
├───backbones/
│   └───video_diffusion_sd/
│       ├───models/
│       │   ├───attention.py
│       │   ├───lora.py
│       │   ├───resnet.py
│       │   ├───unet_3d_blocks.py
│       │   └───unet_3d_condition.py
│       ├───pipelines/
│       │   └───stable_diffusion.py
│       └───pnp_utils.py
├───datasets/
├───eval/
│   ├───video/
│   │   ├───010原始.mp4
│   │   ├───bird0原始.mp4
│   │   ├───时序A010.mp4
│   │   └───时序Abird0.mp4
│   ├───README_eval.md
│   ├───eval_multiscale_optical_flow.sh
│   └───eval_temporal_consistency.py
├───inversion_tools/
│   ├───ddim_inversion.py
│   └───flow_inversion.py
├───scripts/
│   └───start_sd.sh
├───src/
│   ├───sd/
│   │   ├───run_content_inversion_sd.py
│   │   ├───run_style_inversion_sd.py
│   │   └───run_video_style_transfer_sd.py
│   ├───calc_optical_flow.py
│   ├───mask_propagation.py
│   ├───palette.txt
│   └───util.py
├───utils/
│   ├───__init__.py
│   ├───logger.py
│   └───paths.py
├───stable-diffusion-v1-5/
│   ├───feature_extractor/
│   │   └───preprocessor_config.json
│   ├───safety_checker/
│   ├───scheduler/
│   │   └───scheduler_config.json
│   ├───text_encoder/
│   ├───tokenizer/
│   ├───unet/
│   ├───vae/
│   ├───.gitattributes
│   ├───README.md
│   ├───model_index.json
│   ├───v1-5-pruned-emaonly.ckpt
│   ├───v1-5-pruned-emaonly.safetensors
│   ├───v1-5-pruned-emaonly.safetensors.filepart
│   ├───v1-5-pruned.ckpt
│   ├───v1-5-pruned.safetensors.filepart
│   └───v1-inference.yaml
├───.gitignore
├───LICENSE
├───README.md
├───install_env.sh
└───requirements.txt

```

## Usage

Run `scripts/start_sd.sh` to start the style transfer process. Modify the parameters in the script as needed.

The whole process is to run the following three scripts (procedures) sequentially: 
```
└───src/
    └───sd/
        ├───run_content_inversion_sd.py
        ├───run_style_inversion_sd.py
        └───run_video_style_transfer_sd.py
```

### 1. Content Video Inversion (`run_content_inversion_sd.py`)

Performs DDIM inversion on the content video to obtain latent representations.

**Required Arguments:**
- `--content_path`: Path to the input video file (e.g., `examples/contents/bird.mp4`)
- `--output_path`: Directory to save inversion results (e.g., `results/contents-inv`)

**Optional Arguments:**
- `--pretrained_model_path`: Path to Stable Diffusion model (default: `"stable-diffusion-v1-5"`)
- `--weight_dtype`: Data type for model weights (default: `torch.float16`)
- `--height`: Frame height (default: `512`)
- `--width`: Frame width (default: `512`)
- `--time_steps`: Number of DDIM inversion steps (default: `50`)
- `--max_frames`: Maximum frames per chunk to avoid OOM (default: `30`)
- `--overlap_frames`: Overlapping frames between chunks (default: `2`)
- `--ft_indices`: Feature indices for inversion (default: `2`)
- `--ft_timesteps`: Timesteps for feature extraction (default: `301`)
- `--is_opt`: Use Easy-Inv optimization (flag, no argument)
- `--seed`: Random seed (default: `33`)

**Example:**
```bash
CUDA_VISIBLE_DEVICES=0 python src/sd/run_content_inversion_sd.py \
    --content_path examples/contents/bird.mp4 \
    --output_path results/contents-inv \
    --is_opt \
    --max_frames 30
```

### 2. Style Image Inversion (`run_style_inversion_sd.py`)

Performs DDIM inversion on the style image to obtain latent representations.

**Required Arguments:**
- `--style_path`: Path to the style image (e.g., `examples/styles/2.png`)
- `--output_path`: Directory to save inversion results (e.g., `results/styles-inv`)

**Optional Arguments:**
- `--pretrained_model_path`: Path to Stable Diffusion model (default: `"stable-diffusion-v1-5"`)
- `--weight_dtype`: Data type for model weights (default: `torch.float16`)
- `--num_frames`: Number of frames (default: `16`)
- `--height`: Image height (default: `512`)
- `--width`: Image width (default: `512`)
- `--time_steps`: Number of DDIM inversion steps (default: `50`)
- `--is_opt`: Use Easy-Inv optimization (flag, no argument)
- `--seed`: Random seed (default: `33`)
- `--content_name`: Content name for chunk info lookup (default: `"01"`)

**Example:**
```bash
CUDA_VISIBLE_DEVICES=0 python src/sd/run_style_inversion_sd.py \
    --style_path examples/styles/2.png \
    --output_path results/styles-inv \
    --content_name 01
```

### 3. Video Style Transfer (`run_video_style_transfer_sd.py`)

Performs video style transfer using pre-computed content and style inversions.

**Required Arguments:**
- `--content_inv_path`: Path to content inversion results (e.g., `results/contents-inv/sd/bird/inversion`)
- `--style_inv_path`: Path to style inversion results (e.g., `results/styles-inv/sd/2/inversion`)
- `--output_path`: Directory to save stylized videos (e.g., `results/stylizations`)

**Optional Arguments:**
- `--pretrained_model_path`: Path to Stable Diffusion model (default: `"stable-diffusion-v1-5"`)
- `--weight_dtype`: Data type for model weights (default: `torch.float16`)
- `--time_steps`: Number of inference steps (default: `50`)
- `--seed`: Random seed (default: `33`)
- `--mask_path`: Optional mask for localized style transfer
- `--use_parallel`: Enable parallel processing (requires multiple GPUs, default: `True`)
- `--max_workers`: Maximum parallel workers (default: all available GPUs)

**Plan A - Multi-scale Optical Flow Fusion:**
- `--use_multi_scale_flow`: Enable multi-scale optical flow fusion (flag)
- `--flow_scales`: List of scales for flow computation (default: `[1.0, 0.5, 0.25]`)
- `--flow_fusion_method`: Flow fusion method: `weighted_average` or `max_confidence` (default: `weighted_average`)

**Plan B - Temporal Attention Enhancement:**
- `--use_temporal_attention`: Enable temporal attention enhancement (flag)
- `--temporal_attention_channels`: Number of channels (default: `320`)
- `--temporal_attention_heads`: Number of attention heads (default: `8`)
- `--temporal_attention_dropout`: Dropout rate (default: `0.0`)
- `--temporal_attention_steps`: Denoising steps range to apply attention (default: `[20, 30]`)

**Examples:**

1. **Standard transfer (no enhancements):**
```bash
CUDA_VISIBLE_DEVICES=0 python src/sd/run_video_style_transfer_sd.py \
    --content_inv_path results/contents-inv/sd/bird/inversion \
    --style_inv_path results/styles-inv/sd/2/inversion \
    --output_path results/stylizations
```

2. **With Plan A only (Multi-scale Optical Flow):**
```bash
CUDA_VISIBLE_DEVICES=0 python src/sd/run_video_style_transfer_sd.py \
    --content_inv_path results/contents-inv/sd/bird/inversion \
    --style_inv_path results/styles-inv/sd/0/inversion \
    --output_path results/stylizations \
    --use_multi_scale_flow \
    --flow_scales 1.0 0.5 0.25 \
    --flow_fusion_method weighted_average
```

3. **With Plan B only (Temporal Attention):**
```bash
CUDA_VISIBLE_DEVICES=0 python src/sd/run_video_style_transfer_sd.py \
    --content_inv_path results/contents-inv/sd/bird/inversion \
    --style_inv_path results/styles-inv/sd/1/inversion \
    --output_path results/stylizations \
    --use_temporal_attention \
    --temporal_attention_channels 320 \
    --temporal_attention_heads 8 \
    --temporal_attention_steps 20 30
```

4. **With both Plan A and Plan B:**
```bash
CUDA_VISIBLE_DEVICES=0 python src/sd/run_video_style_transfer_sd.py \
    --content_inv_path results/contents-inv/sd/bird/inversion \
    --style_inv_path results/styles-inv/sd/1/inversion \
    --output_path results/stylizations \
    --use_multi_scale_flow \
    --flow_scales 1.0 0.5 0.25 \
    --flow_fusion_method weighted_average \
    --use_temporal_attention \
    --temporal_attention_channels 320 \
    --temporal_attention_heads 8 \
    --temporal_attention_steps 20 30
```

### Notes
- Run scripts sequentially: content inversion → style inversion → style transfer
- Use `--mask_path` for localized style transfer with mask propagation
- Enable `--use_parallel` for multi-GPU parallel processing
- Clear GPU memory between steps using `torch.cuda.empty_cache()`
- Set `HF_ENDPOINT=https://hf-mirror.com` if experiencing network issues

## Acknowledgements

This project is based on [UniVST](https://github.com/QuanjianSong/UniVST) by QuanjianSong.

## License

This project is licensed under the Apache License 2.0.