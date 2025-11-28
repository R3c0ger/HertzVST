import argparse
import os
import sys

import torch

from src.mask_propagation import video_mask_propogation
from src.sd.run_content_inversion_sd import main as content_inversion_main
from src.sd.run_style_inversion_sd import main as style_inversion_main
from src.sd.run_video_style_transfer_sd import main as video_style_transfer_main
from utils import logger, get_exp_dir


def print_env_info():
    logger.info(
        "\n------------------------------------------------------------"
        f"\nEnvironment Information:"
        f"\n- Python version: {sys.version}"
        f"\n- PyTorch version: {torch.__version__}"
        f"\n- CUDA available: {torch.cuda.is_available()}"
        f"\n- CUDA version: {torch.version.cuda}"
        f"\n- GPU device: {torch.cuda.get_device_name(0)}"
        f"\n- Experiment output directory: {get_exp_dir()}"
        "\n------------------------------------------------------------\n"
    )


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the demo pipeline "
        "(content/style inversion, optional mask propagation, and video style transfer)."
    )
    # Paths for different steps
    # General - Pretrained model
    parser.add_argument("--pretrained_model_path", type=str, default="stable-diffusion-v1-5", 
                        help="Pretrained Stable Diffusion model path")
    # Content Inversion
    parser.add_argument("--content_path", type=str, default="examples/contents/mallard-fly", 
                        help="Content frames folder")
    parser.add_argument("--content_out", type=str, 
                        default=get_exp_dir() / "results/contents-inv", 
                        help="Content inversion output folder")
    # Style Inversion
    parser.add_argument( "--style_path", type=str, default="examples/styles/3.png", 
                        help="Style image path")
    parser.add_argument("--style_out", type=str, 
                        default=get_exp_dir() / "results/styles-inv", 
                        help="Style inversion output folder")
    # Mask propagation
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Mask path to use for style transfer (optional). " \
                        "If not provided, transfer runs without mask.")
    parser.add_argument("--feature_path", type=str, default=None, 
                        help="Feature path (will be inferred from content inversion output if not provided)")
    parser.add_argument("--masks_out", type=str, 
                        default=get_exp_dir() / "results/masks", 
                        help="Mask propagation output folder")
    # Final stylization
    parser.add_argument("--stylizations_out", type=str, 
                        default=get_exp_dir() / "results/stylizations", 
                        help="Final stylizations output folder")
    
    # Other parameters
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    parser.add_argument("--num_frames", type=int, default=16, help="The total nums of mask.")
    parser.add_argument("--height", type=int, default=512, help="The height of the frames.")
    parser.add_argument("--width", type=int, default=512, help="The width of the frames.")
    parser.add_argument("--time_steps", type=int, default=50, help="The number of time steps.")
    parser.add_argument("--ft_indices", type=int, default=2, help="The feature indices.")
    parser.add_argument("--ft_timesteps", type=int, default=301, help="The feature timesteps.")
    parser.add_argument("--is_opt", action="store_true", default=True, help="Use Easy-Inv.")
    parser.add_argument("--seed", type=int, default=33, help="Random seed.")
    # For mask propagation
    parser.add_argument("--temperature", default=0.2, type=float, help="The temperature for softmax.")
    parser.add_argument("--n_last_frames", type=int, default=9, help="The numbers of anchor frames.")
    parser.add_argument("--topk", type=int, default=15, help="The hyper-parameters of KNN top k.")
    parser.add_argument("--sample_ratio",type=float,default=0.3,help="The sample ratio of mask propagation.")
    parser.add_argument("--backbone", type=str, default="sd", help="The backbone of feature extractor.")
    
    # Control which steps to run
    parser.add_argument("--run_content", action="store_true", default=True, help="Run content inversion")
    parser.add_argument("--run_style", action="store_true", default=True, help="Run style inversion")
    parser.add_argument("--run_mask", action="store_true", default=False, help="Run mask propagation (optional)")
    parser.add_argument("--run_transfer", action="store_true", default=True, help="Run video style transfer")
    parser.add_argument("--all", action="store_true", default=False, 
                        help="Run all steps in sequence (same as start_sd.sh)",)

    args = parser.parse_args()
    return args


def main():
    print_env_info()
    args = arg_parser()

    # Default behavior: if --all set, enable all steps
    run_content = args.all or args.run_content
    run_style = args.all or args.run_style
    run_mask = args.all or args.run_mask
    run_transfer = args.all or args.run_transfer

    # Run steps
    logger.info(f"Running with settings: \n"
                f"run_content={run_content}, run_style={run_style}, "
                f"run_mask={run_mask}, run_transfer={run_transfer}")
    logger.info(f"Path of pretrained model: {args.pretrained_model_path}")

    if run_content:
        logger.info(
            "Starting content inversion..."
            f"\n- Content path: {args.content_path}"
            f"\n- Content output path: {args.content_out}"
        )
        content_inversion_main(
            pretrained_model_path=args.pretrained_model_path,
            content_path=args.content_path,
            output_path=args.content_out,
            weight_dtype=args.weight_dtype,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            time_steps=args.time_steps,
            ft_indices=args.ft_indices,
            ft_timesteps=args.ft_timesteps,
            is_opt=args.is_opt,
            seed=args.seed,
        )

    if run_style:
        logger.info(
            "Starting style inversion..."
            f"\n- Style path: {args.style_path}"
            f"\n- Style output path: {args.style_out}"
        )
        style_inversion_main(
            pretrained_model_path=args.pretrained_model_path,
            style_path=args.style_path,
            output_path=args.style_out,
            weight_dtype=args.weight_dtype,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            time_steps=args.time_steps,
            is_opt=args.is_opt,
            seed=args.seed,
        )

    if run_mask:
        # try to infer default feature path produced by content inversion
        feature_default = os.path.join(
            args.content_out,
            args.backbone,
            os.path.basename(args.content_path),
            "features",
            "inversion_feature_map_2_block_301_step.pt",
        )
        logger.info(
            "Starting mask propagation..."
            f"\n- Feature path: {args.feature_path or feature_default}"
            f"\n- Mask path: {args.mask_path or 'No mask provided, running without mask'}"
            f"\n- Masks output path: {args.masks_out}"
        )
        video_mask_propogation(
            feature_path=args.feature_path or feature_default,
            mask_path=args.mask_path or "examples/masks/mallard-fly.png",
            output_path=args.masks_out,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            temperature=args.temperature,
            n_last_frames=args.n_last_frames,
            topk=args.topk,
            sample_ratio=args.sample_ratio,
            backbone=args.backbone,
        )

    if run_transfer:
        # construct content/style inversion paths matching scripts' defaults
        content_inv = os.path.join(
            args.content_out, 
            args.backbone, 
            os.path.basename(args.content_path), 
            "inversion"
        )
        style_inv = os.path.join(
            args.style_out,
            args.backbone,
            os.path.basename(args.style_path).split(".")[0],
            "inversion",
        )
        logger.info(
            "Starting video style transfer..."
            f"\n- Content inversion path: {content_inv}"
            f"\n- Style inversion path: {style_inv}"
            f"\n- Mask path: {args.mask_path or 'No mask provided, running without mask'}"
            f"\n- Stylizations output path: {args.stylizations_out}"
        )
        video_style_transfer_main(
            pretrained_model_path=args.pretrained_model_path,
            content_inv_path=content_inv,
            style_inv_path=style_inv,
            mask_path=args.mask_path,
            output_path=args.stylizations_out,
            weight_dtype=args.weight_dtype,
            time_steps=args.time_steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
