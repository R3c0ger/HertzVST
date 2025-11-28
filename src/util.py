import os
import random
from typing import Union, Sequence, Callable

import PIL
import imageio
import numpy as np
import requests
import torch
import torchvision
from PIL import Image
from einops import rearrange

from utils import logger


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_folder(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    # If the input is a numpy array, convert it to a tensor
    if isinstance(videos, np.ndarray):
        videos = torch.from_numpy(videos)
    videos = rearrange(videos, "b c t h w -> t b c h w")

    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0.0, 1.0)
        x = (x * 255).numpy().astype(np.uint8)
        imageio.imsave(os.path.join(path, f"%05d.png" % (i * 1)), x)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    # If the input is a numpy array, convert it to a tensor
    if isinstance(videos, np.ndarray):
        videos = torch.from_numpy(videos)
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []

    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0.0, 1.0)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def save_images_as_mp4(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:
    writer_edit = imageio.get_writer(save_path, fps=10)
    for i in images:
        init_image = i.convert("RGB")
        writer_edit.append_data(np.array(init_image))
    writer_edit.close()


def load_video_frames(frames_path, n_frames, image_size=(512, 512)):
    # Load paths
    paths = [f"{frames_path}/%05d.png" % (i * 1) for i in range(n_frames)]
    # paths = [
    #     os.path.join(frames_path, item)
    #     for item in sorted(os.listdir(frames_path), key=extract_number)
    # ]
    frames = []
    for p in paths:
        img = load_image(p, image_size=image_size)
        # check!
        if img.size != image_size:
            img = img.resize(image_size)
            raise ValueError(f"Frame size does not match config.image_size")
        # transforms to tensor
        np_img = np.array(img)
        # transforms to [-1, 1]
        normalized_img = (np_img / 127.5) - 1.0
        tensor_img = torch.from_numpy(normalized_img).permute(2, 0, 1).float()
        frames.append(tensor_img)
    video_tensor = torch.stack(frames)
    return video_tensor


def load_image(
    image: Union[str, PIL.Image.Image],
    convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] = None,
    image_size=None,
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
            A conversion method to apply to the image after loading it.
            When set to `None` the image will be converted "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image).resize(image_size)
        else:
            raise ValueError(
                "Incorrect path or URL. "
                "URLs must start with `http://` or `https://`, "
                f"and {image} is not a valid path."
            )
    else:
        raise ValueError(
            "Incorrect format used for the image. "
            "Should be a URL linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")
    return image


def load_ddim_latents_at_t(t, ddim_latents_path, is_x0=False):
    if is_x0:
        ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_x0_{t}.pt")
    else:
        ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_latents_{t}.pt")
    assert os.path.exists(
        ddim_latents_at_t_path
    ), f"Missing latents at t {t} path {ddim_latents_at_t_path}"
    ddim_latents_at_t = torch.load(ddim_latents_at_t_path, weights_only=True)
    return ddim_latents_at_t


def load_mask(mask_path="", n_frames=16):
    # image_files = [
    #     os.path.join(mask_path, file)
    #     for file in os.listdir(mask_path)
    #     if file.lower().endswith((".jpg", ".jpeg", ".png"))
    # ]
    image_files = [f"{mask_path}/%05d.png" % (i * 1) for i in range(n_frames)]
    image_files = sorted(image_files)

    images = [np.array(Image.open(image)) * 255 for image in image_files]
    # images = [np.array(Image.open(image)) for image in image_files]

    image_tensor = np.stack(images)
    image_tensor_torch = torch.from_numpy(image_tensor).unsqueeze(0)
    image_tensor_torch = image_tensor_torch.clip(0, 1)
    return image_tensor_torch


def extract_video_frames(video_path: str, output_frames_dir: str, image_size=(512, 512), max_frames: int = None):
    """
    Split a video into frames and save them as images.
    
    
    Args:
        video_path: Input video path
        output_frames_dir: Directory to save the output frames
        image_size: Output image size
        max_frames: Maximum number of frames to extract. If None, extract all frames.
    
    Returns:
        (Number of extracted frames, Video fps)
    """
    import decord
    decord.bridge.set_bridge("torch")
    
    os.makedirs(output_frames_dir, exist_ok=True)
    
    vr = decord.VideoReader(video_path, width=image_size[0], height=image_size[1])
    total_frames = len(vr)
    # Get the video's fps
    try:
        video_fps = vr.get_avg_fps()
    except:
        video_fps = 30.0  # Default fps
    
    if max_frames is not None:
        # Uniformly sample frames
        if len(vr) <= max_frames:
            sample_index = list(range(0, len(vr)))
        else:
            step = len(vr) // max_frames
            sample_index = list(range(0, len(vr), step))[:max_frames]
    else:
        sample_index = list(range(0, len(vr)))
    
    video = vr.get_batch(sample_index)
    
    # Process each frame
    for i, frame in enumerate(video):
        # decord returns a torch tensor when using the torch bridge
        if isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = np.array(frame)
        # Ensure values are within [0, 255] range
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        frame_img = Image.fromarray(frame_np)
        frame_img.save(os.path.join(output_frames_dir, f"%05d.png" % i))
    
    return len(sample_index), video_fps


def frames_to_video(frames_dir: str, output_video_path: str, fps: int = 8):
    """
    Combine frame images into a video.
    
    Args:
        frames_dir: Directory containing frame images
        output_video_path: Output video path
        fps: Video frame rate
    """
    import glob
    import re
    
    # Get all frame images and sort them numerically
    frame_files = glob.glob(os.path.join(frames_dir, "*.png"))
    if not frame_files:
        raise ValueError(f"No frame images found in {frames_dir}")
    
    # Sort by the number in the filename
    def extract_number(filename):
        match = re.search(r'(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else 0
    
    frame_files = sorted(frame_files, key=extract_number)
    
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        frames.append(np.array(img))
    
    os.makedirs(os.path.dirname(output_video_path) if os.path.dirname(output_video_path) else ".", exist_ok=True)
    imageio.mimsave(output_video_path, frames, fps=fps)
    logger.info(f"Video saved to: {output_video_path}")
