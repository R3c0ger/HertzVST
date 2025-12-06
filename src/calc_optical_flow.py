import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from utils import logger


def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(image, device):
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image.unsqueeze(0).to(device)


def compute_optical_flow(model, image1, image2):
    with torch.no_grad():
        flow = model(image1, image2)
    return flow


def compute_multi_scale_optical_flow(
        model, image1, image2, 
        flow_scales=[1.0, 0.5, 0.25], 
        flow_fusion_method='weighted_average'
):
    """
    Multi-scale optical flow computation
    
    Args:
        model: RAFT optical flow model
        image1: First frame image tensor (B, C, H, W)
        image2: Second frame image tensor (B, C, H, W)
        flow_scales: List of scales, e.g., [1.0, 0.5, 0.25]
        flow_fusion_method: Optical flow fusion method, 'weighted_average' or 'max_confidence'
    
    Returns:
        Fused optical flow tensor (B, 2, H, W)
    """
    original_size = image1.shape[-2:]  # (H, W)
    flows = []
    
    for scale in flow_scales:
        # Compute scaled size
        scaled_h = int(original_size[0] * scale)
        scaled_w = int(original_size[1] * scale)
        
        # Resize images
        if scale < 1.0:
            resized_image1 = F.interpolate(
                image1, 
                size=(scaled_h, scaled_w), 
                mode='bilinear', 
                align_corners=False
            )
            resized_image2 = F.interpolate(
                image2, 
                size=(scaled_h, scaled_w), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            resized_image1 = image1
            resized_image2 = image2
        
        # Compute optical flow on resized images
        with torch.no_grad():
            flow = model(resized_image1, resized_image2)
        
        # Get the last layer optical flow
        flow_scaled = flow[-1]
        
        # If scaling was applied, upsample the flow to the original size
        if scale < 1.0:
            # Flow needs to be scaled proportionally (because image size changed)
            flow_scaled = F.interpolate(
                flow_scaled, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
            # Adjust flow values to match the new resolution
            flow_scaled = flow_scaled * (1.0 / scale)
        
        flows.append(flow_scaled)
    
    # Fuse optical flows from multiple scales
    fused_flow = fuse_flows(flows, flow_scales, method=flow_fusion_method)
    return fused_flow


def fuse_flows(flows, flow_scales, method='weighted_average'):
    """
    Fuse optical flows from multiple scales
    
    Args:
        flows: List of optical flow tensors, each of shape (B, 2, H, W)
        flow_scales: Corresponding list of scales
        method: Fusion method, 'weighted_average' or 'max_confidence'
    
    Returns:
        Fused optical flow tensor (B, 2, H, W)
    """
    if method == 'weighted_average':
        # Weighted average: larger scales (higher resolution) have greater weight
        weights = torch.tensor(flow_scales, device=flows[0].device, dtype=flows[0].dtype)
        weights = weights / weights.sum()  # Normalize
        
        fused_flow = torch.zeros_like(flows[0])
        for flow, weight in zip(flows, weights):
            fused_flow += flow * weight
        
        return fused_flow
    
    elif method == 'max_confidence':
        # Confidence-based fusion: select regions with smaller flow magnitude (usually more reliable)
        # Compute magnitude of each flow
        flow_magnitudes = [torch.norm(flow, dim=1, keepdim=True) for flow in flows]
        
        # Select flow with minimum magnitude (smoother, usually more reliable)
        min_magnitude = torch.stack(flow_magnitudes, dim=0).min(dim=0)[0]
        
        # Construct weights: smaller magnitude means higher weight
        weights = []
        for mag in flow_magnitudes:
            # Use Gaussian weights: exp(-(mag - min_mag)^2 / sigma^2)
            diff = (mag - min_magnitude) ** 2
            weight = torch.exp(-diff / (2 * 0.1 ** 2))
            weights.append(weight)
        
        # Normalize weights
        weight_sum = torch.stack(weights, dim=0).sum(dim=0)
        weights = [w / weight_sum for w in weights]
        
        # Weighted fusion
        fused_flow = torch.zeros_like(flows[0])
        for flow, weight in zip(flows, weights):
            fused_flow += flow * weight
        
        return fused_flow
    
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def compute_occlusion_mask(forward_flow_np, backward_flow_np, threshold=1.0):
    h, w, _ = forward_flow_np.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    coords2 = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    coords1_reconstructed = coords2 + forward_flow_np
    coords1_back = coords1_reconstructed + backward_flow_np
    error = np.linalg.norm(coords1_back - coords2, axis=-1)

    occlusion_mask = (error > threshold).astype(np.uint8) * 255
    return occlusion_mask


def warp_image_with_flow(image, flow):
    h, w, _ = flow.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    coords2 = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    coords1_reconstructed = coords2 + flow

    map_x = coords1_reconstructed[..., 0].astype(np.float32)
    map_y = coords1_reconstructed[..., 1].astype(np.float32)

    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return warped_image


def apply_mask(image, mask, original_image):
    mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0
    masked_image = image * (1 - mask_expanded) + original_image * mask_expanded
    return masked_image.astype(np.uint8)


def post_process(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def get_warp(
    image1_path,
    image2_path,
    ref_image1=None,
    ref_image2=None,
    occlusion_mask_save_path=None,
    warped_image_save_path=None,
    use_multi_scale=False,
    flow_scales=[1.0, 0.5, 0.25],
    flow_fusion_method='weighted_average',
):
    """
    Use optical flow to warp images, supporting multi-scale flow fusion.
    
    
    Args:
        image1_path: Path or numpy array of the first frame
        image2_path: Path or numpy array of the second frame
        ref_image1: Reference image 1 (for mask application)
        ref_image2: Reference image 2 (for warping)
        occlusion_mask_save_path: Path to save occlusion mask
        warped_image_save_path: Path to save warped image
        use_multi_scale: Whether to use multi-scale optical flow fusion
        flow_scales: List of scales, e.g., [1.0, 0.5, 0.25]
        flow_fusion_method: Optical flow fusion method, 'weighted_average' or 'max_confidence'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device)
    model.eval()
    if isinstance(image1_path, str):
        image1 = load_image(image1_path)
    else:
        image1 = image1_path
    if isinstance(image2_path, str):
        image2 = load_image(image2_path)
    else:
        image2 = image2_path

    if ref_image2 is None:
        ref_image2 = image2.copy()
    elif isinstance(ref_image2, str):
        ref_image2 = load_image(ref_image2)

    if ref_image1 is None:
        ref_image1 = image1.copy()
    elif isinstance(ref_image1, str):
        ref_image1 = load_image(ref_image1)

    image1_tensor = preprocess_image(image1, device)
    image2_tensor = preprocess_image(image2, device)

    # Use different optical flow computation methods based on multi-scale usage
    if use_multi_scale:
        forward_flow = compute_multi_scale_optical_flow(
            model, image1_tensor, image2_tensor, flow_scales, flow_fusion_method
        )
        backward_flow = compute_multi_scale_optical_flow(
            model, image2_tensor, image1_tensor, flow_scales, flow_fusion_method
        )
    else:
        forward_flow = compute_optical_flow(model, image1_tensor, image2_tensor)[-1]
        backward_flow = compute_optical_flow(model, image2_tensor, image1_tensor)[-1]

    forward_flow_np = forward_flow.squeeze().permute(1, 2, 0).cpu().numpy()
    backward_flow_np = backward_flow.squeeze().permute(1, 2, 0).cpu().numpy()

    occlusion_mask = compute_occlusion_mask(
        forward_flow_np, backward_flow_np, threshold=1.5
    )

    if occlusion_mask_save_path is not None:
        cv2.imwrite(occlusion_mask_save_path, occlusion_mask)
        logger.info(f"Occlusion mask save at {occlusion_mask_save_path}")

    warped_image = warp_image_with_flow(ref_image2, forward_flow_np)
    masked_image = apply_mask(warped_image, occlusion_mask, ref_image1)
    # post_processed_image = post_process(masked_image)
    post_processed_image = masked_image

    if warped_image_save_path is not None:
        post_processed_image_bgr = cv2.cvtColor(post_processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(warped_image_save_path, post_processed_image_bgr)
        logger.info(f"Warped image save at {warped_image_save_path}")
    return post_processed_image


if __name__ == "__main__":
    image1_path = "The path of image1"
    image2_path = "The path of image2"
    occlusion_mask_save_path = "occlusion_mask.png"
    warped_image_save_path = "warped_image_with_mask.png"
    get_warp(
        image1_path,
        image2_path,
        None,
        None,
        occlusion_mask_save_path,
        warped_image_save_path,
        use_multi_scale=True,  #  Enable multi-scale optical flow fusion
        flow_scales=[1.0, 0.5, 0.25],
    )
