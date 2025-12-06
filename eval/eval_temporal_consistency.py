"""
Script to evaluate temporal consistency of videos,
comparing results with and without Plan A (multi-scale flow fusion).
"""
import argparse
import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def extract_frames_from_video(video_path, max_frames=None):
    """Extract frames from a video"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and count >= max_frames:
            break
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        count += 1
    
    cap.release()
    return np.array(frames), fps


def extract_frames_from_dir(frames_dir):
    """Load frames from a directory"""
    frames = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = np.array(Image.open(frame_path))
        if len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)
        frames.append(frame)
    
    return np.array(frames)


def compute_frame_difference(frame1, frame2):
    """Compute the difference between two frames"""
    # Convert to float32 for calculation
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    
    # L2 distance (Euclidean distance)
    l2_diff = np.sqrt(np.sum((f1 - f2) ** 2, axis=-1))
    mean_l2 = np.mean(l2_diff)
    
    # L1 distance (Manhattan distance)
    l1_diff = np.sum(np.abs(f1 - f2), axis=-1)
    mean_l1 = np.mean(l1_diff)
    
    # Perceptual difference: computed in Lab color space
    f1_lab = cv2.cvtColor(f1.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    f2_lab = cv2.cvtColor(f2.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_diff = np.sqrt(np.sum((f1_lab - f2_lab) ** 2, axis=-1))
    mean_lab = np.mean(lab_diff)
    
    return {
        'l2': mean_l2,
        'l1': mean_l1,
        'lab': mean_lab,
        'max_l2': np.max(l2_diff),
        'std_l2': np.std(l2_diff),
    }


def compute_temporal_consistency(frames):
    """Compute temporal consistency metrics"""
    n_frames = len(frames)
    if n_frames < 2:
        return {}
    
    frame_diffs = []
    for i in range(n_frames - 1):
        diff = compute_frame_difference(frames[i], frames[i+1])
        frame_diffs.append(diff)
    
    # Statistics
    l2_diffs = [d['l2'] for d in frame_diffs]
    l1_diffs = [d['l1'] for d in frame_diffs]
    lab_diffs = [d['lab'] for d in frame_diffs]
    
    # Compute various smoothness metrics
    std_l2 = np.std(l2_diffs)
    mean_l2 = np.mean(l2_diffs)
    cv_l2 = std_l2 / mean_l2 if mean_l2 > 0 else 0  # Coefficient of Variation
    
    # Smoothness metric 1: Based on standard deviation (original method)
    smoothness_std = 1.0 / (1.0 + std_l2)
    
    # Smoothness metric 2: Based on coefficient of variation (more reasonable, considering the mean)
    smoothness_cv = 1.0 / (1.0 + cv_l2)
    
    # Smoothness metric 3: Based on maximum difference (penalizes extreme values)
    max_l2 = np.max(l2_diffs)
    smoothness_max = 1.0 / (1.0 + max_l2 / mean_l2) if mean_l2 > 0 else 0
    
    # Combined smoothness metric (weighted average)
    temporal_smoothness = (0.4 * smoothness_cv + 0.4 * smoothness_std + 0.2 * smoothness_max)
    
    return {
        'mean_l2_diff': mean_l2,
        'std_l2_diff': std_l2,
        'cv_l2_diff': cv_l2,  # Coefficient of Variation
        'mean_l1_diff': np.mean(l1_diffs),
        'std_l1_diff': np.std(l1_diffs),
        'mean_lab_diff': np.mean(lab_diffs),
        'std_lab_diff': np.std(lab_diffs),
        'max_l2_diff': max_l2,
        'frame_diffs': frame_diffs,
        'temporal_smoothness': temporal_smoothness,  # Combined smoothness metric
        'temporal_smoothness_std': smoothness_std,  # Smoothness based on standard deviation
        'temporal_smoothness_cv': smoothness_cv,  # Smoothness based on coefficient of variation
    }


def compute_optical_flow_consistency(frames):
    """Compute temporal consistency using optical flow"""
    if len(frames) < 2:
        return {}
    
    # Using Farneback optical flow algorithm
    flow_magnitudes = []
    flow_directions = []
    
    prev_gray = cv2.cvtColor(frames[0].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Compute magnitude and direction of optical flow
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_magnitudes.append(magnitude)
        
        prev_gray = curr_gray
    
    # Calculate optical flow consistency
    mean_flows = [np.mean(mag) for mag in flow_magnitudes]
    std_flows = [np.std(mag) for mag in flow_magnitudes]
    
    return {
        'mean_flow_magnitude': np.mean(mean_flows),
        'std_flow_magnitude': np.std(mean_flows),
        'flow_variance': np.mean(std_flows),  # The smaller the flow variance, the better the consistency
        'flow_smoothness': 1.0 / (1.0 + np.std(mean_flows)),
    }


def compute_color_stability(frames):
    """Compute color stability"""
    mean_colors = []
    std_colors = []
    
    for frame in frames:
        # Compute mean color and standard deviation for each frame
        mean_color = np.mean(frame.reshape(-1, 3), axis=0)
        std_color = np.std(frame.reshape(-1, 3), axis=0)
        mean_colors.append(mean_color)
        std_colors.append(std_color)
    
    mean_colors = np.array(mean_colors)
    std_colors = np.array(std_colors)
    
    # Calculate color changes
    color_changes = np.diff(mean_colors, axis=0)
    color_change_magnitude = np.sqrt(np.sum(color_changes ** 2, axis=1))
    
    return {
        'mean_color_change': np.mean(color_change_magnitude),
        'std_color_change': np.std(color_change_magnitude),
        'max_color_change': np.max(color_change_magnitude),
        'color_stability': 1.0 / (1.0 + np.mean(color_change_magnitude)),
    }


def visualize_frame_differences(frame_diffs, output_path, title="Frame Differences"):
    """Visualize frame-to-frame differences"""
    n_frames = len(frame_diffs)
    frame_indices = np.arange(n_frames)
    
    l2_diffs = [d['l2'] for d in frame_diffs]
    l1_diffs = [d['l1'] for d in frame_diffs]
    lab_diffs = [d['lab'] for d in frame_diffs]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    axes[0].plot(frame_indices, l2_diffs, 'b-', marker='o', linewidth=2, markersize=4)
    axes[0].set_ylabel('L2 Distance', fontsize=12)
    axes[0].set_title('Frame-to-Frame L2 Difference', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=np.mean(l2_diffs), color='r', linestyle='--', label=f'Mean: {np.mean(l2_diffs):.2f}')
    axes[0].legend()
    
    axes[1].plot(frame_indices, l1_diffs, 'g-', marker='s', linewidth=2, markersize=4)
    axes[1].set_ylabel('L1 Distance', fontsize=12)
    axes[1].set_title('Frame-to-Frame L1 Difference', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=np.mean(l1_diffs), color='r', linestyle='--', label=f'Mean: {np.mean(l1_diffs):.2f}')
    axes[1].legend()
    
    axes[2].plot(frame_indices, lab_diffs, 'm-', marker='^', linewidth=2, markersize=4)
    axes[2].set_xlabel('Frame Index', fontsize=12)
    axes[2].set_ylabel('LAB Distance', fontsize=12)
    axes[2].set_title('Frame-to-Frame LAB Color Difference', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=np.mean(lab_diffs), color='r', linestyle='--', label=f'Mean: {np.mean(lab_diffs):.2f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_visualization(frames_baseline, frames_improved, output_dir):
    """Create comparison visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate frame differences
    baseline_diffs = []
    improved_diffs = []
    
    for i in range(len(frames_baseline) - 1):
        baseline_diffs.append(compute_frame_difference(frames_baseline[i], frames_baseline[i+1]))
        improved_diffs.append(compute_frame_difference(frames_improved[i], frames_improved[i+1]))
    
    # Plot comparison
    frame_indices = np.arange(len(baseline_diffs))
    baseline_l2 = [d['l2'] for d in baseline_diffs]
    improved_l2 = [d['l2'] for d in improved_diffs]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frame_indices, baseline_l2, 'r-', marker='o', linewidth=2, markersize=5, label='Baseline (No Plan A)', alpha=0.7)
    ax.plot(frame_indices, improved_l2, 'b-', marker='s', linewidth=2, markersize=5, label='With Plan A (Multi-Scale Flow)', alpha=0.7)
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('L2 Distance', fontsize=12)
    ax.set_title('Temporal Consistency Comparison: Frame-to-Frame L2 Distance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistical information
    baseline_mean = np.mean(baseline_l2)
    improved_mean = np.mean(improved_l2)
    improvement = ((baseline_mean - improved_mean) / baseline_mean) * 100
    
    textstr = f'Baseline Mean: {baseline_mean:.2f}\nImproved Mean: {improved_mean:.2f}\nImprovement: {improvement:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_l2_diff.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot comparison of LAB color differences
    baseline_lab = [d['lab'] for d in baseline_diffs]
    improved_lab = [d['lab'] for d in improved_diffs]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frame_indices, baseline_lab, 'r-', marker='o', linewidth=2, markersize=5, label='Baseline (No Plan A)', alpha=0.7)
    ax.plot(frame_indices, improved_lab, 'b-', marker='s', linewidth=2, markersize=5, label='With Plan A (Multi-Scale Flow)', alpha=0.7)
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('LAB Color Distance', fontsize=12)
    ax.set_title('Temporal Consistency Comparison: LAB Color Difference', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    baseline_lab_mean = np.mean(baseline_lab)
    improved_lab_mean = np.mean(improved_lab)
    improvement_lab = ((baseline_lab_mean - improved_lab_mean) / baseline_lab_mean) * 100
    
    textstr = f'Baseline Mean: {baseline_lab_mean:.2f}\nImproved Mean: {improved_lab_mean:.2f}\nImprovement: {improvement_lab:.2f}%'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_lab_diff.png'), dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_video(video_path_or_dir, use_video=True):
    """Evaluate temporal consistency of a single video"""    
    print(f"Loading video/frames: {video_path_or_dir}")
    
    if use_video and os.path.isfile(video_path_or_dir):
        frames, fps = extract_frames_from_video(video_path_or_dir)
    elif os.path.isdir(video_path_or_dir):
        frames = extract_frames_from_dir(video_path_or_dir)
        fps = None
    else:
        raise ValueError(f"Invalid path: {video_path_or_dir}")
    
    print(f"Loaded {len(frames)} frames")
    
    # Compute various metrics
    temporal_metrics = compute_temporal_consistency(frames)
    flow_metrics = compute_optical_flow_consistency(frames)
    color_metrics = compute_color_stability(frames)
    
    return {
        'temporal': temporal_metrics,
        'optical_flow': flow_metrics,
        'color': color_metrics,
        'frames': frames,
    }


def generate_report(baseline_result, improved_result, output_dir):
    """Generate evaluation report"""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'baseline': {},
        'improved': {},
        'improvement': {},
    }
    
    # Extract metrics
    baseline_temp = baseline_result['temporal']
    improved_temp = improved_result['temporal']
    baseline_flow = baseline_result['optical_flow']
    improved_flow = improved_result['optical_flow']
    baseline_color = baseline_result['color']
    improved_color = improved_result['color']
    
    # Temporal consistency metrics
    baseline_l2 = baseline_temp['mean_l2_diff']
    improved_l2 = improved_temp['mean_l2_diff']
    l2_improvement = ((baseline_l2 - improved_l2) / baseline_l2) * 100
    
    baseline_std = baseline_temp['std_l2_diff']
    improved_std = improved_temp['std_l2_diff']
    std_change = ((improved_std - baseline_std) / baseline_std) * 100
    
    baseline_smooth = baseline_temp.get('temporal_smoothness', 1.0 / (1.0 + baseline_std))
    improved_smooth = improved_temp.get('temporal_smoothness', 1.0 / (1.0 + improved_std))
    smooth_improvement = ((improved_smooth - baseline_smooth) / baseline_smooth) * 100
    
    # Calculate coefficient of variation change
    baseline_cv = baseline_temp.get('cv_l2_diff', baseline_std / baseline_l2 if baseline_l2 > 0 else 0)
    improved_cv = improved_temp.get('cv_l2_diff', improved_std / improved_l2 if improved_l2 > 0 else 0)
    cv_improvement = ((baseline_cv - improved_cv) / baseline_cv) * 100 if baseline_cv > 0 else 0
    
    report['baseline'] = {
        'mean_l2_diff': float(baseline_l2),
        'std_l2_diff': float(baseline_temp['std_l2_diff']),
        'temporal_smoothness': float(baseline_smooth),
        'mean_flow_variance': float(baseline_flow.get('flow_variance', 0)),
        'flow_smoothness': float(baseline_flow.get('flow_smoothness', 0)),
        'color_stability': float(baseline_color['color_stability']),
    }
    
    report['improved'] = {
        'mean_l2_diff': float(improved_l2),
        'std_l2_diff': float(improved_temp['std_l2_diff']),
        'temporal_smoothness': float(improved_smooth),
        'mean_flow_variance': float(improved_flow.get('flow_variance', 0)),
        'flow_smoothness': float(improved_flow.get('flow_smoothness', 0)),
        'color_stability': float(improved_color['color_stability']),
    }
    
    report['improvement'] = {
        'l2_reduction_percent': float(l2_improvement),
        'std_change_percent': float(std_change),
        'smoothness_improvement_percent': float(smooth_improvement),
        'cv_improvement_percent': float(cv_improvement),
        'flow_variance_reduction': float(baseline_flow.get('flow_variance', 0) - improved_flow.get('flow_variance', 0)),
    }
    
    # Save JSON report
    json_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Prepare text description variables
    l2_dir = 'decreased' if l2_improvement > 0 else 'increased'
    l2_desc = 'overall more consistent' if l2_improvement > 0 else 'slightly more divergent'
    std_dir = 'decreased' if std_change < 0 else 'increased'
    std_desc = 'more stable' if std_change < 0 else 'more fluctuation'
    smooth_dir = 'improved' if smooth_improvement > 0 else 'degraded'
    cv_desc = 'more stable' if cv_improvement > 0 else 'more fluctuation'
    flow_dir = 'reduced' if report['improvement']['flow_variance_reduction'] > 0 else 'increased'
    l2_change_desc = 'decreased' if l2_improvement > 0 else 'increased'
    smooth_consistency = 'improved' if smooth_improvement > 0 else 'slightly degraded'
    
    # Generate text report
    text_report = f"""
========================================
Temporal Consistency Evaluation Report
========================================

【Baseline Model (Without Scheme A)】
- Mean Frame-to-Frame L2 Difference: {baseline_l2:.4f}
- L2 Difference Standard Deviation: {baseline_temp['std_l2_diff']:.4f}
- Temporal Smoothness: {baseline_smooth:.4f}
- Optical Flow Smoothness: {baseline_flow.get('flow_smoothness', 0):.4f}
- Color Stability: {baseline_color['color_stability']:.4f}

【Improved Model (With Scheme A: Multi-Scale Optical Flow Fusion)】
- Mean Frame-to-Frame L2 Difference: {improved_l2:.4f}
- L2 Difference Standard Deviation: {improved_temp['std_l2_diff']:.4f}
- Temporal Smoothness: {improved_smooth:.4f}
- Optical Flow Smoothness: {improved_flow.get('flow_smoothness', 0):.4f}
- Color Stability: {improved_color['color_stability']:.4f}

【Improvement】
- L2 Difference Change: {l2_improvement:+.2f}% ({l2_dir}, {l2_desc})
- L2 Difference Standard Deviation Change: {std_change:+.2f}% ({std_dir}, {std_desc})
- Smoothness Change: {smooth_improvement:+.2f}% ({smooth_dir})
- Coefficient of Variation Change: {cv_improvement:+.2f}% ({cv_desc})
- Optical Flow Change: {report['improvement']['flow_variance_reduction']:+.4f} ({flow_dir})

【Analysis】
Smoothness metric calculation: 1 / (1 + standard deviation)
- Interpretation of results:
  1. L2 difference {l2_dir} by {abs(l2_improvement):.2f}%, indicating the average frame-to-frame difference has {l2_change_desc}
  2. Standard deviation {std_dir} by {abs(std_change):.2f}%, indicating the distribution of differences has {std_desc}
  3. Smoothness {smooth_dir} by {abs(smooth_improvement):.2f}%, reflecting temporal consistency {smooth_consistency}
  4. It is recommended to pay attention to the change in the coefficient of variation, which considers relative volatility and better reflects stability.

========================================
"""
    
    print(text_report)
    
    text_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate video temporal consistency')
    parser.add_argument('--baseline_video', type=str, required=True,
                       help='Baseline video path (without Scheme A)')
    parser.add_argument('--improved_video', type=str, required=True,
                       help='Improved video path (with Scheme A)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory')
    parser.add_argument('--baseline_is_dir', action='store_true',
                       help='baseline_video is a frame directory instead of a video file')
    parser.add_argument('--improved_is_dir', action='store_true',
                       help='improved_video is a frame directory instead of a video file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting temporal consistency evaluation...")
    print("=" * 60)
    
    # Evaluate baseline video
    print("\n1. Evaluating baseline model (without Scheme A)...")
    baseline_result = evaluate_video(
        args.baseline_video,
        use_video=not args.baseline_is_dir
    )
    
    # Evaluate improved video
    print("\n2. Evaluating improved model (with Scheme A)...")
    improved_result = evaluate_video(
        args.improved_video,
        use_video=not args.improved_is_dir
    )
    
    # Generate report
    print("\n3. Generating evaluation report...")
    report = generate_report(baseline_result, improved_result, args.output_dir)
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    visualize_frame_differences(
        baseline_result['temporal']['frame_diffs'],
        os.path.join(args.output_dir, 'baseline_frame_diffs.png'),
        title='Baseline: Frame-to-Frame Differences'
    )
    visualize_frame_differences(
        improved_result['temporal']['frame_diffs'],
        os.path.join(args.output_dir, 'improved_frame_diffs.png'),
        title='Improved (Plan A): Frame-to-Frame Differences'
    )
    create_comparison_visualization(
        baseline_result['frames'],
        improved_result['frames'],
        args.output_dir
    )
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")
    print("\nKey improvement metrics:")
    print(f"  - L2 difference reduction: {report['improvement']['l2_reduction_percent']:.2f}%")
    print(f"  - Smoothness improvement: {report['improvement']['smoothness_improvement_percent']:.2f}%")


if __name__ == "__main__":
    main()
