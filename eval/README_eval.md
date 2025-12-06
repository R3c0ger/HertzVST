# Plan A (Multi-scale Optical Flow Fusion) Effect Evaluation Guide

## Overview

This evaluation tool is used to quantitatively assess the improvement effect of Plan A (Multi-scale Optical Flow Fusion) on video temporal consistency.

## Evaluation Metrics

1. **Frame-to-Frame Difference**
   - L2 Distance: Measures overall differences between frames
   - L1 Distance: Measures pixel-level differences between frames
   - LAB Color Space Distance: Color differences more aligned with human visual perception

2. **Temporal Smoothness**
   - Calculated based on standard deviation of frame-to-frame differences
   - Higher values indicate smoother video

3. **Optical Flow Consistency**
   - Uses Farneback optical flow algorithm to calculate inter-frame motion
   - Smaller optical flow changes indicate better temporal consistency

4. **Color Stability**
   - Measures color changes between frames
   - Higher values indicate more stable colors

## Usage Methods

### Method 1: Using the Evaluation Script (Recommended)

```bash
# Run the evaluation script directly
bash eval/eval_multiscale_optical_flow.sh
```

### Method 2: Manually Running the Python Script

```bash
# Evaluate video files
python eval/eval_temporal_consistency.py \
    --baseline_video results/stylizations/sd/01_0/output_video.mp4 \
    --improved_video results/stylizations/sd/01_0/output_video_hjh.mp4 \
    --output_dir results/evaluation/plan_a_01_0

# Evaluate frame directories
python eval/eval_temporal_consistency.py \
    --baseline_video results/stylizations/sd/01_0/frames \
    --improved_video results/stylizations/sd/01_0/frames_plan_a \
    --baseline_is_dir \
    --improved_is_dir \
    --output_dir results/evaluation/plan_a_01_0
```

## Output Results

After evaluation, the following files will be generated in the output directory:

1. **evaluation_report.json** - Detailed report in JSON format
2. **evaluation_report.txt** - Human-readable text report
3. **baseline_frame_diffs.png** - Frame-to-frame difference chart for baseline model
4. **improved_frame_diffs.png** - Frame-to-frame difference chart for improved model
5. **comparison_l2_diff.png** - L2 difference comparison chart
6. **comparison_lab_diff.png** - LAB color difference comparison chart

## Result Interpretation

### Improvement Effect Assessment Criteria

1. **L2 Difference Reduction Percentage**
   - Positive values indicate that the improved model has smaller frame-to-frame differences and better temporal consistency
   - Typically, 3-10% improvement indicates noticeable effects

2. **Smoothness Improvement Percentage**
   - Positive values indicate smoother video
   - 5-15% improvement indicates significant enhancement

3. **Visualization Charts**
   - In comparison charts, the improved curve should generally be below the baseline curve
   - Smaller curve fluctuations are better

### Example Interpretation

If the report shows:
- **L2 Difference Reduced: 8.5%** → Indicates that after enabling Plan A, frame-to-frame differences were reduced by an average of 8.5%
- **Smoothness Improved: 12.3%** → Indicates that video temporal smoothness was improved by 12.3%

## Notes

1. Ensure both videos have the same number of frames
2. Ensure video content corresponds (same content and style)
3. Evaluation takes some time, please be patient
4. If videos are large, evaluation may require more memory

## Troubleshooting

If problems are encountered:

1. **Video Cannot Be Opened**
   - Check if the video path is correct
   - Confirm video format is supported (MP4, AVI, etc.)

2. **Frame Count Mismatch**
   - Ensure both videos have the same number of frames
   - Can first use video processing tools to unify frame counts

3. **Insufficient Memory**
   - Can reduce the number of frames for evaluation
   - Or evaluate frame directories instead of complete videos

## Advanced Usage

### Evaluate Only Specific Metrics

Modify the `eval_temporal_consistency.py` script, comment out the calculation sections for metrics not needed.

### Custom Visualization

Modify the visualization functions in the `eval/eval_temporal_consistency.py` script to customize chart styles and output formats.