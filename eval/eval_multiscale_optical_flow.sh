#!/bin/bash
# Script to evaluate the effectiveness of Plan A (Multi-scale Optical Flow Fusion)

export PYTHONPATH=$(pwd)

# Evaluation parameters
BASELINE_VIDEO="eval/video/bird0原始.mp4"
IMPROVED_VIDEO="eval/video/时序Abird0.mp4"
OUTPUT_DIR="eval/results/plan_a_bird0"

echo "=========================================="
echo "Evaluating the Effectiveness of Plan A (Multi-scale Optical Flow Fusion)"
echo "=========================================="
echo "Baseline Video (Plan A disabled): $BASELINE_VIDEO"
echo "Improved Video (Plan A enabled): $IMPROVED_VIDEO"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run evaluation
python eval/evaluate_temporal_consistency_hjh.py \
    --baseline_video "$BASELINE_VIDEO" \
    --improved_video "$IMPROVED_VIDEO" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Evaluation completed!"
echo "View detailed report: $OUTPUT_DIR/evaluation_report.txt"
echo "View visualization plots: $OUTPUT_DIR/"