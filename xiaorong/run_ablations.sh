#!/bin/bash
# 消融实验顺序运行脚本
# 用法：进入 screen 后执行 bash xiaorong/run_ablations.sh

set -e
cd /root/program2

mkdir -p xiaorong/results

echo "========================================"
echo "消融实验 1/3: 屏蔽字体检测模块"
echo "========================================"
python phi3.5-pipeline/pipeline.py \
  --json xiaorong/OCRBench.json \
  --no-target-detection \
  --output xiaorong/results/ablation_no_target_detection.json
echo "✓ 消融1 完成"

echo ""
echo "========================================"
echo "消融实验 2/3: 屏蔽 Prompt 设计模块"
echo "========================================"
python phi3.5-pipeline/pipeline.py \
  --json xiaorong/OCRBench.json \
  --no-prompt-generation \
  --output xiaorong/results/ablation_no_prompt_generation.json
echo "✓ 消融2 完成"

echo ""
echo "========================================"
echo "消融实验 3/3: 屏蔽相关性选择模块"
echo "========================================"
python phi3.5-pipeline/pipeline.py \
  --json xiaorong/OCRBench.json \
  --no-relevance-selection \
  --output xiaorong/results/ablation_no_relevance_selection.json
echo "✓ 消融3 完成"

echo ""
echo "========================================"
echo "全部消融实验完成！结果保存在 xiaorong/results/"
echo "========================================"
