#!/bin/bash
# 消融实验（强制全部走 pipeline，禁用门控）
# 用法：进入 screen 后执行 bash xiaorong/run_ablations_no_gate.sh

set -e
cd /root/program2

mkdir -p xiaorong/result_no_gate

echo ""
echo "========================================"
echo "消融 1/3: 屏蔽字体检测（无门控）"
echo "========================================"
python phi3.5-pipeline/pipeline.py \
  --json xiaorong/OCRBench.json \
  --no-gating \
  --no-target-detection \
  --output xiaorong/result_no_gate/no_gate_no_td.json
echo "✓ 消融1 完成"

echo ""
echo "========================================"
echo "消融 2/3: 屏蔽 Prompt 设计（无门控）"
echo "========================================"
python phi3.5-pipeline/pipeline.py \
  --json xiaorong/OCRBench.json \
  --no-gating \
  --no-prompt-generation \
  --output xiaorong/result_no_gate/no_gate_no_pg.json
echo "✓ 消融2 完成"

echo ""
echo "========================================"
echo "消融 3/3: 屏蔽相关性选择（无门控）"
echo "========================================"
python phi3.5-pipeline/pipeline.py \
  --json xiaorong/OCRBench.json \
  --no-gating \
  --no-relevance-selection \
  --output xiaorong/result_no_gate/no_gate_no_rs.json
echo "✓ 消融3 完成"

echo ""
echo "========================================"
echo "全部完成！结果保存在 xiaorong/result_no_gate/"
echo "  - no_gate_no_td.json      消融字体检测"
echo "  - no_gate_no_pg.json      消融Prompt设计"
echo "  - no_gate_no_rs.json      消融相关性选择"
echo "========================================"
