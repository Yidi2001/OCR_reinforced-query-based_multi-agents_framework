#!/bin/bash
# 门控模块 threshold 超参数扫描实验（OCRBench 数据集）
# 使用 gate_trainer（threshold 默认=0.505）
# 在默认基础上一大一小：0.405 / 0.605
# 对 phi3.5-pipeline 和 qwen_version/pipeline 各做两次，共 4 次
#
# 用法：进入 screen 后执行 bash xiaorong/run_threshold_sweep.sh

set -e
cd /root/program2

META="gate_trainer/gating_meta.json"
PHI_PY="phi3.5-pipeline/pipeline.py"
QWEN_PY="qwen_version/pipeline.py"
INPUT="ocrbench1/过程文件及脚本/OCRBench.json"
OUTDIR="xiaorong/threshold_sweep"
mkdir -p "$OUTDIR"

# 读取默认 threshold
DEFAULT_THRESH=$(python3 -c "import json; print(json.load(open('$META'))['threshold'])")
echo "gate_trainer 默认 threshold: $DEFAULT_THRESH"

# ---- 工具函数 ----

# 修改 gating_meta.json 中的 threshold
set_threshold() {
    local val=$1
    python3 -c "
import json
with open('$META') as f:
    meta = json.load(f)
meta['threshold'] = $val
with open('$META', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'[threshold] 已设置为 $val')
"
}

# 恢复默认 threshold
restore_threshold() {
    python3 -c "
import json
with open('$META') as f:
    meta = json.load(f)
meta['threshold'] = $DEFAULT_THRESH
with open('$META', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'[threshold] 已恢复默认 $DEFAULT_THRESH')
"
}

# 将 pipeline.py 中的 gate_trainer 路径切换（gate_trainer_2 -> gate_trainer）
switch_to_gate_trainer() {
    sed -i 's|gate_trainer_2|gate_trainer|g' "$PHI_PY"
    sed -i 's|gate_trainer_2|gate_trainer|g' "$QWEN_PY"
    echo "[gate] pipeline 已切换到 gate_trainer"
}

# 恢复回 gate_trainer_2
restore_gate_trainer() {
    sed -i 's|gate_trainer\b|gate_trainer_2|g' "$PHI_PY"
    sed -i 's|gate_trainer\b|gate_trainer_2|g' "$QWEN_PY"
    echo "[gate] pipeline 已恢复到 gate_trainer_2"
}

# 切换到 gate_trainer（实验期间保持）
switch_to_gate_trainer

# ============================================================
# 实验 1/4: phi3.5-pipeline  threshold = 0.405 (低)
# ============================================================
echo ""
echo "========================================"
echo "实验 1/4: phi3.5-pipeline  threshold=0.405 (低)"
echo "========================================"
set_threshold 0.405
python phi3.5-pipeline/pipeline.py \
    --json "$INPUT" \
    --output "$OUTDIR/phi_thresh_low_0.405.json"
restore_threshold
echo "✓ 实验1 完成"

# ============================================================
# 实验 2/4: phi3.5-pipeline  threshold = 0.605 (高)
# ============================================================
echo ""
echo "========================================"
echo "实验 2/4: phi3.5-pipeline  threshold=0.605 (高)"
echo "========================================"
set_threshold 0.605
python phi3.5-pipeline/pipeline.py \
    --json "$INPUT" \
    --output "$OUTDIR/phi_thresh_high_0.605.json"
restore_threshold
echo "✓ 实验2 完成"

# ============================================================
# 实验 3/4: qwen_version/pipeline  threshold = 0.405 (低)
# ============================================================
echo ""
echo "========================================"
echo "实验 3/4: qwen_version/pipeline  threshold=0.405 (低)"
echo "========================================"
set_threshold 0.405
python qwen_version/pipeline.py \
    --json "$INPUT" \
    --output "$OUTDIR/qwen_thresh_low_0.405.json"
restore_threshold
echo "✓ 实验3 完成"

# ============================================================
# 实验 4/4: qwen_version/pipeline  threshold = 0.605 (高)
# ============================================================
echo ""
echo "========================================"
echo "实验 4/4: qwen_version/pipeline  threshold=0.605 (高)"
echo "========================================"
set_threshold 0.605
python qwen_version/pipeline.py \
    --json "$INPUT" \
    --output "$OUTDIR/qwen_thresh_high_0.605.json"
restore_threshold
echo "✓ 实验4 完成"

# 所有实验完成，恢复 gate_trainer_2
restore_gate_trainer

# ============================================================
# 汇总
# ============================================================
echo ""
echo "========================================"
echo "全部完成！结果保存在 $OUTDIR/"
echo ""
echo "  phi_thresh_low_0.405.json    phi  threshold=0.405 (低)"
echo "  phi_thresh_high_0.605.json   phi  threshold=0.605 (高)"
echo "  qwen_thresh_low_0.405.json   qwen threshold=0.405 (低)"
echo "  qwen_thresh_high_0.605.json  qwen threshold=0.605 (高)"
echo ""
echo "默认 threshold ($DEFAULT_THRESH) 对照结果:"
echo "  phi  默认: all_ocr1_result/phi3.5/phi3.5_pipeline_gate.json"
echo "  qwen 默认: all_ocr1_result/gate_qwen.json"
echo "========================================"
