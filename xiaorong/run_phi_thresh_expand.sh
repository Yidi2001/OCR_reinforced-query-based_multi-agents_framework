#!/bin/bash
# phi3.5-pipeline threshold 扩展实验
# 已有: 0.405 / 0.505(默认) / 0.605
# 本次新增: 0.305 (更低) 和 0.705 (更高)
#
# 用法：进入 screen 后执行 bash xiaorong/run_phi_thresh_expand.sh

set -e
cd /root/program2

META="gate_trainer/gating_meta.json"
PHI_PY="phi3.5-pipeline/pipeline.py"
INPUT="ocrbench1/过程文件及脚本/OCRBench.json"
OUTDIR="xiaorong/threshold_sweep"
mkdir -p "$OUTDIR"

DEFAULT_THRESH=$(python3 -c "import json; print(json.load(open('$META'))['threshold'])")
echo "gate_trainer 默认 threshold: $DEFAULT_THRESH"

set_threshold() {
    python3 -c "
import json
with open('$META') as f:
    meta = json.load(f)
meta['threshold'] = $1
with open('$META', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'[threshold] 已设置为 $1')
"
}

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

switch_to_gate_trainer() {
    sed -i 's|gate_trainer_2|gate_trainer|g' "$PHI_PY"
    echo "[gate] pipeline 已切换到 gate_trainer"
}

restore_gate_trainer() {
    sed -i 's|gate_trainer\b|gate_trainer_2|g' "$PHI_PY"
    echo "[gate] pipeline 已恢复到 gate_trainer_2"
}

switch_to_gate_trainer

# ============================================================
# 实验 1/2: phi3.5-pipeline  threshold = 0.305 (更低)
# ============================================================
echo ""
echo "========================================"
echo "实验 1/2: phi3.5-pipeline  threshold=0.305 (更低)"
echo "========================================"
set_threshold 0.305
python phi3.5-pipeline/pipeline.py \
    --json "$INPUT" \
    --output "$OUTDIR/phi_thresh_0.305.json"
restore_threshold
echo "✓ 实验1 完成"

# ============================================================
# 实验 2/2: phi3.5-pipeline  threshold = 0.705 (更高)
# ============================================================
echo ""
echo "========================================"
echo "实验 2/2: phi3.5-pipeline  threshold=0.705 (更高)"
echo "========================================"
set_threshold 0.705
python phi3.5-pipeline/pipeline.py \
    --json "$INPUT" \
    --output "$OUTDIR/phi_thresh_0.705.json"
restore_threshold
echo "✓ 实验2 完成"

restore_gate_trainer

echo ""
echo "========================================"
echo "完成！phi3.5 threshold 完整系列："
echo "  threshold_sweep/phi_thresh_0.305.json   (新) 0.305"
echo "  threshold_sweep/phi_thresh_low_0.405.json      0.405"
echo "  all_ocr1_result/phi3.5/phi3.5_pipeline_gate.json  0.505 (默认)"
echo "  threshold_sweep/phi_thresh_high_0.605.json     0.605"
echo "  threshold_sweep/phi_thresh_0.705.json   (新) 0.705"
echo "========================================"
