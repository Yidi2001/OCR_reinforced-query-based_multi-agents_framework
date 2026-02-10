#!/bin/bash
# 快速测试单张图片的脚本

echo "=================================="
echo "Case2 Pipeline 快速测试"
echo "=================================="
echo ""

# 进入项目目录
cd /root/program2

# 测试图片路径
IMAGE="OCRBench_Images/IC15_1811/imgs/000000229.jpg"

if [ ! -f "$IMAGE" ]; then
    echo "❌ 测试图片不存在: $IMAGE"
    exit 1
fi

echo "测试图片: $IMAGE"
echo ""

# 运行 Pipeline
python case2/orchestrator.py \
    --image "$IMAGE" \
    --query "识别这张图片中的所有文字" \
    --output case2_output/test_single_result.json

echo ""
echo "=================================="
echo "测试完成！"
echo "=================================="
echo ""
echo "查看结果:"
echo "  cat case2_output/test_single_result.json | python -m json.tool"
echo ""

