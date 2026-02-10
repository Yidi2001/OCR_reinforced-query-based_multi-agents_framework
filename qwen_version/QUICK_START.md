# 🚀 Qwen2-VL Pipeline 快速开始

## 📋 前置条件

### 1. 模型准备
确保 Qwen2-VL 模型已下载到正确路径：
```bash
ls -lh /root/program2/models/Qwen2-VL-7B-Instruct/
```

### 2. 依赖安装
```bash
pip install qwen-vl-utils
```

### 3. 其他模型（OCR agents）
- PaddleOCR Layout 模型
- TrOCR 模型
- 分类器模型: `checkpoints/printed_vs_hand_best.pth`

## 🎯 使用示例

### 示例 1: 单张图片 - 简单识别

```bash
cd /root/program2/qwen_version

python3 << 'EOF'
from pipeline import process_image

result = process_image(
    image_path="../ocrbench1/OCRBench_Images/ChartQA/test/png/427.png",
    query="What is written in the image?",
    verbose=True
)

print(f"\n最终答案: {result['final_answer']}")
EOF
```

**预期行为**:
- ✅ 检测到简单识别任务
- ✅ 跳过 OCR agents
- ✅ 直接使用 Qwen2-VL 推理
- ⏱️  耗时: ~10秒

### 示例 2: 单张图片 - 复杂分析

```bash
cd /root/program2/qwen_version

python3 << 'EOF'
from pipeline import process_image

result = process_image(
    image_path="../OCRBench_v2/EN_part/RVL_CDIP/0000049717.tif",
    query="What is the invoice number in the document?",
    verbose=True
)

print(f"\n最终答案: {result['final_answer']}")
EOF
```

**预期行为**:
- ✅ 检测到复杂分析任务
- ✅ 执行完整 OCR pipeline
  - Layout Detection
  - Printed/Hand OCR
  - Layout Selection
  - Qwen2-VL Refinement
- ⏱️  耗时: ~42秒

### 示例 3: 批量处理（从 JSON）

```bash
cd /root/program2/qwen_version

python3 << 'EOF'
from pipeline import process_from_json

process_from_json(
    json_path="../OCRBench_v2/OCRBench_v2.json",
    output_file="result_qwen_test.json",
    limit=5,  # 只处理前5个样本
    enable_refinement=True
)
EOF
```

**预期输出**:
```
处理进度: 100%|████████████| 5/5 [00:50<00:00, 10.00s/it]
✓ 处理完成
  - 总样本数: 5
  - 成功: 5
  - 失败: 0
✓ 结果已保存到: result_qwen_test.json
```

### 示例 4: 使用命令行参数

```bash
cd /root/program2/qwen_version

# 处理单张图片
python pipeline.py \
  --image ../ocrbench1/OCRBench_Images/ChartQA/test/png/427.png \
  --query "What is written in the image?" \
  --output test_output.json

# 批量处理
python pipeline.py \
  --json ../OCRBench_v2/OCRBench_v2.json \
  --output result_qwen.json \
  --limit 100
```

## 🔧 参数说明

### `process_image()` 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `image_path` | str | 必需 | 图片路径 (支持 .jpg, .png, .tif 等) |
| `query` | str | 必需 | 用户查询/问题 |
| `output_path` | str | None | 输出 JSON 路径 (可选) |
| `example_name` | str | None | 任务名称 (用于日志) |
| `generate_summary` | bool | True | 是否生成摘要 |
| `enable_refinement` | bool | True | 是否启用答案精炼 |
| `verbose` | bool | True | 是否显示详细日志 |
| `orchestrator` | Object | None | 复用的编排器 (批处理优化) |
| `refiner` | Object | None | 复用的 refiner (批处理优化) |

### `process_from_json()` 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `json_path` | str | 必需 | 输入 JSON 文件路径 |
| `output_file` | str | "predictions.json" | 输出 JSON 文件路径 |
| `limit` | int | None | 限制处理样本数 (None=全部) |
| `enable_refinement` | bool | True | 是否启用答案精炼 |

## 📊 输出格式

### 单张图片输出 (JSON)

```json
{
  "image_path": "path/to/image.jpg",
  "query": "What is written in the image?",
  "query_type": "simple_recognition",
  "skip_agents": true,
  "final_answer": "The image contains...",
  "planning_time": 1.23,
  "total_time": 10.45
}
```

### 批量处理输出 (JSON)

```json
[
  {
    "index": 0,
    "question": "What is written in the image?",
    "image_path": "path/to/image1.jpg",
    "predict": "The image contains...",
    "processing_time": 10.45
  },
  {
    "index": 1,
    "question": "What is the invoice number?",
    "image_path": "path/to/image2.jpg",
    "predict": "Invoice number: 12345",
    "processing_time": 42.18
  }
]
```

## 🐛 常见问题

### 问题 1: ModuleNotFoundError: No module named 'qwen_vl_utils'
**解决方法**:
```bash
pip install qwen-vl-utils
```

### 问题 2: CUDA out of memory
**解决方法**:
- 使用量化版本模型 (INT8)
- 减少 `num_crops` 参数
- 或使用 Phi3.5-Vision 版本 (显存更小)

### 问题 3: RuntimeContext 相关错误
**解决方法**:
确保 `runtime_context.py` 在同一目录下：
```bash
ls qwen_version/runtime_context.py
```

### 问题 4: 图片格式不支持 (.tif)
**解决方法**:
Pipeline 自动支持 `.tif` 格式，会自动转换为 `.png`。
确保 `image_format_utils.py` 存在。

## 📈 性能优化

### 1. 批处理模式
使用 `process_from_json()` 时，模型只加载一次，多个样本复用：
```python
process_from_json("OCRBench.json", limit=100)  # 模型只加载1次
```

### 2. 禁用详细输出
```python
process_image(..., verbose=False)  # 减少日志开销
```

### 3. 跳过答案精炼
如果只需要 OCR 结果，不需要最终答案：
```python
process_image(..., enable_refinement=False)
```

### 4. 使用量化模型
修改模型加载参数（需要手动修改 `qwen_task_planner.py` 和 `qwen_refiner.py`）：
```python
torch_dtype=torch.bfloat16  # 或 torch.int8
```

## 🎯 推荐工作流程

### 开发/调试阶段
```python
# 使用 verbose=True，查看详细日志
result = process_image(
    image_path="test.jpg",
    query="test query",
    verbose=True
)
```

### 生产/批量处理阶段
```python
# 使用 verbose=False，提高速度
process_from_json(
    json_path="OCRBench.json",
    output_file="result.json",
    limit=None  # 处理全部
)
```

## 🔗 更多信息

- 详细文档: `README.md`
- 版本对比: `COMPARE.md`
- 测试脚本: `test_qwen_pipeline.py`
- 原版 (Phi3.5): `../case2/`

## ✅ 验证安装

运行测试脚本验证安装：
```bash
cd /root/program2/qwen_version
python test_qwen_pipeline.py
```

预期输出:
```
✓ 简单识别测试完成
✓ 复杂分析测试完成
✓ 所有测试完成
```
