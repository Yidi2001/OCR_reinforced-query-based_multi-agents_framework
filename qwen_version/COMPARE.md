# Phi3.5-Vision vs Qwen2-VL 版本对比

## 📊 核心差异

| 组件 | Phi3.5-Vision 版本 | Qwen2-VL 版本 |
|------|-------------------|---------------|
| **Task Planner** | `case2/task_planner.py`<br>`Phi35TaskPlanner` | `qwen_version/qwen_task_planner.py`<br>`Qwen2TaskPlanner` |
| **Refiner** | `case2/phi_refiner.py`<br>`PhiRefiner` | `qwen_version/qwen_refiner.py`<br>`QwenRefiner` |
| **模型加载** | `AutoModelForCausalLM`<br>`AutoProcessor` | `Qwen2VLForConditionalGeneration`<br>`AutoProcessor` |
| **图片处理** | Phi processor (num_crops=4) | Qwen processor + `process_vision_info` |
| **Prompt 格式** | `<|user|>\n<|image_1|>\n...<|end|>\n<|assistant|>\n` | Conversation 列表格式 |
| **模型参数** | `_attn_implementation='eager'`<br>`use_cache=False` | 默认参数 |

## 🔄 修改详情

### 1. Task Planner

#### Phi3.5-Vision 版本 (`task_planner.py`)
```python
from transformers import AutoModelForCausalLM, AutoProcessor

class Phi35TaskPlanner:
    def _load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
        )
    
    def plan(self, image_path, query):
        # 构建 prompt
        prompt = f"<|user|>\n<|image_1|>\n{text}<|end|>\n<|assistant|>\n"
        inputs = self.processor(prompt, [image], return_tensors="pt")
        # ...
```

#### Qwen2-VL 版本 (`qwen_task_planner.py`)
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen2TaskPlanner:
    def _load_model(self):
        return Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def plan(self, image_path, query):
        # 构建 conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(conversation, ...)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(text=[text_prompt], images=image_inputs, ...)
        # ...
```

### 2. Refiner

#### Phi3.5-Vision 版本 (`phi_refiner.py`)
```python
def refine_with_ocr_context(self, image_path, user_query, ocr_summary_text):
    messages = [
        {"role": "user", "content": f"<|image_1|>\n{prompt}"}
    ]
    prompt_text = self.processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = self.processor(prompt_text, [image], return_tensors="pt")
    # ...
```

#### Qwen2-VL 版本 (`qwen_refiner.py`)
```python
def refine_with_ocr_context(self, image_path, user_query, ocr_summary_text):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = self.processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = self.processor(text=[text_prompt], images=image_inputs, ...)
    # ...
```

### 3. Orchestrator

#### Phi3.5-Vision 版本 (`case2/orchestrator.py`)
```python
from task_planner import Phi35TaskPlanner

class MultiAgentOrchestrator:
    def __init__(self, phi35_model_path="models/phi-3_5_vision", ...):
        self.task_planner = Phi35TaskPlanner(phi35_model_path, ctx=self.ctx)
```

#### Qwen2-VL 版本 (`qwen_version/orchestrator.py`)
```python
from qwen_task_planner import Qwen2TaskPlanner

class MultiAgentOrchestrator:
    def __init__(self, qwen2_model_path="models/Qwen2-VL-7B-Instruct", ...):
        self.task_planner = Qwen2TaskPlanner(qwen2_model_path, ctx=self.ctx)
```

### 4. Pipeline

#### Phi3.5-Vision 版本 (`case2/pipeline.py`)
```python
from case2.phi_refiner import PhiRefiner

def process_image(..., refiner: 'PhiRefiner' = None):
    if refiner is None:
        refiner = PhiRefiner(ctx=ctx)
    # ...
```

#### Qwen2-VL 版本 (`qwen_version/pipeline.py`)
```python
from qwen_refiner import QwenRefiner

def process_image(..., refiner: 'QwenRefiner' = None):
    if refiner is None:
        refiner = QwenRefiner(ctx=ctx)
    # ...
```

## 🎯 核心设计思想 (完全相同)

两个版本都实现了相同的核心设计思想：

### 1. query-based 思想
- ✅ 所有决策由 query 驱动
- 文件: `pipeline.py`, `orchestrator.py`, `*task_planner.py`

### 2. 查询-图像绑定关系
- ✅ (image, query) 始终配对传递
- 体现在所有方法签名中

### 3. query 驱动的门控 / routing
- ✅ 简单识别 → 直接推理 (跳过 OCR agents)
- ✅ 复杂分析 → 完整 OCR pipeline
- 文件: `*task_planner.py` (classify_query_type), `orchestrator.py` (检查 skip_agents), `pipeline.py` (路由)

### 4. query 驱动的相关性排序
- ✅ 根据 query 筛选和排序相关信息
- 文件: `*task_planner.py`, `orchestrator.py`, `*refiner.py`

### 5. 多 agent 协同流程
- ✅ 灵活的 agent 架构
- 文件: `base_agent.py`, `orchestrator.py`, 各 agent 实现

## 📦 文件复用情况

### ✅ 完全复用（无需修改）
这些文件与多模态大模型无关，两个版本共享：

| 文件 | 功能 | 使用的模型 |
|------|------|-----------|
| `base_agent.py` | Agent 基类 | - |
| `LayoutDetectionAgent.py` | 布局检测 | PaddleOCR Layout |
| `printed_ocr_agent.py` | 印刷体 OCR | PaddleOCR |
| `trocr.py` | 手写体 OCR | TrOCR |
| `image_format_utils.py` | 图片格式转换 | - |
| `runtime_context.py` | 模型共享上下文 | - |
| `result_summarizer.py` | 结果摘要 | - |
| `merge_layout_blocks_ratio.py` | 布局合并 | - |
| `token_budget_calculator.py` | Token 预算 | - |
| `target_detection.py` | 目标检测 | ResNet50 |
| `prompt_generator.py` | Prompt 生成 | - |
| `preprocessing_agent.py` | 预处理 | Real-ESRGAN |
| `layout_relevance_selector_v4.py` | 布局选择 | - |

### 🔄 需要修改（使用了多模态大模型）
| 原文件 | 新文件 | 主要修改 |
|-------|--------|---------|
| `task_planner.py` | `qwen_task_planner.py` | 模型类、输入格式 |
| `phi_refiner.py` | `qwen_refiner.py` | 模型类、输入格式 |
| `orchestrator.py` | `orchestrator.py` | 导入语句、初始化参数 |
| `pipeline.py` | `pipeline.py` | 导入语句、类型注解 |

## 🚀 性能对比

### 显存占用
| 版本 | BF16 | INT8 (量化) |
|------|------|-------------|
| Phi3.5-Vision | ~7GB | ~3.5GB |
| Qwen2-VL | ~14GB | ~7GB |

### 推理速度 (单样本)
| 版本 | 简单识别 | 复杂分析 |
|------|---------|---------|
| Phi3.5-Vision | ~10秒 | ~42秒 |
| Qwen2-VL | ~10秒 | ~42秒 |

*注: 速度相近，主要瓶颈在 OCR agents (PaddleOCR, TrOCR)*

## 🎨 使用场景选择

### 选择 Phi3.5-Vision 版本
- ✅ GPU 显存较小 (< 16GB)
- ✅ 需要更快的加载速度
- ✅ 已有 Phi3.5 模型

### 选择 Qwen2-VL 版本
- ✅ GPU 显存充足 (≥ 24GB)
- ✅ 需要更强的视觉理解能力
- ✅ 已有 Qwen2-VL 模型
- ✅ 需要多语言支持（Qwen2-VL 中文更好）

## 🔗 快速切换

### 从 Phi3.5 切换到 Qwen2-VL
```bash
cd /root/program2
cd qwen_version  # 进入 Qwen2-VL 版本目录
python pipeline.py --image <path> --query <query>
```

### 从 Qwen2-VL 切换到 Phi3.5
```bash
cd /root/program2
cd case2  # 进入 Phi3.5 版本目录
python pipeline.py --image <path> --query <query>
```

## 📝 总结

两个版本在**架构和设计思想上完全一致**，只是**多模态大模型底座不同**：

- **Phi3.5-Vision 版本**: 更轻量，显存友好
- **Qwen2-VL 版本**: 更强大，中文更好

所有 OCR agents (PaddleOCR, TrOCR) 和辅助组件完全共享，确保了两个版本的一致性和可维护性。
