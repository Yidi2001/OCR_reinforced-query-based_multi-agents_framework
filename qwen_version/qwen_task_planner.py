#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2-VL Task Planner
作为"大脑"分析图片和query，生成任务执行计划
"""

import json
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen2TaskPlanner:
    """使用 Qwen2-VL 作为任务规划器"""
    
    def __init__(self, model_path: str = "models/Qwen2-VL-2B-Instruct", ctx: Optional['RuntimeContext'] = None):
        """
        初始化 Task Planner
        
        Args:
            model_path: Qwen2-VL 模型路径
            ctx: RuntimeContext 实例（用于共享模型，避免重复加载）
        """
        self.model_path = model_path
        self.ctx = ctx
        
        print(f"[Qwen2TaskPlanner] 初始化: {model_path}")
        
        # 从 RuntimeContext 获取或创建模型（避免重复加载）
        if self.ctx is not None:
            self.model = self._get_model_from_ctx()
            self.processor = self._get_processor_from_ctx()
        else:
            # 退化模式：不使用 ctx（向后兼容）
            print(f"  ⚠️  未提供 RuntimeContext，将独立加载模型")
            self.model = self._load_model()
            self.processor = self._load_processor()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("  ✓ Task Planner 准备完成")
    
    def _get_model_from_ctx(self):
        """从 RuntimeContext 获取 Qwen2 模型"""
        from runtime_context import make_model_key
        key = make_model_key("qwen2_model", self.model_path)
        return self.ctx.get(key, self._load_model)
    
    def _get_processor_from_ctx(self):
        """从 RuntimeContext 获取 Qwen2 processor"""
        from runtime_context import make_model_key
        key = make_model_key("qwen2_processor", self.model_path)
        return self.ctx.get(key, self._load_processor)
    
    def _load_model(self):
        """加载 Qwen2 模型（工厂函数）"""
        return Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def _load_processor(self):
        """加载 Qwen2 processor（工厂函数）"""
        return AutoProcessor.from_pretrained(self.model_path)
    
    def _analyze_image_quality(self, image: Image.Image) -> Dict:
        """
        分析图片质量
        
        Args:
            image: PIL Image
            
        Returns:
            质量分析结果
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # 转灰度图
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 计算清晰度（拉普拉斯方差）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())
        
        # 判断是否模糊
        is_blurry = laplacian_var < 100
        
        # 判断分辨率是否足够
        is_low_resolution = width < 1024 or height < 1024
        
        return {
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "laplacian_var": laplacian_var,
            "is_blurry": is_blurry,
            "is_low_resolution": is_low_resolution,
            "sharpness_score": min(100, laplacian_var / 5)
        }
    
    def classify_query_type(self, query: str, image_path: str) -> str:
        """
        判断问题类型：简单识别 vs 复杂分析
        纯文本判断，不依赖图片内容
        
        Args:
            query: 用户查询
            image_path: 图片路径（未使用，保留接口一致性）
            
        Returns:
            "simple" - 简单识别问题（直接识别图中内容）
            "complex" - 复杂分析问题（需要OCR辅助定位细节）
        """
        # 使用规则判断，不依赖模型
        query_lower = query.lower().strip()
        
        # 严格定义：只有这两种问题是简单识别
        simple_patterns = [
            "what is written in the image?",
            "what is the number in the image?",
            "recognize all text in the image",
            "read all text in the image"
        ]
        
        # 完全匹配才算简单
        for pattern in simple_patterns:
            if query_lower == pattern or query_lower == pattern.rstrip('?'):
                print(f"[TaskPlanner] 问题类型: simple (匹配: {pattern})")
                return "simple"
        
        # 其他所有情况都是复杂分析
        print(f"[TaskPlanner] 问题类型: complex (包含具体信息需求)")
        return "complex"
    
    def _build_planning_prompt(self, query: str, quality_info: Dict, image_path: str) -> List[Dict]:
        """
        构建任务规划的 conversation（Qwen2-VL 格式）
        
        Args:
            query: 用户查询
            quality_info: 图片质量信息
            image_path: 图片路径
            
        Returns:
            conversation 列表
        """
        text_prompt = f"""你是一个 OCR 预处理规划专家。请分析这张图片的预处理需求：

**用户查询**: {query}

**图片基本信息**:
- 分辨率: {quality_info['resolution']}
- 清晰度分数: {quality_info['sharpness_score']:.1f}/100
- 是否模糊: {'是' if quality_info['is_blurry'] else '否'}

**请分析并回答**:
1. 图片是否需要超分辨率增强？(考虑模糊程度和分辨率)
2. 图片中的文字布局是否复杂？是否需要布局检测？(表格、公式、多栏等)

注意：文字类型（手写体/印刷体）将由专门的分类模型判断，你只需要关注预处理需求。

**请严格按以下 JSON 格式输出**:
```json
{{
  "needs_super_resolution": true/false,
  "super_resolution_reason": "原因说明",
  "needs_layout_detection": true/false,
  "layout_detection_reason": "原因说明",
  "text_complexity": "simple/medium/complex",
  "reasoning": "整体推理过程"
}}
```"""
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        return conversation
    
    def plan(self, image_path: str, query: str) -> Dict:
        """
        生成任务执行计划
        
        Args:
            image_path: 图片路径
            query: 用户查询
            
        Returns:
            Task Plan (dict)
        """
        # 步骤1: 判断问题类型
        query_type = self.classify_query_type(query, image_path)
        print(f"[TaskPlanner] 问题类型: {query_type}")
        
        # 如果是简单识别问题，返回特殊标记，跳过 OCR pipeline
        if query_type == "simple":
            return {
                "query_type": "simple_recognition",
                "image_path": str(image_path),
                "query": query,
                "skip_agents": True,
                "reasoning": "This is a simple recognition task that doesn't require OCR agents"
            }
        
        # 步骤2: 对于复杂问题，继续正常的任务规划
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 分析图片质量
        quality_info = self._analyze_image_quality(image)
        
        # 构建 conversation
        conversation = self._build_planning_prompt(query, quality_info, image_path)
        
        # 生成规划
        text_prompt = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(conversation)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
        }
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # 只保留生成的部分
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        # 解析输出
        plan = self._parse_planning_output(output, quality_info)
        plan["query_type"] = "complex_analysis"
        plan["image_path"] = str(image_path)
        plan["query"] = query
        plan["raw_output"] = output
        
        return plan
    
    def _parse_planning_output(self, output: str, quality_info: Dict) -> Dict:
        """
        解析 Qwen2-VL 的输出
        
        Args:
            output: 模型输出
            quality_info: 质量信息
            
        Returns:
            解析后的 Task Plan
        """
        # 尝试提取 JSON
        try:
            # 查找 JSON 块
            if "```json" in output:
                start = output.find("```json") + 7
                end = output.find("```", start)
                json_str = output[start:end].strip()
            elif "{" in output and "}" in output:
                start = output.find("{")
                end = output.rfind("}") + 1
                json_str = output[start:end]
            else:
                json_str = "{}"
            
            plan_data = json.loads(json_str)
            
        except json.JSONDecodeError:
            # 如果解析失败，使用默认值
            print(f"⚠️ JSON解析失败，使用默认规划")
            plan_data = {
                "needs_super_resolution": quality_info["is_blurry"] or quality_info["is_low_resolution"],
                "super_resolution_reason": "图片模糊或分辨率较低",
                "needs_layout_detection": False,
                "layout_detection_reason": "无法判断",
                "text_complexity": "medium",
                "reasoning": "模型输出解析失败，使用默认规划"
            }
        
        # 添加质量信息
        plan_data["quality_analysis"] = quality_info
        
        return plan_data


def test_task_planner():
    """测试 Task Planner"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python qwen_task_planner.py <image_path> [query]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "识别图片中的文字"
    
    # 创建 planner
    planner = Qwen2TaskPlanner()
    
    # 生成计划
    print(f"\n{'='*60}")
    print("Task Planning")
    print(f"{'='*60}")
    print(f"图片: {image_path}")
    print(f"查询: {query}")
    
    plan = planner.plan(image_path, query)
    
    print(f"\n{'='*60}")
    print("Task Plan 结果")
    print(f"{'='*60}")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    
    # 保存结果
    output_path = Path("task_plan_output_qwen.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到: {output_path}")


if __name__ == "__main__":
    test_task_planner()
