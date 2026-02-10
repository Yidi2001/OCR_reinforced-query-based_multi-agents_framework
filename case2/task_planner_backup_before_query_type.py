#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi3.5-Vision Task Planner
作为"大脑"分析图片和query，生成任务执行计划
"""

import json
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor


class Phi35TaskPlanner:
    """使用 Phi3.5-Vision 作为任务规划器"""
    
    def __init__(self, model_path: str = "models/phi-3_5_vision", ctx: Optional['RuntimeContext'] = None):
        """
        初始化 Task Planner
        
        Args:
            model_path: Phi3.5-Vision 模型路径
            ctx: RuntimeContext 实例（用于共享模型，避免重复加载）
        """
        self.model_path = model_path
        self.ctx = ctx
        
        print(f"[Phi35TaskPlanner] 初始化: {model_path}")
        
        # 从 RuntimeContext 获取或创建模型（避免重复加载）
        if self.ctx is not None:
            self.model = self._get_model_from_ctx()
            self.processor = self._get_processor_from_ctx()
        else:
            # 退化模式：不使用 ctx（向后兼容）
            print(f"  ⚠️  未提供 RuntimeContext，将独立加载模型")
            self.model = self._load_model()
            self.processor = self._load_processor()
        
        # 配置 tokenizer
        tok = self.processor.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.model.config.pad_token_id = tok.pad_token_id
        self.model.config.use_cache = False
        
        print("  ✓ Task Planner 准备完成")
    
    def _get_model_from_ctx(self):
        """从 RuntimeContext 获取 Phi 模型"""
        from runtime_context import make_model_key
        key = make_model_key("phi_model", self.model_path)
        return self.ctx.get(key, self._load_model)
    
    def _get_processor_from_ctx(self):
        """从 RuntimeContext 获取 Phi processor"""
        from runtime_context import make_model_key
        key = make_model_key("phi_processor", self.model_path)
        return self.ctx.get(key, self._load_processor)
    
    def _load_model(self):
        """加载 Phi 模型（工厂函数）"""
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            torch_dtype="auto",
            local_files_only=True,
            _attn_implementation="eager",
        )
    
    def _load_processor(self):
        """加载 Phi processor（工厂函数）"""
        return AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            num_crops=1,
        )
    
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
    
    def _build_planning_prompt(self, query: str, quality_info: Dict) -> str:
        """
        构建任务规划的 prompt
        
        Args:
            query: 用户查询
            quality_info: 图片质量信息
            
        Returns:
            prompt 字符串
        """
        prompt = f"""<|user|>
<|image_1|>
你是一个 OCR 预处理规划专家。请分析这张图片的预处理需求：

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
```<|end|>
<|assistant|>
"""
        return prompt
    
    def plan(self, image_path: str, query: str) -> Dict:
        """
        生成任务执行计划
        
        Args:
            image_path: 图片路径
            query: 用户查询
            
        Returns:
            Task Plan (dict)
        """
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 分析图片质量
        quality_info = self._analyze_image_quality(image)
        
        # 构建 prompt
        prompt = self._build_planning_prompt(query, quality_info)
        
        # 生成规划
        inputs = self.processor(prompt, [image], return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.0,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "use_cache": False,
        }
        
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kwargs)
        
        out_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        output = self.processor.batch_decode(
            out_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 解析输出
        plan = self._parse_planning_output(output, quality_info)
        plan["image_path"] = str(image_path)
        plan["query"] = query
        plan["raw_output"] = output
        
        return plan
    
    def _parse_planning_output(self, output: str, quality_info: Dict) -> Dict:
        """
        解析 Phi3.5 的输出
        
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
        print("用法: python task_planner.py <image_path> [query]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "识别图片中的文字"
    
    # 创建 planner
    planner = Phi35TaskPlanner()
    
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
    output_path = Path("task_plan_output.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到: {output_path}")


if __name__ == "__main__":
    test_task_planner()

