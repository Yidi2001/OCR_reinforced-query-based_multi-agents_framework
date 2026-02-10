#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Generator Module
为每个 Agent 生成定制化的 task prompt
"""

import json
from typing import Dict, List
from pathlib import Path


class PromptGenerator:
    """为不同的 Agent 生成定制化的 task prompt"""
    
    def __init__(self):
        """初始化 Prompt Generator"""
        # Agent 模板定义（只保留4个Agent）
        self.agent_templates = {
            "PreprocessingAgent": {
                "description": "图像超分辨率增强（预处理）",
                "base_prompt": "Enhance the image quality using super-resolution techniques."
            },
            "LayoutDetectionAgent": {
                "description": "布局检测和分析",
                "base_prompt": "Detect and analyze the layout of the document, including tables, formulas, and text regions."
            },
            "HandOCRAgent": {
                "description": "手写体文字识别",
                "base_prompt": "Recognize handwritten text from the image with high accuracy."
            },
            "PrintedOCRAgent": {
                "description": "印刷体文字识别",
                "base_prompt": "Recognize printed text from the image with high accuracy."
            }
        }
    
    def generate(self, task_plan: Dict, classification: Dict, query: str) -> Dict:
        """
        生成结构化的执行计划
        
        Args:
            task_plan: Phi3.5 生成的任务计划（只包含预处理需求）
            classification: 目标检测结果（确定手写/印刷）
            query: 用户查询
            
        Returns:
            Structured Execution Plan (dict)
        """
        print(f"\n{'='*60}")
        print("Prompt Generation")
        print(f"{'='*60}")
        
        # 根据 task_plan 和 classification 生成 Agent 链
        agent_chain = self._build_agent_chain(task_plan, classification)
        
        print(f"\nAgent 调用链: {' → '.join(agent_chain)}")
        
        # 为每个 Agent 生成 prompt
        agents_plan = []
        for order, agent_name in enumerate(agent_chain, start=1):
            task_prompt = self._generate_agent_prompt(
                agent_name=agent_name,
                task_plan=task_plan,
                classification=classification,
                query=query,
                order=order
            )
            
            agent_info = {
                "name": agent_name,
                "order": order,
                "description": self.agent_templates.get(agent_name, {}).get("description", ""),
                "task_prompt": task_prompt,
                "input": task_plan.get("image_path"),
                "output": None  # 将在执行时填充
            }
            
            agents_plan.append(agent_info)
            
            print(f"\n{order}. {agent_name}")
            print(f"   Prompt: {task_prompt[:100]}...")
        
        # 构建完整的执行计划
        execution_plan = {
            "image_path": task_plan.get("image_path"),
            "query": query,
            "task_plan": task_plan,
            "classification": classification,
            "execution_plan": {
                "agents": agents_plan,
                "execution_flow": "sequential",  # 顺序执行
                "total_agents": len(agents_plan)
            },
            "metadata": {
                "phi35_reasoning": task_plan.get("reasoning", ""),
                "text_type_verified": classification.get("label", "unknown"),
                "classification_confidence": classification.get("confidence", 0.0)
            }
        }
        
        return execution_plan
    
    def _build_agent_chain(self, task_plan: Dict, classification: Dict) -> List[str]:
        """
        根据任务计划和分类结果构建 Agent 调用链
        
        Args:
            task_plan: Phi3.5 的任务计划（预处理需求）
            classification: 分类模型的结果（手写/印刷）
            
        Returns:
            Agent 名称列表
        """
        chain = []
        
        # 1. 预处理：超分辨率增强
        if task_plan.get("needs_super_resolution", False):
            chain.append("PreprocessingAgent")
        
        # 2. 预处理：布局检测
        if task_plan.get("needs_layout_detection", False):
            chain.append("LayoutDetectionAgent")
        
        # 3. OCR：根据分类模型的结果选择 Agent
        text_type = classification.get("label", "printed")
        if text_type == "hand":
            chain.append("HandOCRAgent")
        else:  # printed
            chain.append("PrintedOCRAgent")
        
        return chain
    
    def _generate_agent_prompt(
        self, 
        agent_name: str, 
        task_plan: Dict, 
        classification: Dict, 
        query: str,
        order: int
    ) -> str:
        """
        为特定 Agent 生成 task prompt
        
        Args:
            agent_name: Agent 名称
            task_plan: 任务计划
            classification: 分类结果
            query: 用户查询
            order: 执行顺序
            
        Returns:
            Task prompt 字符串
        """
        # 获取基础 prompt
        base_prompt = self.agent_templates.get(agent_name, {}).get(
            "base_prompt", 
            f"Execute {agent_name} task."
        )
        
        # 根据不同的 Agent 类型生成定制化 prompt
        if agent_name == "PreprocessingAgent":
            prompt = self._generate_super_resolution_prompt(task_plan, query)
        
        elif agent_name == "LayoutDetectionAgent":
            prompt = self._generate_layout_detection_prompt(task_plan, query)
        
        elif agent_name == "HandOCRAgent":
            prompt = self._generate_hand_ocr_prompt(task_plan, classification, query)
        
        elif agent_name == "PrintedOCRAgent":
            prompt = self._generate_printed_ocr_prompt(task_plan, classification, query)
        
        else:
            prompt = base_prompt
        
        return prompt
    
    def _generate_super_resolution_prompt(self, task_plan: Dict, query: str) -> str:
        """生成超分 Agent 的 prompt"""
        quality = task_plan.get("quality_analysis", {})
        reason = task_plan.get("super_resolution_reason", "提升图片质量")
        
        prompt = f"""This image requires super-resolution enhancement.

Image Quality:
- Resolution: {quality.get('resolution', 'unknown')}
- Sharpness Score: {quality.get('sharpness_score', 0):.1f}/100
- Is Blurry: {quality.get('is_blurry', False)}

Reason: {reason}

Task: Enhance the image quality using Real-ESRGAN to improve OCR accuracy.
Focus on: Sharpening text edges and improving overall clarity.

User Query: {query}
"""
        return prompt.strip()
    
    def _generate_layout_detection_prompt(self, task_plan: Dict, query: str) -> str:
        """生成布局检测 Agent 的 prompt"""
        reason = task_plan.get("layout_detection_reason", "检测文档布局")
        complexity = task_plan.get("text_complexity", "medium")
        
        prompt = f"""This image requires layout detection and analysis.

Text Complexity: {complexity}
Reason: {reason}

Task: Detect and analyze the layout structure of the document.
Focus on: 
- Identifying tables, formulas, charts
- Detecting text regions and reading order
- Segmenting different content types

User Query: {query}
"""
        return prompt.strip()
    
    def _generate_hand_ocr_prompt(
        self, 
        task_plan: Dict, 
        classification: Dict, 
        query: str
    ) -> str:
        """生成手写体 OCR Agent 的 prompt"""
        confidence = classification.get("confidence", 0.0)
        complexity = task_plan.get("text_complexity", "medium")
        
        prompt = f"""Recognize handwritten text from this image.

Classification Confidence: {confidence:.3f}
Text Complexity: {complexity}

Task: Accurately recognize all handwritten text in the image.
Focus on:
- Handling cursive and script writing styles
- Maintaining proper word boundaries
- Preserving text layout and structure
- Handling varying writing quality

User Query: {query}

Please extract all visible text and maintain the original reading order.
"""
        return prompt.strip()
    
    def _generate_printed_ocr_prompt(
        self, 
        task_plan: Dict, 
        classification: Dict, 
        query: str
    ) -> str:
        """生成印刷体 OCR Agent 的 prompt"""
        confidence = classification.get("confidence", 0.0)
        complexity = task_plan.get("text_complexity", "medium")
        
        prompt = f"""Recognize printed text from this image.

Classification Confidence: {confidence:.3f}
Text Complexity: {complexity}

Task: Accurately recognize all printed text in the image.
Focus on:
- Handling various fonts and sizes
- Recognizing special characters and symbols
- Preserving formatting (bold, italic, etc.)
- Maintaining text structure

User Query: {query}

Please extract all visible text and maintain the original reading order.
"""
        return prompt.strip()



def test_prompt_generator():
    """测试 Prompt Generator"""
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python prompt_generator.py <task_plan.json> <classification.json>")
        sys.exit(1)
    
    task_plan_path = sys.argv[1]
    classification_path = sys.argv[2]
    query = sys.argv[3] if len(sys.argv) > 3 else "识别图片中的文字"
    
    # 加载输入
    with open(task_plan_path, 'r', encoding='utf-8') as f:
        task_plan = json.load(f)
    
    with open(classification_path, 'r', encoding='utf-8') as f:
        classification = json.load(f)
    
    # 创建生成器
    generator = PromptGenerator()
    
    # 生成执行计划
    execution_plan = generator.generate(task_plan, classification, query)
    
    print(f"\n{'='*60}")
    print("Structured Execution Plan")
    print(f"{'='*60}")
    print(json.dumps(execution_plan, indent=2, ensure_ascii=False))
    
    # 保存结果
    output_path = Path("execution_plan.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(execution_plan, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到: {output_path}")


if __name__ == "__main__":
    test_prompt_generator()

