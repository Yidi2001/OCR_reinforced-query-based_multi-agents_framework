#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2-VL Refiner (Optimized Version)
基于OCR结果，让Qwen2-VL重新理解和回答用户query
优化版：减少OCR噪声干扰，增强视觉验证，提升准确率
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


class QwenRefinerOptimized:
    """
    Qwen2-VL Refiner (Optimized Version)
    利用OCR识别结果作为上下文，让Qwen2-VL基于用户query给出更准确的回答
    
    优化点：
    1. OCR噪声过滤：只保留top-3最相关区域
    2. Query类型自适应：针对number/title/name等生成特定提示
    3. 强化指令：STEP-BY-STEP + CRITICAL RULES
    4. 强调视觉验证：优先看图，OCR仅作位置提示
    """
    
    def __init__(self, model_path: str = "models/Qwen2-VL-2B-Instruct", ctx: Optional['RuntimeContext'] = None):
        """
        Args:
            model_path: Qwen2-VL模型路径
            ctx: RuntimeContext 实例（用于共享模型，避免重复加载）
        """
        self.model_path = model_path
        self.ctx = ctx
        self.model = None
        self.processor = None
        self.device = None
        self._model_loaded = False
    
    def load_model(self):
        """加载Qwen2-VL模型"""
        if self._model_loaded:
            return
        
        print(f"[QwenRefinerOptimized] 初始化: {self.model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从 RuntimeContext 获取或创建模型（避免重复加载）
        if self.ctx is not None:
            self.model = self._get_model_from_ctx()
            self.processor = self._get_processor_from_ctx()
        else:
            # 退化模式：不使用 ctx（向后兼容）
            print(f"  ⚠️  未提供 RuntimeContext，将独立加载模型")
            self.model = self._load_model()
            self.processor = self._load_processor()
        
        self._model_loaded = True
        print(f"  ✓ QwenRefinerOptimized 准备完成 (device: {self.device})")
    
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
    
    def _clean_ocr_context(self, ocr_summary_text: str, top_k: int = 3) -> str:
        """
        清洗OCR上下文，只保留最相关的top-k个区域，减少噪声干扰
        
        Args:
            ocr_summary_text: 原始OCR摘要文本
            top_k: 保留的区域数量（默认3）
            
        Returns:
            清洗后的OCR文本
        """
        # 如果文本很短（<200字符），直接返回
        if len(ocr_summary_text) < 200:
            return ocr_summary_text
        
        # 解析区域信息（假设格式为 "Region X: ..." 或 "区域X：..."）
        region_pattern = r'(?:Region|区域)\s*\d+[:：]\s*([^\n]+)'
        regions = re.findall(region_pattern, ocr_summary_text, re.IGNORECASE)
        
        if len(regions) <= top_k:
            # 区域数量已经很少，无需过滤
            return ocr_summary_text
        
        # 简单启发式：保留前top_k个区域（假设已按相关性排序）
        # 在实际使用中，这些区域应该已经通过 layout_relevance_selector 排序过
        cleaned_regions = regions[:top_k]
        
        # 重新构建清洗后的文本
        cleaned_text = f"Top {top_k} Most Relevant Regions:\n"
        for i, region_text in enumerate(cleaned_regions, 1):
            cleaned_text += f"Region {i}: {region_text.strip()}\n"
        
        return cleaned_text
    
    def refine_with_ocr_context(self, 
                                 image_path: str, 
                                 user_query: str,
                                 ocr_summary_text: str) -> str:
        """
        基于OCR结果提示，让Qwen2-VL重新进行OCR识别（优化版）
        
        Args:
            image_path: 原始图片路径
            user_query: 用户的查询/任务
            ocr_summary_text: OCR识别结果的摘要文本
            
        Returns:
            Qwen2-VL基于OCR提示和图片给出的识别结果
        """
        self.load_model()
        
        # 步骤1: 清洗OCR上下文（减少噪声）
        cleaned_ocr_text = self._clean_ocr_context(ocr_summary_text, top_k=3)
        
        # 步骤2: 构建优化的提示词
        prompt_text = self._build_refine_prompt(user_query, cleaned_ocr_text)
        
        # 步骤3: 准备图片和对话
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # 步骤4: 应用聊天模板
        text_prompt = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 步骤5: 处理图片和文本
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # 步骤6: 生成回答
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                top_k=50
            )
        
        # 步骤7: 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return response[0].strip()
    
    def direct_inference(self, image_path: str, user_query: str) -> str:
        """
        直接推理模式（不使用OCR上下文）
        适用于简单识别任务
        
        Args:
            image_path: 原始图片路径
            user_query: 用户的查询/任务
            
        Returns:
            Qwen2-VL 直接基于图片给出的识别结果
        """
        self.load_model()
        
        print(f"[QwenRefinerOptimized] 直接推理模式（无 OCR 辅助）...")
        
        # 准备对话
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": user_query},
                ],
            }
        ]
        
        # 应用聊天模板
        text_prompt = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 处理图片和文本
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # 生成回答
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                top_k=50
            )
        
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return response[0].strip()
    
    def _build_refine_prompt(self, user_query: str, ocr_summary_text: str) -> str:
        """
        构建给Qwen2-VL的优化提示词
        
        优化策略：
        1. 识别query类型（number/title/name等）
        2. 生成针对性的task hint
        3. 强化STEP-BY-STEP指令
        4. 添加CRITICAL RULES强调视觉验证
        5. 弱化OCR作为答案来源
        """
        # 分析query类型
        query_lower = user_query.lower()
        task_hint = ""
        
        if 'number' in query_lower or 'digit' in query_lower or '数字' in user_query:
            task_hint = "**TASK**: Extract the NUMBER from the image. VERIFY visually - don't be distracted by other numbers in irrelevant areas."
        elif 'title' in query_lower or '标题' in user_query:
            task_hint = "**TASK**: Find the TITLE in the image. Usually it's the most prominent text at the top. VERIFY by looking at the image."
        elif 'name' in query_lower or '名字' in user_query or 'author' in query_lower:
            task_hint = "**TASK**: Identify the NAME/AUTHOR in the image. VERIFY by looking at the image, not just OCR text."
        else:
            task_hint = "**TASK**: Answer the question by LOOKING AT THE IMAGE carefully."
        
        # 构建优化后的prompt
        prompt = f"""You are a visual question answering assistant with strong visual understanding capabilities.

{task_hint}

**Question**: {user_query}

**STEP-BY-STEP Instructions**:
1. **LOOK AT THE IMAGE FIRST** - Understand the visual content and layout
2. **Identify the target** - Find where the answer is located in the image
3. **Verify visually** - Read the text directly from the image
4. **Cross-check** - Use OCR hints below ONLY to confirm location, NOT as the answer source
5. **Focus on relevance** - IGNORE all text from irrelevant regions (wrong page, different section, etc.)
6. **Extract answer** - Provide the exact text/number you see in the target location
7. **Be concise** - Give the direct answer without extra explanation

**CRITICAL RULES**:
1. The IMAGE is your PRIMARY source - trust what you SEE
2. Use OCR text ONLY as location hints, NOT as the answer
3. IGNORE all OCR text from irrelevant regions and any other text not related to the question
4. If the question asks for a NUMBER, respond with ONLY that number
5. If OCR text contradicts what you see in the image, TRUST THE IMAGE

**Top 3 Most Relevant OCR Regions** (use ONLY as location hints):
{ocr_summary_text}

**Remember**: LOOK AT THE IMAGE first, then verify with OCR hints. Don't let irrelevant OCR text distract you."""

        return prompt
    
    def refine_from_summary_file(self, summary_json_path: str) -> Dict[str, Any]:
        """
        从摘要JSON文件读取信息，并调用Qwen2-VL进行refinement（优化版）
        
        Args:
            summary_json_path: 摘要JSON文件路径
            
        Returns:
            包含refined_response和其他元信息的字典
        """
        # 读取摘要JSON
        with open(summary_json_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # 提取关键信息
        image_path = summary_data.get('image_path')
        user_query = summary_data.get('user_query')
        
        # 提取OCR摘要文本
        summary_sections = summary_data.get('summary', {})
        
        # 尝试多种可能的字段名
        ocr_summary_text = (
            summary_sections.get('relevant_regions') or 
            summary_sections.get('selected_layouts') or
            summary_sections.get('layout_summary') or
            summary_sections.get('ocr_summary') or
            ""
        )
        
        # 如果是字典或列表，转为文本
        if isinstance(ocr_summary_text, dict):
            # 从字典提取文本
            if 'regions' in ocr_summary_text:
                regions = ocr_summary_text['regions']
                if isinstance(regions, list):
                    ocr_summary_text = "\n".join([
                        f"Region {i+1}: {r.get('text', r.get('content', ''))}" 
                        for i, r in enumerate(regions)
                    ])
        elif isinstance(ocr_summary_text, list):
            # 从列表提取文本
            ocr_summary_text = "\n".join([
                f"Region {i+1}: {item.get('text', item.get('content', str(item)))}" 
                for i, item in enumerate(ocr_summary_text)
            ])
        
        # 如果还是找不到OCR文本，尝试从merged_blocks获取
        if not ocr_summary_text and 'merged_blocks' in summary_data:
            blocks = summary_data['merged_blocks']
            if isinstance(blocks, list) and len(blocks) > 0:
                ocr_summary_text = "\n".join([
                    f"Block {i+1}: {block.get('text', '')}" 
                    for i, block in enumerate(blocks[:5])  # 最多取前5个
                ])
        
        # 调用优化版的refinement
        if ocr_summary_text:
            print(f"[QwenRefinerOptimized] 使用 OCR 上下文...")
            print(f"[QwenRefinerOptimized]   图片: {Path(image_path).name}")
            print(f"[QwenRefinerOptimized]   Query: '{user_query}'")
            print(f"[QwenRefinerOptimized]   OCR文本长度: {len(ocr_summary_text)} 字符")
            
            refined_response = self.refine_with_ocr_context(
                image_path=image_path,
                user_query=user_query,
                ocr_summary_text=ocr_summary_text
            )
        else:
            # 降级为直接推理
            print(f"[QwenRefinerOptimized] ⚠️  未找到OCR摘要，使用直接推理...")
            refined_response = self.direct_inference(image_path, user_query)
        
        # 返回结果
        result = {
            'refined_response': refined_response,
            'image_path': image_path,
            'user_query': user_query,
            'ocr_summary_used': bool(ocr_summary_text),
            'summary_file': summary_json_path
        }
        
        return result


def test_refiner():
    """测试Qwen Refiner Optimized"""
    print("=" * 80)
    print("测试 Qwen2-VL Refiner (Optimized Version)")
    print("=" * 80)
    
    # 示例：使用之前生成的摘要文件
    summary_file = "case2_output/example_task/result_summary.json"
    
    if not Path(summary_file).exists():
        print(f"\n❌ 测试文件不存在: {summary_file}")
        print("请先运行 pipeline.py 生成测试数据")
        return
    
    # 创建 refiner（不使用 ctx，独立测试）
    refiner = QwenRefinerOptimized(model_path="models/Qwen2-VL-2B-Instruct")
    
    # 执行refinement
    print("\n正在基于OCR结果调用Qwen2-VL (Optimized)...")
    result = refiner.refine_from_summary_file(summary_file)
    
    # 保存结果
    output_file = "case2_output/example_task/refined_answer_optimized.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 显示结果
    print("\n" + "=" * 80)
    print("Qwen2-VL Refinement 结果 (Optimized)")
    print("=" * 80)
    print(f"\n📷 图像: {result['image_path']}")
    print(f"❓ 用户查询: {result['user_query']}")
    print(f"📄 使用的OCR摘要: {result['ocr_summary_used']}")
    print("\n" + "-" * 80)
    print("💡 Qwen2-VL 的回答 (Optimized):")
    print("-" * 80)
    print(result['refined_response'])
    print("\n" + "=" * 80)
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    test_refiner()
