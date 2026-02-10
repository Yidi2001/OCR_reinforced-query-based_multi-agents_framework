#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi3.5-Vision Refiner
基于OCR结果，让Phi3.5重新理解和回答用户query
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


class PhiRefiner:
    """
    Phi3.5-Vision Refiner
    利用OCR识别结果作为上下文，让Phi3.5基于用户query给出更准确的回答
    """
    
    def __init__(self, model_path: str = "models/phi-3_5_vision", ctx: Optional['RuntimeContext'] = None):
        """
        Args:
            model_path: Phi3.5-Vision模型路径
            ctx: RuntimeContext 实例（用于共享模型，避免重复加载）
        """
        self.model_path = model_path
        self.ctx = ctx
        self.model = None
        self.processor = None
        self.device = None
        self._model_loaded = False
    
    def load_model(self):
        """加载Phi3.5-Vision模型"""
        if self._model_loaded:
            return
        
        print(f"[PhiRefiner] 初始化: {self.model_path}")
        
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
        print(f"  ✓ PhiRefiner 准备完成 (device: {self.device})")
    
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
            _attn_implementation='eager'
        )
        
    def _load_processor(self):
        """加载 Phi processor（工厂函数）"""
        return AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            num_crops=4
        )
    
    def refine_with_ocr_context(self, 
                                 image_path: str, 
                                 user_query: str,
                                 ocr_summary_text: str) -> str:
        """
        基于OCR结果提示，让Phi3.5重新进行OCR识别
        
        Args:
            image_path: 原始图片路径
            user_query: 用户的查询/任务
            ocr_summary_text: OCR识别结果的摘要文本
            
        Returns:
            Phi3.5基于OCR提示和图片给出的识别结果
        """
        self.load_model()
        
        # 构建提示词
        prompt = self._build_refine_prompt(user_query, ocr_summary_text)
        
        # 加载图片
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return f"错误：无法加载图片 - {e}"
        
        # 准备输入
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]
        
        # 应用聊天模板
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理输入
        inputs = self.processor(
            prompt_text,
            [image],
            return_tensors="pt"
        ).to(self.device)
        
        # 生成输出
        print(f"[PhiRefiner] 正在基于OCR结果分析图片...")
        
        generation_args = {
            "max_new_tokens": 2000,
            "temperature": 0.1,
            "do_sample": False,
            "use_cache": False,  # 避免cache相关的兼容性问题
        }
        
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **generation_args
            )
        
        # 移除输入部分，只保留生成的内容
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # 检查响应是否为空
        response_stripped = response.strip()
        if not response_stripped:
            print(f"[PhiRefiner] ⚠️  警告: 模型生成了空响应")
            print(f"[PhiRefiner]   原始响应长度: {len(response)}")
            print(f"[PhiRefiner]   生成的token数: {generate_ids.shape[1]}")
            # 返回一个默认消息而不是空字符串
            return "[模型未能生成有效响应]"
        
        return response_stripped
    
    def _build_refine_prompt(self, user_query: str, ocr_summary_text: str) -> str:
        """
        构建给Phi3.5的提示词 - 让它基于图片和OCR结果回答用户问题
        """
        prompt = f"""You are a visual question answering assistant. You must LOOK AT THE IMAGE and use the OCR reference below to help you answer accurately.

Question: {user_query}

OCR Reference (sorted by relevance to the question):
**IMPORTANT**: The layout regions below are ranked by relevance. The FIRST regions are MOST RELEVANT.

{ocr_summary_text}

Instructions:
1. LOOK AT THE IMAGE carefully - the OCR text is just a reference
2. Use the OCR text to locate relevant regions in the image
3. Verify the answer by examining the actual image content
4. Focus on top-ranked regions first (they are most relevant to the question)
5. Give ONLY the direct answer - be as brief as possible (1-2 sentences max)
6. The answers should not be repetitive and should not contain any repetitive content
7. If asking for a name/number/entity, output only that information
8. Do NOT explain, do NOT repeat the question

Answer:"""
        
        return prompt
    
    def refine_from_summary_file(self, summary_json_path: str) -> Dict[str, Any]:
        """
        从摘要JSON文件读取信息，并调用Phi3.5进行refinement
        
        Args:
            summary_json_path: 摘要JSON文件路径（支持两种格式）
                - 格式1: *_summary.json + 对应的 *_prompt.txt
                - 格式2: 包含 blocks 数组的 JSON（如 evidence.json）
            
        Returns:
            包含refinement结果的字典
        """
        # 读取摘要
        with open(summary_json_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        image_path = summary.get('image_path', '')
        user_query = summary.get('user_query', summary.get('query', '识别图片中的文字'))
        
        # 尝试方法1：查找对应的 prompt.txt 文件
        summary_path = Path(summary_json_path)
        
        # 尝试多种命名方式
        possible_prompt_paths = [
            # 方式1: 在同一文件夹中的 prompt.txt（样本文件夹模式）
            summary_path.parent / 'prompt.txt',
            # 方式2: xxx_summary.json -> xxx_prompt.txt（旧模式）
            summary_path.parent / summary_path.name.replace('_summary.json', '_prompt.txt').replace('.json', '.txt')
        ]
        
        ocr_summary_text = None
        source_info = None
        
        # 尝试找到 prompt 文件
        for prompt_txt_path in possible_prompt_paths:
            if prompt_txt_path.exists() and str(prompt_txt_path) != str(summary_path):
                with open(prompt_txt_path, 'r', encoding='utf-8') as f:
                    ocr_summary_text = f.read()
                source_info = prompt_txt_path.name
                break
        
        # 方法2：从 JSON 的 blocks 中提取文本
        if not ocr_summary_text and 'blocks' in summary:
            ocr_summary_text = self._extract_text_from_blocks(summary['blocks'])
            source_info = f"从 {summary_path.name} 的 blocks 提取"
        
        # 方法3：从 JSON 的 ocr_results 中提取文本
        if not ocr_summary_text and 'ocr_results' in summary:
            ocr_results = summary['ocr_results']
            if isinstance(ocr_results, dict):
                # 格式1: {"type": "whole_image", "text": "..."}
                if 'text' in ocr_results:
                    ocr_summary_text = ocr_results['text']
                    source_info = f"从 {summary_path.name} 的 ocr_results.text 提取"
                # 格式2: {"total_regions": N, "blocks": [...]}
                elif 'blocks' in ocr_results:
                    ocr_summary_text = self._extract_text_from_blocks(ocr_results['blocks'])
                    source_info = f"从 {summary_path.name} 的 ocr_results.blocks 提取"
        
        # 如果所有方法都失败
        if not ocr_summary_text:
            return {
                "error": f"无法获取OCR文本。未找到对应的 prompt.txt，JSON中也没有 blocks 或 ocr_results.text 字段。"
            }
        
        # 检查 OCR 文本质量
        if len(ocr_summary_text.strip()) < 10:
            print(f"[PhiRefiner] ⚠️  警告: OCR文本过短 (长度: {len(ocr_summary_text)})")
            print(f"[PhiRefiner]   OCR文本: '{ocr_summary_text}'")
        
        # 调用Phi3.5
        refined_response = self.refine_with_ocr_context(
            image_path=image_path,
            user_query=user_query,
            ocr_summary_text=ocr_summary_text
        )
        
        return {
            "image_path": image_path,
            "user_query": user_query,
            "refined_response": refined_response,
            "ocr_summary_used": source_info
        }
    
    def _extract_text_from_blocks(self, blocks: list) -> str:
        """
        从 blocks 数组中提取 OCR 文本并格式化
        
        Args:
            blocks: 包含 region_id, text, label, bbox 等信息的列表
            
        Returns:
            格式化的 OCR 摘要文本
        """
        if not blocks:
            return ""
        
        lines = ["【OCR识别结果】\n"]
        
        for i, block in enumerate(blocks, 1):
            region_id = block.get('region_id', i)
            label = block.get('label', '未知类型')
            text = block.get('text', '').strip()
            confidence = block.get('confidence', 0.0)
            bbox = block.get('bbox', [])
            
            lines.append(f"区域 {region_id} ({label}, 置信度: {confidence:.2f}):")
            if text:
                lines.append(f"{text}")
            else:
                lines.append("(无文本)")
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)


def test_refiner():
    """测试Phi Refiner"""
    print("=" * 80)
    print("测试 Phi3.5-Vision Refiner")
    print("=" * 80)
    
    # 测试文件
    summary_file = "evidence.json"
    
    if not Path(summary_file).exists():
        print(f"❌ 测试文件不存在: {summary_file}")
        return
    
    # 创建Refiner
    refiner = PhiRefiner()
    
    # 执行refinement
    print("\n正在基于OCR结果调用Phi3.5...")
    result = refiner.refine_from_summary_file(summary_file)
    
    if "error" in result:
        print(f"❌ 错误: {result['error']}")
        return
    
    # 显示结果
    print("\n" + "=" * 80)
    print("Phi3.5 Refinement 结果")
    print("=" * 80)
    print(f"\n📷 图像: {result['image_path']}")
    print(f"❓ 用户查询: {result['user_query']}")
    print(f"📄 使用的OCR摘要: {result['ocr_summary_used']}")
    print("\n" + "-" * 80)
    print("💡 Phi3.5 的回答:")
    print("-" * 80)
    print(result['refined_response'])
    print("\n" + "=" * 80)
    
    # 保存结果
    output_file = "evidence_refined.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结果已保存到: {output_file}")


if __name__ == "__main__":
    test_refiner()

