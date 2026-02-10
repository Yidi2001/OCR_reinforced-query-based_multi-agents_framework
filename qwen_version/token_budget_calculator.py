#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token Budget Calculator for Phi-3.5-Vision
动态计算图像和文本的 token 预算
"""

from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any


class TokenBudgetCalculator:
    """
    根据输入图像动态计算可用的文本 token 预算
    """
    
    # Phi-3.5-Vision 的配置参数
    MAX_CONTEXT_LENGTH = 131072  # 128K tokens
    BASE_IMG_TOKENS = 144  # 每个图像块的基础 token 数
    NUM_CROPS = 4  # 默认的图像分块数
    SYSTEM_PROMPT_RESERVE = 500  # 为系统 prompt 预留的 token
    SAFETY_MARGIN = 200  # 安全边距
    
    def __init__(self, num_crops: int = 4):
        """
        Args:
            num_crops: 图像分块数量（与 processor 中的 num_crops 对应）
        """
        self.num_crops = num_crops
    
    def estimate_image_tokens(self, image_path: str) -> int:
        """
        估算图像占用的 token 数量
        
        基于 Phi-3.5-Vision 的图像处理机制：
        - 基础图像：144 tokens
        - 每个额外的 crop：144 tokens
        
        Args:
            image_path: 图像路径
            
        Returns:
            估算的图像 token 数量
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # 基础图像 tokens
            base_tokens = self.BASE_IMG_TOKENS
            
            # 根据图像分辨率和 num_crops 估算额外的 tokens
            # Phi-3.5-Vision 会根据图像大小动态决定使用多少 crops
            # 高分辨率图像会使用更多 crops
            aspect_ratio = max(width, height) / min(width, height)
            
            # 如果图像很大或宽高比很极端，会使用更多 crops
            if width > 1024 or height > 1024 or aspect_ratio > 2.0:
                # 使用全部 crops
                actual_crops = self.num_crops
            elif width > 512 or height > 512:
                # 使用部分 crops
                actual_crops = self.num_crops // 2
            else:
                # 小图像可能不需要额外 crops
                actual_crops = 0
            
            # 总 token 数 = 基础 + crops
            total_img_tokens = base_tokens * (1 + actual_crops)
            
            return total_img_tokens
            
        except Exception as e:
            # 如果无法读取图像，使用保守估计
            print(f"⚠️ 无法读取图像 {image_path}，使用最大估计值")
            return self.BASE_IMG_TOKENS * (1 + self.num_crops)
    
    def calculate_text_budget(self, image_path: str, 
                             max_output_tokens: int = 2000) -> Dict[str, int]:
        """
        计算可用的文本输入 token 预算
        
        Args:
            image_path: 图像路径
            max_output_tokens: 预期的最大输出 token 数
            
        Returns:
            包含各部分 token 分配的字典
        """
        # 1. 估算图像 tokens
        image_tokens = self.estimate_image_tokens(image_path)
        
        # 2. 计算已使用的 tokens
        used_tokens = (
            image_tokens +  # 图像
            self.SYSTEM_PROMPT_RESERVE +  # 系统 prompt
            max_output_tokens +  # 输出预留
            self.SAFETY_MARGIN  # 安全边距
        )
        
        # 3. 计算剩余可用于 OCR 文本的 tokens
        available_text_tokens = self.MAX_CONTEXT_LENGTH - used_tokens
        
        # 4. 确保不为负数
        if available_text_tokens < 0:
            print(f"⚠️ Token 预算不足！需要减少输出或使用更小的图像")
            available_text_tokens = 1000  # 最小预算
        
        return {
            "max_context": self.MAX_CONTEXT_LENGTH,
            "image_tokens": image_tokens,
            "system_prompt_reserve": self.SYSTEM_PROMPT_RESERVE,
            "output_reserve": max_output_tokens,
            "safety_margin": self.SAFETY_MARGIN,
            "used_tokens": used_tokens,
            "available_text_tokens": available_text_tokens,
        }
    
    def get_text_budget(self, image_path: str, 
                       max_output_tokens: int = 2000) -> int:
        """
        简化版：直接返回可用的文本 token 数量
        
        Args:
            image_path: 图像路径
            max_output_tokens: 预期的最大输出 token 数
            
        Returns:
            可用的文本 token 数量
        """
        budget = self.calculate_text_budget(image_path, max_output_tokens)
        return budget["available_text_tokens"]
    
    def print_budget_info(self, image_path: str, max_output_tokens: int = 2000):
        """
        打印详细的 token 预算信息
        """
        budget = self.calculate_text_budget(image_path, max_output_tokens)
        
        print("\n" + "=" * 80)
        print("📊 Token 预算分析")
        print("=" * 80)
        print(f"图像路径: {image_path}")
        
        # 读取图像尺寸
        try:
            img = Image.open(image_path)
            print(f"图像尺寸: {img.size[0]}x{img.size[1]}")
        except:
            pass
        
        print("\n" + "-" * 80)
        print(f"模型最大上下文: {budget['max_context']:,} tokens")
        print(f"  - 图像占用:     {budget['image_tokens']:,} tokens ({budget['image_tokens']/budget['max_context']*100:.1f}%)")
        print(f"  - 系统 Prompt:  {budget['system_prompt_reserve']:,} tokens ({budget['system_prompt_reserve']/budget['max_context']*100:.1f}%)")
        print(f"  - 输出预留:     {budget['output_reserve']:,} tokens ({budget['output_reserve']/budget['max_context']*100:.1f}%)")
        print(f"  - 安全边距:     {budget['safety_margin']:,} tokens ({budget['safety_margin']/budget['max_context']*100:.1f}%)")
        print("-" * 80)
        print(f"已使用 tokens:   {budget['used_tokens']:,} tokens ({budget['used_tokens']/budget['max_context']*100:.1f}%)")
        print(f"✓ 可用文本预算:  {budget['available_text_tokens']:,} tokens ({budget['available_text_tokens']/budget['max_context']*100:.1f}%)")
        print("=" * 80 + "\n")


def test_calculator():
    """测试 token 预算计算器"""
    calculator = TokenBudgetCalculator(num_crops=4)
    
    # 测试图像
    test_image = "OCRBench_Images/docVQA/val/documents/flpp0227_16.png"
    
    if Path(test_image).exists():
        # 打印详细信息
        calculator.print_budget_info(test_image, max_output_tokens=2000)
        
        # 获取简单的预算值
        text_budget = calculator.get_text_budget(test_image)
        print(f"推荐的文本 token 预算: {text_budget}")
        
        # 转换为字符数（粗略估计：1 token ≈ 4 字符）
        approx_chars = text_budget * 4
        print(f"大约可以输入: {approx_chars:,} 个字符的 OCR 文本")
    else:
        print(f"测试图像不存在: {test_image}")


if __name__ == "__main__":
    test_calculator()

