#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand OCR Agent - 手写体识别 Agent
使用 TrOCR 模型 + LangChain 封装
"""

import sys
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from langchain.tools import tool
import os
import torch
from typing import Callable

# 添加 case2 目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from base_agent import BaseAgent


class HandOCRAgent(BaseAgent):
    """手写体识别 Agent - 使用 TrOCR 模型"""
    
    def __init__(self, instructions: str = "", verbose: bool = False, ctx=None):
        """
        初始化手写体识别 Agent
        
        Args:
            instructions: 额外的任务指令
            verbose: 是否打印详细日志
            ctx: RuntimeContext 实例（用于共享模型）
        """
        super().__init__(
            agent_name="HandOCRAgent",
            description="手写体文字识别（基于 TrOCR）",
            instructions=instructions,
            verbose=verbose,
            ctx=ctx
        )
        
        # TrOCR 模型组件
        self._processor = None
        self._device = None
    
    def load_model(self):
        """加载 TrOCR 模型"""
        if self._model is not None:
            return  # 已加载
        
        print(f"[{self.agent_name}] 初始化 TrOCR 手写体识别模型...")
        
        # 设置设备
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从 RuntimeContext 获取或创建模型（避免重复加载）
        if self.ctx is not None:
            from runtime_context import make_ocr_key
            
            processor_key = make_ocr_key("trocr_processor", "trocr-base-handwritten")
            self._processor = self.ctx.get(
                processor_key,
                lambda: TrOCRProcessor.from_pretrained('trocr-base-handwritten')
            )
            
            model_key = make_ocr_key("trocr_model", "trocr-base-handwritten")
            self._model = self.ctx.get(
                model_key,
                lambda: self._load_trocr_model()
            )
        else:
            # 退化模式：不使用 ctx
            print(f"  ⚠️  未提供 RuntimeContext，将独立加载模型")
            self._processor = TrOCRProcessor.from_pretrained('trocr-base-handwritten')
            self._model = self._load_trocr_model()
        
        print(f"  ✓ TrOCR 准备完成 (device: {self._device})")
    
    def _load_trocr_model(self):
        """加载 TrOCR 模型（工厂函数）"""
        model = VisionEncoderDecoderModel.from_pretrained('trocr-base-handwritten')
        model.to(self._device)
        model.eval()
        return model
    
    def get_tool_function(self) -> Callable:
        """返回手写体识别工具函数"""
        
        # 捕获 self 以便在工具函数中使用
        agent = self
        
        @tool
        def recognize_handwritten_image(image_path: str) -> str:
            """使用 TrOCR 识别手写体图片中的文字内容
            
            Args:
                image_path: 图片文件路径
                
            Returns:
                识别出的手写文字内容
            """
            try:
                # 检查文件是否存在
                if not os.path.exists(image_path):
                    return f"错误：图片文件不存在 - {image_path}"
                
                # 确保模型已加载
                if agent._model is None:
                    agent.load_model()
                
                # 加载图片
                image = Image.open(image_path).convert("RGB")
                
                # 预处理图片
                pixel_values = agent._processor(images=image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(agent._device)
                
                # 生成识别结果
                with torch.no_grad():
                    generated_ids = agent._model.generate(pixel_values)
                
                # 解码结果
                generated_text = agent._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if not generated_text.strip():
                    return "识别结果：未检测到任何手写文字"
                
                return f"识别结果：\n{generated_text}"
            
            except Exception as e:
                return f"识别失败：{str(e)}"
        
        return recognize_handwritten_image
    
    def get_system_prompt(self) -> str:
        """返回手写体识别的 System Prompt"""
        return """你是一个专门识别手写文字的 OCR 专家。

你的专业领域：
- 手写英文文字识别
- 处理各种手写风格和笔迹
- 识别潦草、连笔的文字
- 处理不规则的文字排列

识别原则：
- 逐行识别手写内容
- 保留原始格式（bullet points、破折号等）
- 对于模糊或难以辨认的字符，标记为 [unreadable]
- 只展开明确无歧义的缩写

**重要输出规则：**
- 只输出识别到的原始文字内容
- 不要翻译成其他语言
- 不要添加解释、说明或总结
- 不要添加额外的格式化标记
- 工具返回什么就直接输出什么

当用户提供图片路径时，使用 recognize_handwritten_image 工具进行识别。"""


# 保持向后兼容的工厂函数
def create_hand_ocr_agent(instructions: str = "", verbose: bool = False):
    """
    创建手写体识别专用 Agent（工厂函数）
    
    Args:
        instructions: 可选的额外指令
        verbose: 是否打印详细输出（默认False）
    
    Returns:
        HandOCRAgent 实例
    """
    agent = HandOCRAgent(instructions=instructions, verbose=verbose)
    return agent.build_agent()


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试手写体识别 Agent (TrOCR)")
    print("=" * 60)
    
    # 方式 1: 使用工厂函数（向后兼容）
    print("\n方式 1: 使用工厂函数")
    hand_agent_v1 = create_hand_ocr_agent(verbose=True)
    
    # 方式 2: 直接使用类（推荐）
    print("\n方式 2: 直接使用类")
    hand_agent_v2 = HandOCRAgent(
        instructions="Focus on cursive handwriting",
        verbose=True
    )
    
    print(f"\nAgent 信息: {hand_agent_v2.get_info()}")
    
    # 测试图片路径
    test_image = "OCRBench_Images/IC15_1811/imgs/000000229.jpg"
    
    if os.path.exists(test_image):
        result = hand_agent_v2.invoke({
            "input": f"请识别这张手写图片：{test_image}"
        })
        print("\n手写体识别结果：")
        print(result['output'])
    else:
        print(f"\n测试图片不存在: {test_image}")
        print("请提供有效的图片路径进行测试")
