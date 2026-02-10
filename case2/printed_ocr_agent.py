#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Printed OCR Agent - 印刷体识别 Agent
使用 PaddleOCR + LangChain 封装
"""

import sys
from pathlib import Path
from paddleocr import PaddleOCR
from langchain.tools import tool
import os
from PIL import Image
import tempfile
from typing import Callable

# 添加 case2 目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from base_agent import BaseAgent
from image_format_utils import convert_to_paddleocr_format, cleanup_temp_file


class PrintedOCRAgent(BaseAgent):
    """印刷体识别 Agent - 使用 PaddleOCR"""
    
    def __init__(self, instructions: str = "", verbose: bool = False, ctx=None):
        """
        初始化印刷体识别 Agent
        
        Args:
            instructions: 额外的任务指令
            verbose: 是否打印详细日志
            ctx: RuntimeContext 实例（用于共享模型）
        """
        super().__init__(
            agent_name="PrintedOCRAgent",
            description="印刷体文字识别（基于 PaddleOCR）",
            instructions=instructions,
            verbose=verbose,
            ctx=ctx
        )
    
    def load_model(self):
        """加载 PaddleOCR 模型"""
        if self._model is not None:
            return  # 已加载
        
        print(f"[{self.agent_name}] 初始化 PaddleOCR 印刷体识别模型...")
        
        # 从 RuntimeContext 获取或创建模型（避免重复加载）
        if self.ctx is not None:
            from runtime_context import make_ocr_key
            
            # 配置参数作为 key 的一部分
            config_str = "en_limit20_min_box0.3_thresh0.2"
            model_key = make_ocr_key("paddleocr", config_str)
            self._model = self.ctx.get(model_key, lambda: self._create_paddleocr())
        else:
            # 退化模式：不使用 ctx
            print(f"  ⚠️  未提供 RuntimeContext，将独立加载模型")
            self._model = self._create_paddleocr()
        
        print(f"  ✓ PaddleOCR 准备完成")
    
    def _create_paddleocr(self):
        """创建 PaddleOCR 实例（工厂函数）"""
        return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='en',
            # 针对小图片优化的检测参数
            text_det_limit_side_len=20,
            text_det_limit_type='min',
            text_det_box_thresh=0.3,
            text_det_thresh=0.2
        )
    
    def get_tool_function(self) -> Callable:
        """返回印刷体识别工具函数"""
        
        # 捕获 self 以便在工具函数中使用
        agent = self
        
        @tool
        def recognize_printed_image(image_path: str) -> str:
            """使用 PaddleOCR 识别印刷体图片中的文字内容
            
            Args:
                image_path: 图片文件路径
                
            Returns:
                识别出的印刷体文字内容
            """
            try:
                # 检查文件是否存在
                if not os.path.exists(image_path):
                    return f"错误：图片文件不存在 - {image_path}"
                
                # 确保模型已加载
                if agent._model is None:
                    agent.load_model()
                
                # 读取图片
                img = Image.open(image_path)
                
                # 如果图片太小（任一边<100px），先放大4倍
                min_side = min(img.size)
                if min_side < 100:
                    scale_factor = max(4, int(100 / min_side) + 1)
                    new_size = (img.size[0] * scale_factor, img.size[1] * scale_factor)
                    img_resized = img.resize(new_size, Image.LANCZOS)
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        img_resized.save(tmp.name)
                        temp_path = tmp.name
                    
                    use_path = temp_path
                else:
                    use_path = image_path
                
                # 转换图片格式（如果需要）
                converted_path, is_temp_converted = convert_to_paddleocr_format(use_path)
                
                try:
                # 执行识别
                    result = agent._model.predict(converted_path)
                finally:
                    # 清理格式转换的临时文件
                    cleanup_temp_file(converted_path, is_temp_converted)
                
                    # 清理缩放的临时文件
                if use_path != image_path:
                    try:
                        os.unlink(use_path)
                    except:
                        pass
                
                # 提取所有识别的文字
                all_texts = []
                all_scores = []
                
                for res in result:
                    if isinstance(res, dict):
                        rec_texts = res.get('rec_texts', [])
                        rec_scores = res.get('rec_scores', [])
                        all_texts.extend(rec_texts)
                        all_scores.extend(rec_scores)
                
                if not all_texts:
                    return "识别结果：未检测到任何印刷体文字"
                
                # 格式化输出：文字 + 置信度
                output_lines = []
                for text, score in zip(all_texts, all_scores):
                    output_lines.append(f"{text} (置信度: {score:.2f})")
                
                return "识别结果：\n" + "\n".join(output_lines)
            
            except Exception as e:
                return f"识别失败：{str(e)}"
        
        return recognize_printed_image
    
    def get_system_prompt(self) -> str:
        """返回印刷体识别的 System Prompt"""
        return """你是一个专门识别印刷体文字的 OCR 专家。

你的专业领域：
- 印刷体英文文字识别
- 处理各种字体和排版
- 识别书籍、文档、标签上的文字
- 处理表格和多栏布局

识别原则：
- 按照自然阅读顺序转录文字
- 保持原始换行和段落
- 保留标点符号和大小写
- 不添加原文中不存在的内容
- 对于无法识别的文字，标记为 [unreadable]

**重要输出规则：**
- 只输出识别到的原始文字内容
- 不要翻译成其他语言
- 不要添加解释、说明或总结
- 不要添加额外的格式化标记
- 工具返回什么就直接输出什么

当用户提供图片路径时，使用 recognize_printed_image 工具进行识别。"""


# 保持向后兼容的工厂函数
def create_printed_ocr_agent(instructions: str = "", verbose: bool = False):
    """
    创建印刷体识别专用 Agent（工厂函数）
    
    Args:
        instructions: 可选的额外指令
        verbose: 是否打印详细输出（默认False）
    
    Returns:
        PrintedOCRAgent 实例
    """
    agent = PrintedOCRAgent(instructions=instructions, verbose=verbose)
    return agent.build_agent()


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试印刷体识别 Agent (PaddleOCR)")
    print("=" * 60)
    
    # 方式 1: 使用工厂函数（向后兼容）
    print("\n方式 1: 使用工厂函数")
    printed_agent_v1 = create_printed_ocr_agent(verbose=True)
    
    # 方式 2: 直接使用类（推荐）
    print("\n方式 2: 直接使用类")
    printed_agent_v2 = PrintedOCRAgent(
        instructions="Focus on table text extraction",
        verbose=True
    )
    
    print(f"\nAgent 信息: {printed_agent_v2.get_info()}")
    
    # 测试图片路径（需要替换为实际的印刷体图片）
    test_image = "test_images/printed_sample.jpg"
    
    if os.path.exists(test_image):
        result = printed_agent_v2.invoke({
            "input": f"请识别这张印刷体图片：{test_image}"
        })
        print("\n印刷体识别结果：")
        print(result['output'])
    else:
        print(f"\n测试图片不存在: {test_image}")
        print("请提供有效的印刷体图片路径进行测试")

