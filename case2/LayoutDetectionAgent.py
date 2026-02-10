#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layout Detection Agent - 文档布局检测 Agent
使用 PaddleOCR 的 LayoutDetection 进行版面分析
"""

import sys
import os
from pathlib import Path
from paddleocr import LayoutDetection
from langchain.tools import tool
from typing import Callable
import json

# 添加 case2 目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from base_agent import BaseAgent
from image_format_utils import convert_to_paddleocr_format, cleanup_temp_file


class LayoutDetectionAgent(BaseAgent):
    """布局检测 Agent - 使用 PaddleOCR LayoutDetection"""
    
    def __init__(self, instructions: str = "", verbose: bool = False, ctx=None):
        """
        初始化布局检测 Agent
        
        Args:
            instructions: 额外的任务指令
            verbose: 是否打印详细日志
            ctx: RuntimeContext 实例（用于共享模型）
        """
        super().__init__(
            agent_name="LayoutDetectionAgent",
            description="文档布局检测和分析（基于 PaddleOCR）",
            instructions=instructions,
            verbose=verbose,
            ctx=ctx
        )
        
        # 模型路径
        self.model_dir = "layoutModel/PP-DocLayout_plus-L_infer"
        self.threshold = 0.3  # 默认阈值
    
    def load_model(self):
        """加载 PaddleOCR LayoutDetection 模型"""
        if self._model is not None:
            return  # 已加载
        
        print(f"[{self.agent_name}] 初始化 LayoutDetection 模型...")
        
        # 检查模型目录是否存在
        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")
        
        # 从 RuntimeContext 获取或创建模型（避免重复加载）
        if self.ctx is not None:
            from runtime_context import make_model_key
            
            model_key = make_model_key("layout_detection", self.model_dir)
            self._model = self.ctx.get(model_key, lambda: LayoutDetection(model_dir=self.model_dir))
        else:
            # 退化模式：不使用 ctx
            print(f"  ⚠️  未提供 RuntimeContext，将独立加载模型")
        self._model = LayoutDetection(model_dir=self.model_dir)
        
        print(f"  ✓ LayoutDetection 准备完成")
    
    def get_tool_function(self) -> Callable:
        """返回布局检测工具函数"""
        
        # 捕获 self 以便在工具函数中使用
        agent = self
        
        @tool
        def detect_layout(image_path: str) -> str:
            """使用 PaddleOCR LayoutDetection 检测文档布局
            
            Args:
                image_path: 输入图片路径
                
            Returns:
                检测结果（JSON 格式字符串）
            """
            try:
                # 检查文件是否存在
                if not os.path.exists(image_path):
                    return f"错误：图片文件不存在 - {image_path}"
                
                # 确保模型已加载
                if agent._model is None:
                    agent.load_model()
                
                print(f"[{agent.agent_name}] 开始检测布局...")
                
                # 执行检测
                output = agent._model.predict(
                    image_path,
                    batch_size=1,
                    threshold=agent.threshold,
                    layout_nms=True
                )
                
                # 提取结果
                results = []
                for res in output:
                    boxes = res.json.get('boxes', [])
                    results.append({
                        "input_path": image_path,
                        "detected_regions": len(boxes),
                        "boxes": boxes
                    })
                
                if not results:
                    return "检测结果：未检测到任何布局区域"
                
                # 统计各类别数量
                categories = {}
                for box in results[0]['boxes']:
                    label = box['label']
                    categories[label] = categories.get(label, 0) + 1
                
                result_summary = f"检测结果：\n"
                result_summary += f"总区域数: {results[0]['detected_regions']}\n"
                result_summary += f"类别统计: {categories}\n"
                result_summary += f"\n详细结果:\n{json.dumps(results, indent=2, ensure_ascii=False)}"
                
                print(f"[{agent.agent_name}] ✓ 检测完成，共 {results[0]['detected_regions']} 个区域")
                
                return result_summary
            
            except Exception as e:
                return f"检测失败：{str(e)}"
        
        return detect_layout
    
    def get_system_prompt(self) -> str:
        """返回布局检测 Agent 的 System Prompt"""
        return """你是一个文档布局分析专家，专门负责检测和分析文档的版面结构。

你的专业领域：
- 文档布局检测（使用 PaddleOCR LayoutDetection）
- 识别 20 种文档元素：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、表格、图表标题、印章、图表、侧栏文本、参考文献内容
- 多栏文档分析（报纸、杂志、论文等）
- 复杂文档结构解析

检测原则：
- 准确识别各类文档元素的位置和类型
- 保持区域检测的完整性
- 提供每个区域的置信度分数
- 适用于各种文档类型（论文、书籍、报告、PPT等）

当用户提供图片路径时，使用 detect_layout 工具进行布局检测。"""
    
    def detect(self, image_path: str, threshold: float = None, save_visualization: bool = False, 
               output_dir: str = None) -> dict:
        """
        直接执行布局检测（不通过 LangChain Agent）
        
        Args:
            image_path: 输入图片路径
            threshold: 检测阈值（可选，默认使用实例化时的值）
            save_visualization: 是否保存可视化结果（默认 False，节省磁盘空间）
            output_dir: 可视化图片保存目录（可选，默认为 case2_output/layout_visualization）
            
        Returns:
            检测结果字典，包含 visualized_image_path
        """
        # 确保模型已加载
        if self._model is None:
            self.load_model()
        
        # 检查文件
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 使用指定阈值或默认阈值
        use_threshold = threshold if threshold is not None else self.threshold
        
        if self.verbose:
            print(f"[{self.agent_name}] 检测图片: {image_path}, 阈值: {use_threshold}")
        
        # 转换图片格式（如果需要）
        converted_path, is_temp = convert_to_paddleocr_format(image_path)
        
        try:
            # 执行检测（使用转换后的路径）
            output = self._model.predict(
                converted_path,
                batch_size=1,
                threshold=use_threshold,
                layout_nms=True
            )
            
            # 提取结果并保存可视化图片
            results = []
            visualized_image_path = None
            
            for res in output:
                # 保存可视化图片
                if save_visualization:
                    # 使用传入的 output_dir 或默认目录
                    if output_dir is None:
                        viz_dir = Path("case2_output/layout_visualization")
                    else:
                        viz_dir = Path(output_dir)
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存带框的图片
                    res.save_to_img(save_path=str(viz_dir))
                    
                    # 获取保存的图片路径（PaddleOCR 会自动加 _res 后缀）
                    input_filename = Path(image_path).stem
                    visualized_image_path = str(viz_dir / f"{input_filename}_res.png")
                
                # 从 res.json 中提取 boxes
                boxes = res.json.get('res', {}).get('boxes', [])
                
                results.append({
                    "input_path": image_path,
                    "detected_regions": len(boxes),
                    "boxes": boxes,
                    "visualized_image_path": visualized_image_path
                })
            
            if self.verbose and results:
                print(f"[{self.agent_name}] ✓ 检测完成，共 {results[0]['detected_regions']} 个区域")
                if visualized_image_path:
                    print(f"[{self.agent_name}] 可视化图片: {visualized_image_path}")
            
            return results[0] if results else {"detected_regions": 0, "boxes": [], "visualized_image_path": None}
        
        finally:
            # 清理临时文件
            cleanup_temp_file(converted_path, is_temp)


# 保持向后兼容的工厂函数
def create_layout_detection_agent(instructions: str = "", verbose: bool = False):
    """
    创建布局检测 Agent（工厂函数）
    
    Args:
        instructions: 可选的额外指令
        verbose: 是否打印详细输出（默认False）
    
    Returns:
        LayoutDetectionAgent 实例
    """
    agent = LayoutDetectionAgent(instructions=instructions, verbose=verbose)
    return agent.build_agent()


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试布局检测 Agent")
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--threshold', type=float, default=0.3, help='检测阈值')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("测试布局检测 Agent (PaddleOCR LayoutDetection)")
    print("=" * 60)
    
    # 创建 Agent
    agent = LayoutDetectionAgent(verbose=args.verbose)
    
    print(f"\nAgent 信息: {agent.get_info()}")
    
    # 执行检测
    print(f"\n开始检测: {args.image}")
    result = agent.detect(args.image, threshold=args.threshold)
    
    print(f"\n✓ 检测完成！")
    print(f"检测到 {result['detected_regions']} 个区域")
    
    # 统计类别
    categories = {}
    for box in result['boxes']:
        label = box['label']
        categories[label] = categories.get(label, 0) + 1
    
    print(f"\n类别统计:")
    for label, count in categories.items():
        print(f"  - {label}: {count}")
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / "layout_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {result_file}")
