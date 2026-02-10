#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Agent - 图像预处理 Agent
使用 Real-ESRGAN 进行超分辨率增强
"""

import sys
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from langchain.tools import tool
from typing import Callable

# 添加 case2 目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from base_agent import BaseAgent


class PreprocessingAgent(BaseAgent):
    """预处理 Agent - 使用 Real-ESRGAN 进行超分辨率增强"""
    
    def __init__(self, instructions: str = "", verbose: bool = False, ctx=None):
        """
        初始化预处理 Agent
        
        Args:
            instructions: 额外的任务指令
            verbose: 是否打印详细日志
            ctx: RuntimeContext 实例（用于共享 LLM client）
        """
        super().__init__(
            agent_name="PreprocessingAgent",
            description="图像超分辨率增强（基于 Real-ESRGAN）",
            instructions=instructions,
            verbose=verbose,
            ctx=ctx
        )
        
        # Real-ESRGAN 配置
        self.realesrgan_dir = Path(__file__).parent.parent /"chaofen" / "Real-ESRGAN"
        self.output_dir = self.realesrgan_dir / "results"
        self.model_name = "RealESRGAN_x4plus_anime_6B"
        self.conda_env = "chaofen"
    
    def load_model(self):
        """
        加载 Real-ESRGAN 模型（验证环境）
        
        注意：Real-ESRGAN 是通过 subprocess 调用的，不需要在 Python 中加载
        这里主要是验证环境和路径是否正确
        """
        if self._model is not None:
            return  # 已验证
        
        print(f"[{self.agent_name}] 验证 Real-ESRGAN 环境...")
        
        # 检查 Real-ESRGAN 目录
        if not self.realesrgan_dir.exists():
            raise FileNotFoundError(f"Real-ESRGAN 目录不存在: {self.realesrgan_dir}")
        
        # 检查推理脚本
        inference_script = self.realesrgan_dir / "inference_realesrgan.py"
        if not inference_script.exists():
            raise FileNotFoundError(f"推理脚本不存在: {inference_script}")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标记为已验证
        self._model = "verified"
        
        print(f"[{self.agent_name}] ✓ Real-ESRGAN 环境验证完成")
    
    def get_tool_function(self) -> Callable:
        """返回图像增强工具函数"""
        
        # 捕获 self 以便在工具函数中使用
        agent = self
        
        @tool
        def enhance_image(image_path: str) -> str:
            """使用 Real-ESRGAN 进行图像超分辨率增强
            
            Args:
                image_path: 输入图片路径
                
            Returns:
                增强后的图片路径
            """
            try:
                # 检查文件是否存在
                if not os.path.exists(image_path):
                    return f"错误：图片文件不存在 - {image_path}"
                
                # 确保环境已验证
                if agent._model is None:
                    agent.load_model()
                
                # 创建临时输入目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_input = Path(temp_dir) / "input"
                    temp_input.mkdir(parents=True, exist_ok=True)
                    
                    # 复制图片到临时目录
                    src_path = Path(image_path)
                    dst_path = temp_input / src_path.name
                    shutil.copy2(src_path, dst_path)
                    
                    # 构建 Real-ESRGAN 命令
                    cmd = [
                        "conda", "run", "-n", agent.conda_env,
                        "python", "inference_realesrgan.py",
                        "-n", agent.model_name,
                        "-i", str(temp_input),
                        "-o", str(agent.output_dir),
                        "--fp32"  # 使用 FP32 提高兼容性
                    ]
                    
                    print(f"[{agent.agent_name}] 执行超分辨率增强...")
                    print(f"[{agent.agent_name}] 命令: {' '.join(cmd)}")
                    
                    # 执行 Real-ESRGAN
                    result = subprocess.run(
                        cmd,
                        cwd=agent.realesrgan_dir,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5分钟超时
                    )
                    
                    if result.returncode != 0:
                        error_msg = result.stderr if result.stderr else "未知错误"
                        return f"增强失败：Real-ESRGAN 执行出错 - {error_msg}"
                    
                    # 查找增强后的图片
                    enhanced_name = src_path.stem + "_out" + src_path.suffix
                    enhanced_path = agent.output_dir / enhanced_name
                    
                    if not enhanced_path.exists():
                        return f"增强失败：未找到输出图片 {enhanced_name}"
                    
                    result_msg = f"增强成功：\n输入: {image_path}\n输出: {str(enhanced_path)}"
                    print(f"[{agent.agent_name}] ✓ {result_msg}")
                    
                    return result_msg
            
            except subprocess.TimeoutExpired:
                return "增强失败：处理超时（超过5分钟）"
            except Exception as e:
                return f"增强失败：{str(e)}"
        
        return enhance_image
    
    def get_system_prompt(self) -> str:
        """返回预处理 Agent 的 System Prompt"""
        return """你是一个图像预处理专家，专门负责图像超分辨率增强。

你的专业领域：
- 图像超分辨率增强（使用 Real-ESRGAN）
- 提升图像质量以改善 OCR 识别效果
- 处理模糊、低分辨率的图像
- 4倍放大增强

处理原则：
- 保持图像的原始内容和特征
- 增强图像的清晰度和细节
- 优化文字边缘以提升 OCR 准确度
- 适用于手写体和印刷体图像

当用户提供图片路径时，使用 enhance_image 工具进行超分辨率增强。"""
    
    def enhance(self, image_path: str) -> str:
        """
        直接执行图像增强（不通过 LangChain Agent）
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            增强后的图片路径
        """
        # 确保环境已验证
        if self._model is None:
            self.load_model()
        
        # 检查文件
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 创建临时输入目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / "input"
            temp_input.mkdir(parents=True, exist_ok=True)
            
            # 复制图片
            src_path = Path(image_path)
            dst_path = temp_input / src_path.name
            shutil.copy2(src_path, dst_path)
            
            # 构建命令
            cmd = [
                "conda", "run", "-n", self.conda_env,
                "python", "inference_realesrgan.py",
                "-n", self.model_name,
                "-i", str(temp_input),
                "-o", str(self.output_dir),
                "--fp32"
            ]
            
            if self.verbose:
                print(f"[{self.agent_name}] 执行: {' '.join(cmd)}")
            
            # 执行
            result = subprocess.run(
                cmd,
                cwd=self.realesrgan_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Real-ESRGAN 执行失败: {result.stderr}")
            
            # 获取输出路径
            enhanced_name = src_path.stem + "_out" + src_path.suffix
            enhanced_path = self.output_dir / enhanced_name
            
            if not enhanced_path.exists():
                raise FileNotFoundError(f"未找到输出图片: {enhanced_name}")
            
            return str(enhanced_path)


# 保持向后兼容的工厂函数
def create_preprocessing_agent(instructions: str = "", verbose: bool = False):
    """
    创建预处理 Agent（工厂函数）
    
    Args:
        instructions: 可选的额外指令
        verbose: 是否打印详细输出（默认False）
    
    Returns:
        PreprocessingAgent 实例
    """
    agent = PreprocessingAgent(instructions=instructions, verbose=verbose)
    return agent.build_agent()


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试预处理 Agent")
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    parser.add_argument('--use-agent', action='store_true', help='使用 LangChain Agent 模式（否则直接调用）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("测试预处理 Agent (Real-ESRGAN)")
    print("=" * 60)
    
    # 创建 Agent
    preprocessing_agent = PreprocessingAgent(verbose=args.verbose)
    
    print(f"\nAgent 信息: {preprocessing_agent.get_info()}")
    
    if args.use_agent:
        # 方式 1: 使用 LangChain Agent（完整的对话式）
        print("\n使用 LangChain Agent 模式")
        result = preprocessing_agent.invoke({
            "input": f"请对这张图片进行超分辨率增强：{args.image}"
        })
        print("\n增强结果：")
        print(result['output'])
    else:
        # 方式 2: 直接调用增强方法（更快）
        print("\n使用直接调用模式")
        try:
            enhanced_path = preprocessing_agent.enhance(args.image)
            print(f"\n✓ 增强成功！")
            print(f"输入: {args.image}")
            print(f"输出: {enhanced_path}")
        except Exception as e:
            print(f"\n✗ 增强失败: {str(e)}")

