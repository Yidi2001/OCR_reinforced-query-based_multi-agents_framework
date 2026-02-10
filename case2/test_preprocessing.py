#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 PreprocessingAgent
演示图像超分辨率增强功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from case2.preprocessing_agent import PreprocessingAgent


def test_preprocessing_agent():
    """测试预处理 Agent"""
    print("\n" + "="*70)
    print(" " * 20 + "PreprocessingAgent 测试")
    print("="*70)
    
    # 测试图片路径（使用一张小分辨率图片）
    test_image = "OCRBench_Images/IC15_1811/imgs/000000229.jpg"
    
    if not Path(test_image).exists():
        print(f"❌ 测试图片不存在: {test_image}")
        print("请提供有效的图片路径")
        return
    
    print(f"\n测试图片: {test_image}")
    
    # 方式 1: 创建 Agent 实例
    print("\n【方式 1】使用 Agent 类（推荐）")
    agent = PreprocessingAgent(verbose=True)
    
    # 获取 Agent 信息
    info = agent.get_info()
    print(f"\nAgent 名称: {info['name']}")
    print(f"Agent 描述: {info['description']}")
    print(f"Agent 类型: {info['type']}")
    
    # 执行增强
    print("\n开始图像增强...")
    try:
        enhanced_path = agent.enhance(test_image)
        print(f"\n✓ 增强成功！")
        print(f"输入: {test_image}")
        print(f"输出: {enhanced_path}")
        
        # 比较文件大小
        from PIL import Image
        original = Image.open(test_image)
        enhanced = Image.open(enhanced_path)
        
        print(f"\n图片尺寸对比:")
        print(f"  原图: {original.size[0]}x{original.size[1]}")
        print(f"  增强: {enhanced.size[0]}x{enhanced.size[1]}")
        print(f"  放大倍数: {enhanced.size[0] / original.size[0]:.1f}x")
        
    except Exception as e:
        print(f"\n✗ 增强失败: {str(e)}")
    
    # 方式 2: 使用 LangChain Agent 接口（可选）
    print("\n\n【方式 2】使用 LangChain Agent 接口")
    print("（需要 DeepSeek API，更慢但支持对话）")
    print("跳过此测试...")
    
    # 如果需要测试 LangChain 接口，可以取消注释：
    # result = agent.invoke({
    #     "input": f"请对这张图片进行超分辨率增强：{test_image}"
    # })
    # print(result['output'])


def test_in_pipeline():
    """测试在 Pipeline 中使用 PreprocessingAgent"""
    print("\n\n" + "="*70)
    print(" " * 20 + "Pipeline 集成测试")
    print("="*70)
    
    print("\nPreprocessingAgent 已集成到 orchestrator.py 中")
    print("当 Task Planner 判断需要超分辨率增强时，会自动调用此 Agent")
    print("\n运行完整 Pipeline 测试:")
    print("  python case2/orchestrator.py --image YOUR_IMAGE.jpg")


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print(" " * 15 + "PreprocessingAgent 测试")
    print(" " * 10 + "基于 BaseAgent 的超分辨率增强")
    print("="*70)
    
    # 测试 Agent
    test_preprocessing_agent()
    
    # 测试 Pipeline 集成
    test_in_pipeline()
    
    print("\n" + "="*70)
    print("✓ 测试完成")
    print("="*70)
    print("\n提示：")
    print("  - 查看增强后的图片: case1/chaofen/Real-ESRGAN/results/")
    print("  - 运行完整 Pipeline: python case2/orchestrator.py --image IMAGE.jpg")
    print()


if __name__ == "__main__":
    main()

