#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Agent 架构
展示如何使用基于 BaseAgent 的新架构
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from case2.trocr import HandOCRAgent, create_hand_ocr_agent
from case2.printed_ocr_agent import PrintedOCRAgent, create_printed_ocr_agent


def test_hand_ocr_agent():
    """测试手写体识别 Agent"""
    print("=" * 70)
    print(" " * 20 + "测试 HandOCRAgent")
    print("=" * 70)
    
    # 使用类（推荐方式）
    print("\n【方式 1】直接使用类")
    agent = HandOCRAgent(
        instructions="Focus on cursive handwriting and maintain proper word spacing.",
        verbose=False
    )
    
    # 获取 Agent 信息
    info = agent.get_info()
    print(f"\nAgent 名称: {info['name']}")
    print(f"Agent 描述: {info['description']}")
    print(f"Agent 类型: {info['type']}")
    print(f"额外指令: {info['instructions']}")
    
    # 测试图片（需要替换为实际路径）
    test_image = "OCRBench_Images/IC15_1811/imgs/000000229.jpg"
    
    if Path(test_image).exists():
        print(f"\n测试图片: {test_image}")
        print("正在识别...")
        
        # 执行识别
        result = agent.invoke({
            "input": f"请识别这张手写图片：{test_image}"
        })
        
        print("\n识别结果:")
        print("-" * 70)
        print(result['output'])
        print("-" * 70)
    else:
        print(f"\n⚠️ 测试图片不存在: {test_image}")
        print("请提供有效的手写体图片路径")
    
    print("\n【方式 2】使用工厂函数（向后兼容）")
    agent_executor = create_hand_ocr_agent(verbose=False)
    print(f"✓ Agent Executor 创建成功: {type(agent_executor)}")


def test_printed_ocr_agent():
    """测试印刷体识别 Agent"""
    print("\n\n" + "=" * 70)
    print(" " * 20 + "测试 PrintedOCRAgent")
    print("=" * 70)
    
    # 使用类（推荐方式）
    print("\n【方式 1】直接使用类")
    agent = PrintedOCRAgent(
        instructions="Extract all text with high accuracy, including special characters.",
        verbose=False
    )
    
    # 获取 Agent 信息
    info = agent.get_info()
    print(f"\nAgent 名称: {info['name']}")
    print(f"Agent 描述: {info['description']}")
    print(f"Agent 类型: {info['type']}")
    print(f"额外指令: {info['instructions']}")
    
    # 测试图片（需要替换为实际路径）
    test_image = "test_images/printed_sample.jpg"
    
    if Path(test_image).exists():
        print(f"\n测试图片: {test_image}")
        print("正在识别...")
        
        # 执行识别
        result = agent.invoke({
            "input": f"请识别这张印刷体图片：{test_image}"
        })
        
        print("\n识别结果:")
        print("-" * 70)
        print(result['output'])
        print("-" * 70)
    else:
        print(f"\n⚠️ 测试图片不存在: {test_image}")
        print("请提供有效的印刷体图片路径")
    
    print("\n【方式 2】使用工厂函数（向后兼容）")
    agent_executor = create_printed_ocr_agent(verbose=False)
    print(f"✓ Agent Executor 创建成功: {type(agent_executor)}")


def test_agent_comparison():
    """对比两种 Agent 的使用方式"""
    print("\n\n" + "=" * 70)
    print(" " * 20 + "Agent 架构对比")
    print("=" * 70)
    
    print("\n【统一的接口】")
    print("所有 Agent 都继承自 BaseAgent，具有相同的接口：")
    print("  - load_model()         : 加载模型")
    print("  - get_tool_function()  : 获取工具函数")
    print("  - get_system_prompt()  : 获取 System Prompt")
    print("  - build_agent()        : 构建 LangChain Agent")
    print("  - invoke(input_dict)   : 标准 LangChain 接口")
    print("  - execute(path, query) : 简化执行接口")
    print("  - get_info()           : 获取 Agent 信息")
    
    print("\n【已实现的 Agents】")
    agents = [
        HandOCRAgent(),
        PrintedOCRAgent()
    ]
    
    for agent in agents:
        info = agent.get_info()
        print(f"  ✅ {info['name']:<20} - {info['description']}")
    
    print("\n【待实现的 Agents】")
    print("  ⏳ SuperResolutionAgent  - 图像超分辨率增强")
    print("  ⏳ LayoutDetectionAgent  - 文档布局检测和分析")
    
    print("\n【架构优势】")
    print("  1. 统一接口 - 所有 Agent 使用方式一致")
    print("  2. 代码复用 - 通用逻辑在基类中实现")
    print("  3. 易于扩展 - 新 Agent 只需继承基类")
    print("  4. 单例模式 - 模型只加载一次")
    print("  5. 灵活配置 - 支持自定义指令和 LLM")


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print(" " * 15 + "Case2 Agent 架构测试")
    print(" " * 10 + "基于 BaseAgent 的统一架构")
    print("=" * 70)
    
    # 测试手写体 Agent
    test_hand_ocr_agent()
    
    # 测试印刷体 Agent
    test_printed_ocr_agent()
    
    # Agent 对比
    test_agent_comparison()
    
    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)
    print("\n提示：")
    print("  - 查看 AGENT_ARCHITECTURE.md 了解详细架构设计")
    print("  - 查看 AGENTS_README.md 了解使用说明")
    print("  - 运行单个 Agent: python case2/trocr.py")
    print("  - 运行单个 Agent: python case2/printed_ocr_agent.py")
    print()


if __name__ == "__main__":
    main()

