#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2-VL Pipeline 测试脚本
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import process_image

def test_simple_recognition():
    """测试简单识别任务"""
    print("="*80)
    print("测试 1: 简单识别任务 (直接推理模式)")
    print("="*80)
    
    test_image = "../ocrbench1/OCRBench_Images/ChartQA/test/png/427.png"
    test_query = "What is written in the image?"
    
    result = process_image(
        image_path=test_image,
        query=test_query,
        output_path="test_simple_output.json",
        verbose=True
    )
    
    if result:
        print("\n✓ 简单识别测试完成")
        print(f"问题类型: {result.get('query_type', 'unknown')}")
        print(f"是否跳过 Agent: {result.get('skip_agents', False)}")
        if 'final_answer' in result:
            print(f"最终答案: {result['final_answer'][:200]}...")
    else:
        print("\n✗ 简单识别测试失败")
    
    return result


def test_complex_analysis():
    """测试复杂分析任务"""
    print("\n" + "="*80)
    print("测试 2: 复杂分析任务 (完整 OCR pipeline)")
    print("="*80)
    
    test_image = "../ocrbench1/OCRBench_Images/DocVQA/test/pngs/page1.png"
    test_query = "What is the document title?"
    
    result = process_image(
        image_path=test_image,
        query=test_query,
        output_path="test_complex_output.json",
        verbose=True
    )
    
    if result:
        print("\n✓ 复杂分析测试完成")
        print(f"问题类型: {result.get('query_type', 'unknown')}")
        print(f"是否跳过 Agent: {result.get('skip_agents', False)}")
        if 'final_answer' in result:
            print(f"最终答案: {result['final_answer'][:200]}...")
    else:
        print("\n✗ 复杂分析测试失败")
    
    return result


def main():
    """主测试函数"""
    print("\n" + "🎯 " + "="*75)
    print("Qwen2-VL Pipeline 测试")
    print("="*80 + "\n")
    
    # 测试1: 简单识别
    try:
        test_simple_recognition()
    except Exception as e:
        print(f"\n❌ 简单识别测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 复杂分析
    try:
        test_complex_analysis()
    except Exception as e:
        print(f"\n❌ 复杂分析测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✓ 所有测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
