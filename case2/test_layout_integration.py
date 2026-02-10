#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试布局整合功能
验证 pipeline 中的自动整合是否工作
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from case2.pipeline import process_image
import json


def test_layout_integration():
    """测试布局整合功能"""
    
    print("=" * 80)
    print("测试 Pipeline 布局整合功能")
    print("=" * 80)
    
    # 测试图片（复杂文档，会触发布局检测）
    test_image = "OCRBench_Images/docVQA/val/documents/flpp0227_16.png"
    
    if not Path(test_image).exists():
        print(f"❌ 测试图片不存在: {test_image}")
        print("\n请使用一个实际的复杂文档图片进行测试")
        return
    
    print(f"\n📷 测试图片: {test_image}")
    print("\n正在执行完整流程...")
    print("-" * 80)
    
    # 执行 pipeline
    result = process_image(
        image_path=test_image,
        query="识别图片中的所有文字，保持原有结构",
        output_path="case2_output/layout_test_result.json",
        example_name="布局整合测试"
    )
    
    if not result:
        print("\n❌ Pipeline 执行失败")
        return
    
    # 分析结果
    print("\n" + "=" * 80)
    print("📊 结果分析")
    print("=" * 80)
    
    # 检查是否有布局检测
    exec_results = result.get('execution_results', {})
    layout_result = exec_results.get('layout_result', {})
    
    if not layout_result:
        print("\n⚠️  此图片未触发布局检测，无法测试整合功能")
        return
    
    # 显示整合统计
    merge_stats = layout_result.get('merge_stats', {})
    merged_blocks = layout_result.get('merged_blocks', [])
    
    print(f"\n✓ 布局检测结果:")
    print(f"  原始区域数: {layout_result.get('detected_regions', 0)}")
    
    if merge_stats:
        print(f"\n✓ 整合统计:")
        print(f"  原始区域: {merge_stats.get('original_regions', 0)}")
        print(f"  整合后块数: {merge_stats.get('merged_blocks', 0)}")
        print(f"  压缩比例: {merge_stats.get('original_regions', 0) / max(merge_stats.get('merged_blocks', 1), 1):.1f}x")
        
        merge_params = merge_stats.get('merge_params', {})
        if merge_params:
            print(f"\n  整合参数:")
            for key, value in merge_params.items():
                print(f"    - {key}: {value}")
    
    if merged_blocks:
        print(f"\n✓ 整合后的块:")
        for i, block in enumerate(merged_blocks[:3], 1):  # 只显示前3个
            title = block.get('title', '(无标题)')
            text_preview = block.get('text', '')[:80] + '...' if len(block.get('text', '')) > 80 else block.get('text', '')
            children_count = len(block.get('children', []))
            
            print(f"\n  Block {i}:")
            print(f"    标题: {title}")
            print(f"    子区域数: {children_count}")
            print(f"    文字预览: {text_preview}")
        
        if len(merged_blocks) > 3:
            print(f"\n  ... 还有 {len(merged_blocks) - 3} 个块")
    
    # 检查生成的文件
    print("\n" + "=" * 80)
    print("📁 生成的文件")
    print("=" * 80)
    
    files_to_check = [
        ("完整结果", "case2_output/layout_test_result.json"),
        ("摘要JSON", "case2_output/layout_test_summary.json"),
        ("证据包文本", "case2_output/layout_test_prompt.txt")
    ]
    
    for name, filepath in files_to_check:
        if Path(filepath).exists():
            size_kb = Path(filepath).stat().st_size / 1024
            print(f"✓ {name}: {filepath} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {name}: {filepath} (未生成)")
    
    # 显示证据包预览
    prompt_file = Path("case2_output/layout_test_prompt.txt")
    if prompt_file.exists():
        print("\n" + "=" * 80)
        print("📝 证据包文本预览")
        print("=" * 80)
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 显示前1000个字符
            preview = content[:1000] + "\n\n... (完整内容见文件)" if len(content) > 1000 else content
            print(preview)
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
    
    print("\n💡 提示:")
    print("  1. 查看 case2_output/layout_test_prompt.txt 了解证据包格式")
    print("  2. 对比 result.json 中的 merged_blocks 和原始 boxes")
    print("  3. 整合后的块可以直接用于 phi_refiner")
    print()


if __name__ == "__main__":
    test_layout_integration()



