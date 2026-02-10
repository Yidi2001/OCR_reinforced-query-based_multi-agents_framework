#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的摘要生成工具
为已有的Pipeline结果JSON生成关键信息摘要
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from case2.result_summarizer import ResultSummarizer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='为Pipeline结果生成关键信息摘要',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为单个结果文件生成摘要
  python case2/generate_summary.py case2_output/example1_result.json
  
  # 批量处理多个文件
  python case2/generate_summary.py case2_output/*.json
  
  # 指定自定义文本长度
  python case2/generate_summary.py -l 1000 case2_output/example1_result.json
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='输入的Pipeline结果JSON文件路径（支持多个）'
    )
    
    parser.add_argument(
        '-l', '--max-length',
        type=int,
        default=800,
        help='OCR文本预览的最大长度（默认: 800）'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='只生成summary.json，不生成prompt.txt'
    )
    
    parser.add_argument(
        '--prompt-only',
        action='store_true',
        help='只生成prompt.txt，不生成summary.json'
    )
    
    args = parser.parse_args()
    
    # 创建摘要器
    summarizer = ResultSummarizer(max_ocr_text_length=args.max_length)
    
    # 处理文件
    success_count = 0
    fail_count = 0
    
    print("=" * 80)
    print(" " * 25 + "生成Pipeline结果摘要")
    print("=" * 80)
    
    for input_file in args.input_files:
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"\n❌ 文件不存在: {input_file}")
            fail_count += 1
            continue
        
        if not input_path.suffix == '.json':
            print(f"\n⚠️  跳过非JSON文件: {input_file}")
            continue
        
        # 跳过摘要文件本身
        if '_summary' in input_path.stem or '_prompt' in input_path.stem:
            print(f"\n⚠️  跳过摘要文件: {input_file}")
            continue
        
        print(f"\n处理: {input_file}")
        print("-" * 80)
        
        try:
            # 提取摘要
            summary = summarizer.summarize_from_file(input_file)
            
            # 生成输出路径
            base_path = input_path.parent / input_path.stem
            summary_json_path = f"{base_path}_summary.json"
            prompt_txt_path = f"{base_path}_prompt.txt"
            
            # 生成summary.json
            if not args.prompt_only:
                summarizer.save_summary(summary, summary_json_path)
                
                # 显示文件大小对比
                original_size = input_path.stat().st_size / 1024
                summary_size = Path(summary_json_path).stat().st_size / 1024
                compression_ratio = (summary_size / original_size) * 100
                
                print(f"  ✓ 摘要JSON: {summary_json_path}")
                print(f"    - 原始大小: {original_size:.2f} KB")
                print(f"    - 摘要大小: {summary_size:.2f} KB")
                print(f"    - 压缩率: {compression_ratio:.1f}%")
            
            # 生成prompt.txt
            if not args.summary_only:
                prompt = summarizer.format_as_prompt(summary)
                with open(prompt_txt_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                prompt_size = Path(prompt_txt_path).stat().st_size / 1024
                print(f"  ✓ 提示文本: {prompt_txt_path}")
                print(f"    - 文件大小: {prompt_size:.2f} KB")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    # 打印总结
    print("\n" + "=" * 80)
    print(f"完成! 成功: {success_count}, 失败: {fail_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()

