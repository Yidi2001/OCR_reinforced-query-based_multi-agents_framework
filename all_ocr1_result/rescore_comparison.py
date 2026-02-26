#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新评分 baseline 和 pipeline 对比文件
使用更宽松的归一化规则：去 LaTeX 符、全角转半角、去空格、可选不区分大小写
"""

import json
import re
import unicodedata


def normalize_for_comparison(text, case_insensitive=True):
    """
    归一化文本用于比较
    
    处理：
    1. 去除 LaTeX 包装符（$$, $, \\[, \\]）
    2. 全角转半角
    3. 去除所有空格、换行
    4. 可选：转小写
    """
    if text is None:
        return ""
    
    text = str(text).strip()
    
    # 1. 去除 LaTeX 包装符
    for wrapper in ['$$', '\\[', '\\]']:
        text = text.replace(wrapper, '')
    # 单独处理 $ (避免误删 LaTeX 命令里的)
    text = re.sub(r'(?<![\\a-zA-Z])\$(?![a-zA-Z])', '', text)
    
    # 2. 全角转半角
    text = unicodedata.normalize('NFKC', text)
    
    # 3. 去除空格和换行
    text = text.replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', '')
    
    # 4. 转小写（可选）
    if case_insensitive:
        text = text.lower()
    
    return text


def is_semantically_same(predict, answers, case_insensitive=True):
    """
    判断 predict 和 answers 是否语义一致
    
    宽松规则：
    1. 归一化后完全一致
    2. predict 包含 answer（处理多余解释文字）
    3. answer 包含 predict（处理 predict 不完整但核心对）
    
    Args:
        predict: 预测结果（字符串）
        answers: 标准答案（字符串或字符串列表）
        case_insensitive: 是否不区分大小写（默认 True）
    
    Returns:
        bool: 是否一致
    """
    pred_norm = normalize_for_comparison(predict, case_insensitive)
    
    # 处理 answers 为列表的情况
    if isinstance(answers, list):
        ans_list = answers
    else:
        ans_list = [answers]
    
    # 如果 predict 为空
    if not pred_norm:
        return any(not normalize_for_comparison(a, case_insensitive) for a in ans_list)
    
    # 与任一答案匹配即可
    for ans in ans_list:
        ans_norm = normalize_for_comparison(ans, case_insensitive)
        
        if not ans_norm:
            continue
        
        # 1. 完全一致
        if pred_norm == ans_norm:
            return True
        
        # 2. predict 包含 answer（处理多余解释，如 "The answer is XXX"）
        if ans_norm in pred_norm:
            # 对于短答案（< 15 字符），只要包含就算对
            if len(ans_norm) < 15:
                return True
            # 对于长答案，检查比例
            len_ratio = len(ans_norm) / len(pred_norm)
            if len_ratio >= 0.25:  # answer 至少占 predict 的 25%
                return True
        
        # 3. answer 包含 predict（处理 predict 不完整但核心内容对）
        if pred_norm in ans_norm:
            # 对于短 predict（< 10 字符），比例要求更高
            if len(pred_norm) < 10:
                len_ratio = len(pred_norm) / len(ans_norm)
                if len_ratio >= 0.6:  # predict 至少占 answer 的 60%
                    return True
            else:
                len_ratio = len(pred_norm) / len(ans_norm)
                if len_ratio >= 0.4:  # predict 至少占 answer 的 40%
                    return True
    
    return False


def rescore_file(input_file, output_file, case_insensitive=True):
    """
    重新评分对比文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        case_insensitive: 是否不区分大小写
    """
    print(f"\n📋 处理文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    baseline_changed = 0
    pipeline_changed = 0
    baseline_0to1 = 0  # 从错变对
    baseline_1to0 = 0  # 从对变错
    pipeline_0to1 = 0
    pipeline_1to0 = 0
    
    for item in data:
        answers = item.get('answers')
        baseline_pred = item.get('baseline_predict', '')
        pipeline_pred = item.get('pipeline_predict', '')
        
        old_baseline_result = item.get('baseline_result', 0)
        old_pipeline_result = item.get('pipeline_result', 0)
        
        # 重新判断
        new_baseline_result = 1 if is_semantically_same(baseline_pred, answers, case_insensitive) else 0
        new_pipeline_result = 1 if is_semantically_same(pipeline_pred, answers, case_insensitive) else 0
        
        # 更新
        item['baseline_result'] = new_baseline_result
        item['pipeline_result'] = new_pipeline_result
        
        # 统计变化
        if old_baseline_result != new_baseline_result:
            baseline_changed += 1
            if new_baseline_result == 1:
                baseline_0to1 += 1
            else:
                baseline_1to0 += 1
        
        if old_pipeline_result != new_pipeline_result:
            pipeline_changed += 1
            if new_pipeline_result == 1:
                pipeline_0to1 += 1
            else:
                pipeline_1to0 += 1
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 完成重新评分")
    print(f"  - 总条目: {len(data)}")
    print(f"\n  baseline_result 变化:")
    print(f"    - 从错变对 (0→1): {baseline_0to1} 条")
    print(f"    - 从对变错 (1→0): {baseline_1to0} 条")
    print(f"    - 总变化: {baseline_changed} 条")
    
    print(f"\n  pipeline_result 变化:")
    print(f"    - 从错变对 (0→1): {pipeline_0to1} 条")
    print(f"    - 从对变错 (1→0): {pipeline_1to0} 条")
    print(f"    - 总变化: {pipeline_changed} 条")
    
    print(f"\n  - 已保存到: {output_file}")
    
    # 统计新的结果
    baseline_correct = sum(1 for x in data if x.get('baseline_result') == 1)
    pipeline_correct = sum(1 for x in data if x.get('pipeline_result') == 1)
    
    print(f"\n📊 重新评分后统计:")
    print(f"  - baseline 正确: {baseline_correct}/{len(data)}")
    print(f"  - pipeline 正确: {pipeline_correct}/{len(data)}")
    
    return data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='重新评分 baseline vs pipeline 对比文件')
    parser.add_argument('--input', type=str, nargs='+', help='输入文件路径（可多个）')
    parser.add_argument('--case-sensitive', action='store_true', help='区分大小写（默认不区分）')
    parser.add_argument('--batch', action='store_true', help='批量处理 baseline0_pipeline1 和 baseline1_pipeline0')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理两个文件
        files = [
            'baseline0_pipeline1_items.json',
            'baseline1_pipeline0_items.json'
        ]
    elif args.input:
        files = args.input
    else:
        print("❌ 请指定 --input 或使用 --batch")
        return
    
    print("\n" + "="*70)
    print(" " * 20 + "重新评分对比文件")
    print("="*70)
    print(f"\n规则:")
    print(f"  - 去除 LaTeX 包装符（$$, $, \\[, \\]）")
    print(f"  - 全角转半角")
    print(f"  - 去除所有空格、换行")
    print(f"  - {'不' if not args.case_sensitive else ''}区分大小写")
    print(f"  - 支持 predict 包含 answer 或 answer 包含 predict")
    print("="*70)
    
    for input_file in files:
        output_file = input_file.replace('.json', '_rescored.json')
        rescore_file(input_file, output_file, case_insensitive=not args.case_sensitive)
    
    print("\n" + "="*70)
    print("✓ 所有文件重新评分完成！")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
