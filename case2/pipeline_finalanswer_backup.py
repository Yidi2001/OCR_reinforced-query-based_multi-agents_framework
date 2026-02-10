#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Agent OCR Pipeline
完整的智能 OCR 处理流程：自动判断图片类型并调用对应的 Agent
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from case2.orchestrator import MultiAgentOrchestrator
from case2.result_summarizer import ResultSummarizer
from case2.merge_layout_blocks_ratio import merge_layout_regions_ratio_dedup_v2
from case2.token_budget_calculator import TokenBudgetCalculator
from case2.phi_refiner import PhiRefiner
import json
import re
import math
import hashlib


def process_image(image_path: str, query: str, output_path: str = None, example_name: str = None, 
                  generate_summary: bool = True, enable_refinement: bool = True, 
                  auto_token_budget: bool = True, verbose: bool = True, sample_output_dir: str = None,
                  orchestrator: 'MultiAgentOrchestrator' = None, refiner: 'PhiRefiner' = None):
    """
    处理单张图片的完整流程
    
    Pipeline 工作流程：
    1. Target Detection - 自动判断图片类型（手写体/印刷体）
    2. Task Planning - Phi3.5-Vision 分析图片质量和复杂度
    3. Prompt Generation - 生成结构化执行计划
    4. Agent Execution - 根据计划自动选择并执行对应的 Agent
    5. Layout Integration - 整合布局检测结果
    6. Summary Generation - 自动生成关键信息摘要
    7. Layout Selection - 根据相关性选择布局区域（可选）
    8. Answer Refinement - 使用 Phi3.5-Vision 生成最终答案（可选）
    
    Args:
        image_path: 图片路径
        query: 用户查询/任务描述
        output_path: 输出结果路径（可选）
        example_name: 任务名称（可选）
        generate_summary: 是否自动生成摘要（默认True）
        enable_refinement: 是否启用最终答案生成（默认True）
        auto_token_budget: 是否自动计算 token 预算（默认True）
        verbose: 是否显示详细输出（默认True）
        sample_output_dir: 样本专属输出目录（用于保存可视化图片等）
        orchestrator: 复用的 orchestrator 实例（可选，用于批处理时避免重复加载模型）
        refiner: 复用的 refiner 实例（可选，用于批处理时避免重复加载模型）
    
    Returns:
        执行结果字典，包含分类、规划、执行结果和最终答案
    """
    if example_name and verbose:
        print("\n" + "="*70)
        print(f" " * 20 + f"任务: {example_name}")
        print("="*70)
    
    if not Path(image_path).exists():
        if verbose:
            print(f"❌ 错误：图片不存在 - {image_path}")
        return None
    
    # 创建或复用编排器
    if orchestrator is None:
        orchestrator = MultiAgentOrchestrator(execute_agents=True)
    
    # 运行完整流程（如果 verbose=False，抑制输出）
    import sys
    from io import StringIO
    
    if verbose:
        result = orchestrator.run(
            image_path=image_path,
            query=query,
            output_path=output_path,
            sample_output_dir=sample_output_dir
        )
    else:
        # 临时重定向 stdout 来抑制输出
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            result = orchestrator.run(
                image_path=image_path,
                query=query,
                output_path=output_path,
                sample_output_dir=sample_output_dir
            )
        finally:
            sys.stdout = old_stdout
    
    # 如果有布局检测结果，先进行布局整合
    if result:
        result = _integrate_layout_results(result, image_path, verbose=verbose)
    
    # 自动生成摘要
    if generate_summary and result:
        try:
            if verbose:
                print("\n📊 生成结果摘要...")
            summarizer = ResultSummarizer(max_ocr_text_length=800)
            
            # 生成摘要JSON（如果在样本文件夹中，使用更简洁的名称）
            if sample_output_dir and output_path and output_path.startswith(sample_output_dir):
                summary_json_path = str(Path(sample_output_dir) / "summary.json")
                prompt_txt_path = str(Path(sample_output_dir) / "prompt.txt")
            elif output_path:
                summary_json_path = output_path.replace('.json', '_summary.json')
                prompt_txt_path = output_path.replace('.json', '_prompt.txt')
            else:
                # 批量处理模式：使用临时路径（不实际保存）
                import tempfile
                temp_id = id(result)
                summary_json_path = str(Path(tempfile.gettempdir()) / f"summary_{temp_id}.json")
                prompt_txt_path = str(Path(tempfile.gettempdir()) / f"prompt_{temp_id}.txt")
            
            summary = summarizer.summarize(result)
            summarizer.save_summary(summary, summary_json_path)
            
            # 生成提示文本
            prompt = summarizer.format_as_prompt(summary)
            with open(prompt_txt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            if verbose:
                print(f"✓ 提示文本已保存到: {prompt_txt_path}")
            
            # 添加摘要信息到结果
            result['summary_files'] = {
                'summary_json': summary_json_path,
                'prompt_txt': prompt_txt_path
            }
            
        except Exception as e:
            print(f"\n⚠️  生成摘要失败: {e}")
            import traceback
            traceback.print_exc()
            # 摘要生成失败，设置变量为 None，后续步骤会跳过
            summary = None
            summary_json_path = None
        
        # 步骤 7 & 8: 生成最终答案（独立的 try-except）
        if enable_refinement:
            try:
                # 如果摘要生成失败，跳过最终答案生成
                if summary is None or summary_json_path is None:
                    if verbose:
                        print("⚠️  跳过最终答案生成：摘要生成失败")
                    return result
                
                layout_result = result.get('execution_results', {}).get('layout_result')
                has_layout = layout_result and layout_result.get('detected_regions', 0) > 0
                
                if has_layout:
                    # 复杂场景：先选择相关布局，再生成答案
                    if verbose:
                        print("\n📋 步骤 7: 选择相关布局区域...")
                    
                    # 生成文件名
                    if sample_output_dir and output_path and output_path.startswith(sample_output_dir):
                        selected_layout_path = str(Path(sample_output_dir) / "selected_layout.json")
                    elif output_path:
                        selected_layout_path = output_path.replace('.json', '_selected_layout.json')
                    else:
                        # 批量处理模式：使用临时路径（不实际保存）
                        import tempfile
                        selected_layout_path = str(Path(tempfile.gettempdir()) / f"selected_layout_{id(summary)}.json")
                    
                    selected_layout = _select_relevant_layouts(
                        summary, 
                        image_path, 
                        selected_layout_path,
                        auto_token_budget=auto_token_budget,
                        verbose=verbose
                    )
                    
                    if selected_layout:
                        result['selected_layout_file'] = selected_layout_path
                        input_for_refiner = selected_layout_path
                    else:
                        input_for_refiner = summary_json_path
                else:
                    # 简单场景：直接使用 summary
                    if verbose:
                        print("\n💡 简单场景，跳过布局选择，直接生成答案...")
                    input_for_refiner = summary_json_path
                
                # 步骤 8: 使用 Phi3.5-Vision 生成最终答案
                if verbose:
                    print("\n🤖 步骤 8: 使用 Phi3.5-Vision 生成最终答案...")
                
                # 生成文件名
                if sample_output_dir and output_path and output_path.startswith(sample_output_dir):
                    final_answer_path = str(Path(sample_output_dir) / "final_answer.json")
                elif output_path:
                    final_answer_path = output_path.replace('.json', '_final_answer.json')
                else:
                    # 批量处理模式：使用临时路径（不实际保存）
                    import tempfile
                    final_answer_path = str(Path(tempfile.gettempdir()) / f"final_answer_{id(summary)}.json")
                
                final_answer = _generate_final_answer(
                    input_for_refiner,
                    final_answer_path,
                    verbose=verbose,
                    refiner=refiner
                )
                
                if final_answer:
                    result['final_answer'] = final_answer.get('refined_response', '')
                    result['final_answer_file'] = final_answer_path
                    if verbose:
                        print(f"\n💡 最终答案: {result['final_answer']}")
                    
            except Exception as e:
                print(f"\n⚠️  生成最终答案失败: {e}")
                import traceback
                traceback.print_exc()
    
    return result


def _integrate_layout_results(result: dict, image_path: str, verbose: bool = True) -> dict:
    """
    整合布局检测结果
    
    如果执行结果中包含布局检测（多个区域），使用 merge_layout_blocks_ratio 进行：
    1. 区域去重（删除重复的检测框）
    2. 文本行合并（将同一段落的行合并）
    3. 块级整合（基于标题等锚点合并相关内容）
    4. 跨块去重（删除被其他块包含的小块）
    
    Args:
        result: orchestrator 的执行结果
        image_path: 图片路径
        
    Returns:
        整合后的结果
    """
    exec_results = result.get('execution_results', {})
    layout_result = exec_results.get('layout_result', {})
    
    # 检查是否有布局检测结果
    if not layout_result or layout_result.get('detected_regions', 0) == 0:
        return result
    
    if verbose:
        print("\n🔄 检测到布局区域，开始整合...")
    
    # 构建符合 merge_layout_blocks_ratio 输入格式的数据
    # 需要从 layout_result 和 OCR 结果中提取信息
    boxes = layout_result.get('boxes', [])
    
    # 从 agents_executed 中提取 OCR 结果
    agents_executed = exec_results.get('agents_executed', [])
    ocr_agent_output = None
    for agent in agents_executed:
        if 'OCRAgent' in agent.get('agent_name', ''):
            ocr_agent_output = agent.get('output', '')
            break
    
    if not ocr_agent_output or not boxes:
        print("⚠️  没有足够的信息进行布局整合")
        return result
    
    # 解析 OCR 输出，提取每个区域的文字
    regions = _parse_ocr_output_to_regions(boxes, ocr_agent_output)
    
    if not regions:
        print("⚠️  无法解析 OCR 输出")
        return result
    
    # 获取图片分辨率
    from PIL import Image
    try:
        img = Image.open(image_path)
        resolution = f"{img.width}x{img.height}"
    except:
        resolution = "unknown"
    
    # 构建输入文档
    doc = {
        "image_path": image_path,
        "resolution": resolution,
        "ocr_results": {
            "regions": regions
        }
    }
    
    # 调用布局整合函数
    try:
        merged_doc = merge_layout_regions_ratio_dedup_v2(doc)
        merged_blocks = merged_doc.get('ocr_results', {}).get('merged_blocks', [])
        
        if verbose:
            print(f"✓ 布局整合完成:")
            print(f"  原始区域: {len(regions)}")
            print(f"  整合后块数: {len(merged_blocks)}")
        
        # 更新 result 中的 layout_result
        if merged_blocks:
            layout_result['merged_blocks'] = merged_blocks
            layout_result['merge_stats'] = {
                'original_regions': len(regions),
                'merged_blocks': len(merged_blocks),
                'merge_params': merged_doc.get('ocr_results', {}).get('merge_params', {})
            }
        
    except Exception as e:
        print(f"⚠️  布局整合失败: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def _parse_ocr_output_to_regions(boxes: list, ocr_output: str) -> list:
    """
    从 OCR Agent 的输出文本中解析出每个区域的信息
    
    Args:
        boxes: 布局检测的框列表
        ocr_output: OCR Agent 的输出文本
        
    Returns:
        regions 列表，每个元素包含 region_id, label, confidence, coordinate, text
    """
    regions = []
    
    # OCR 输出格式：
    # 基于布局检测的 OCR 结果（共 N 个区域）：
    # 
    # 区域 1: label (置信度: X.XXX)
    # 位置: [x1, y1, x2, y2]
    # 文字内容:
    # ...
    # ------------------------------------------------------------
    # 区域 2: ...
    
    # 按分隔符分割
    region_blocks = ocr_output.split('------------------------------------------------------------')
    
    for i, box in enumerate(boxes):
        region_id = i + 1
        
        # 找到对应的文本块
        block = None
        if i == 0 and region_blocks:
            # 第一个区域在第一个块中（包含标题）
            block = region_blocks[0]
        elif i < len(region_blocks):
            block = region_blocks[i]
        
        if not block:
            # 如果没有对应的文本块，使用空文本
            regions.append({
                "region_id": region_id,
                "label": box.get('label', 'unknown'),
                "confidence": box.get('score', 0.0),
                "coordinate": box.get('coordinate', []),
                "text": ""
            })
            continue
        
        # 提取文字内容
        text = _extract_text_from_ocr_block(block)
        
        regions.append({
            "region_id": region_id,
            "label": box.get('label', 'unknown'),
            "confidence": box.get('score', 0.0),
            "coordinate": box.get('coordinate', []),
            "text": text
        })
    
    return regions


def _extract_text_from_ocr_block(block: str) -> str:
    """
    从 OCR 输出块中提取文字内容
    
    查找 "文字内容:" 标记之后的所有文本
    """
    if not block:
        return ""
    
    # 查找 "文字内容:" 标记
    marker = "文字内容:"
    if marker not in block:
        marker = "文字内容："
    
    if marker not in block:
        return ""
    
    # 提取标记之后的内容
    parts = block.split(marker, 1)
    if len(parts) < 2:
        return ""
    
    text = parts[1].strip()
    
    # 移除后面可能的分隔线
    if '----' in text:
        text = text.split('----')[0].strip()
    
    return text


def _select_relevant_layouts(summary: dict, image_path: str, output_path: str, 
                            auto_token_budget: bool = True, verbose: bool = True) -> dict:
    """
    根据用户 query 选择相关的布局区域
    
    Args:
        summary: pipeline 生成的 summary JSON
        image_path: 图片路径
        output_path: 输出文件路径
        auto_token_budget: 是否自动计算 token 预算
        
    Returns:
        选择后的布局数据
    """
    try:
        # 导入必要的函数（从 layout_relevance_selector_v4.py）
        from typing import List, Dict, Any
        
        def normalize_text(text: str) -> str:
            text = text.replace("-\n", "")
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        
        def approx_token_len(text: str) -> int:
            return max(1, len(text) // 4)
        
        def cosine(a, b) -> float:
            dot = sum(x*y for x,y in zip(a,b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            return dot / (na*nb + 1e-9)
        
        def hash_embed(text: str, dim: int = 256):
            """使用确定性哈希"""
            vec = [0.0]*dim
            for w in re.findall(r"\w+", text.lower()):
                hash_val = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
                vec[hash_val % dim] += 1.0
            return vec
        
        def select_blocks_by_relevance(regions, query, token_budget, per_block_max_tokens=200):
            q_emb = hash_embed(query)
            scored = []
            for r in regions:
                text = r.get("text") or ""
                normalized_text = normalize_text(text)
                
                emb = hash_embed(normalized_text[:800])
                score = cosine(q_emb, emb)
                scored.append({
                    "region_id": r.get("region_id"),
                    "score": score,
                    "text": normalized_text,
                    "bbox": r.get("coordinate"),
                    "label": r.get("label"),
                    "confidence": r.get("confidence")
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            
            selected, used = [], 0
            for s in scored:
                t = approx_token_len(s["text"])
                if t > per_block_max_tokens:
                    s["text"] = s["text"][:per_block_max_tokens*4]
                    t = per_block_max_tokens
                if used + t > token_budget:
                    break
                s["tokens"] = t
                selected.append(s)
                used += t
            return selected
        
        # 获取 query 和 regions
        query = summary.get("user_query", "")
        regions = summary.get("ocr_results", {}).get("regions", [])
        
        if not regions:
            print("⚠️  没有布局区域可供选择")
            return None
        
        # 计算 token 预算
        if auto_token_budget:
            calculator = TokenBudgetCalculator(num_crops=4)
            token_budget = calculator.get_text_budget(image_path, max_output_tokens=2000)
            if verbose:
                print(f"  ✓ 自动计算的 token 预算: {token_budget}")
        else:
            token_budget = 127000  # 默认预算
        
        # 选择相关区域
        selected = select_blocks_by_relevance(regions, query, token_budget)
        
        # 构建输出数据
        result = {
            "image_path": summary.get("image_path"),
            "resolution": summary.get("resolution"),
            "user_query": query,
            "classification": summary.get("classification"),
            "agent_sequence": summary.get("agent_sequence"),
            "total_time": summary.get("total_time"),
            "mode": "query_relevance",
            "num_selected": len(selected),
            "blocks": selected
        }
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"  ✓ 选择了 {len(selected)} 个相关区域")
            print(f"  ✓ 结果已保存到: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"  ⚠️ 布局选择失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_final_answer(selected_layout_path: str, output_path: str, verbose: bool = True, 
                          refiner: 'PhiRefiner' = None) -> dict:
    """
    使用 Phi3.5-Vision 生成最终答案
    
    Args:
        selected_layout_path: 选择后的布局 JSON 文件路径
        output_path: 输出文件路径
        verbose: 是否显示详细输出
        refiner: 复用的 refiner 实例（可选）
        
    Returns:
        包含最终答案的字典
    """
    try:
        # 创建或复用 PhiRefiner
        if refiner is None:
            from phi_refiner import PhiRefiner
            refiner = PhiRefiner()
        
        # 从 selected_layout JSON 生成答案
        result = refiner.refine_from_summary_file(selected_layout_path)
        
        if "error" in result:
            print(f"  ⚠️ 生成答案失败: {result['error']}")
            return None
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"  ✓ 结果已保存到: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"  ⚠️ 生成最终答案失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_summary(results: list):
    """打印处理结果摘要"""
    print("\n\n" + "="*70)
    print(" " * 25 + "处理结果摘要")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        if result is None:
            continue
        
        # 获取分类结果
        classification = result.get('classification', {})
        label = classification.get('label', 'unknown')
        confidence = classification.get('confidence', 0.0)
        
        # 获取执行结果
        exec_results = result.get("execution_results", {})
        final_output = exec_results.get("final_output", "无")
        layout_result = exec_results.get("layout_result", {})
        
        # 获取使用的 Agent
        agents_executed = exec_results.get("agents_executed", [])
        agent_names = [a.get("agent_name", "") for a in agents_executed]
        agent_sequence = " → ".join(agent_names) if agent_names else "None"
        
        print(f"\n【任务 {i}】")
        print("-" * 70)
        print(f"图片: {result['image_path']}")
        print(f"分类: {label} (置信度: {confidence:.3f})")
        print(f"Agent 执行序列: {agent_sequence}")
        
        # 如果有布局检测结果，显示区域数量
        if layout_result and layout_result.get('detected_regions', 0) > 0:
            print(f"布局检测: {layout_result['detected_regions']} 个区域")
        
        print(f"规划时间: {result.get('planning_time', 0):.2f}秒")
        print(f"执行时间: {result.get('execution_time', 0):.2f}秒")
        print(f"总时间: {result.get('total_time', 0):.2f}秒")
        
        # 显示摘要文件位置
        if 'summary_files' in result:
            print(f"\n摘要文件:")
            print(f"  📊 JSON: {result['summary_files']['summary_json']}")
            print(f"  📝 提示: {result['summary_files']['prompt_txt']}")
        
        print(f"\nOCR 结果预览:")
        preview_text = final_output[:200] + "..." if len(final_output) > 200 else final_output
        print(preview_text)
    
    print("\n" + "="*70)


def process_from_json(json_path: str, output_file: str = "predictions.json", 
                     limit: int = None, enable_refinement: bool = True):
    """
    从 JSON 文件批量处理测试数据，输出单个汇总 JSON 文件
    
    Args:
        json_path: JSON 文件路径（如 OCRBench.json）
        output_file: 输出 JSON 文件路径（默认: predictions.json）
        limit: 限制处理的数量，None 表示处理全部
        enable_refinement: 是否启用最终答案生成
        
    Returns:
        处理结果列表
    """
    from pathlib import Path
    try:
        from tqdm import tqdm
    except ImportError:
        print("⚠️  未安装 tqdm，使用简单进度显示。安装: pip install tqdm")
        tqdm = None
    
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    total = len(data) if limit is None else min(limit, len(data))
    
    print(f"\n📋 从 {json_path} 读取到 {len(data)} 个测试样本")
    if limit:
        print(f"   限制处理前 {limit} 个样本")
    print("=" * 70)
    
    # 在批处理开始前创建共享的模型实例（避免重复加载）
    print("\n⚡ 初始化模型（批处理模式，模型复用 + RuntimeContext）...")
    shared_orchestrator = MultiAgentOrchestrator(execute_agents=True)
    
    # 创建共享的 refiner（如果需要），注入 orchestrator 的 ctx
    shared_refiner = None
    if enable_refinement:
        from phi_refiner import PhiRefiner
        shared_refiner = PhiRefiner(ctx=shared_orchestrator.ctx)
        print("✓ 模型初始化完成")
    
    # 使用 tqdm 显示进度条
    if tqdm:
        pbar = tqdm(data[:total], desc="处理进度", unit="样本", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = data[:total]
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, item in enumerate(pbar, 1):
        image_path = item.get('image_path', '')
        query = item.get('question', item.get('query', ''))
        
        if not image_path or not query:
            # 跳过但保留原始数据，predict 设为空
            item_copy = item.copy()
            item_copy['predict'] = ""
            results.append(item_copy)
            skip_count += 1
            if tqdm:
                pbar.set_postfix({'成功': success_count, '失败': fail_count, '跳过': skip_count})
            else:
                print(f"\n[{i}/{total}] ⚠️  跳过：缺少 image_path 或 question")
            continue
        
        # 修正图片路径：如果路径不以 OCRBench_Images/ 开头，且文件不存在，尝试添加前缀
        original_image_path = image_path
        if not Path(image_path).exists():
            # 尝试添加 OCRBench_Images/ 前缀
            prefixed_path = f"OCRBench_Images/{image_path}" if not image_path.startswith("OCRBench_Images/") else image_path
            if Path(prefixed_path).exists():
                image_path = prefixed_path
        
        if not tqdm:
            print(f"\n[{i}/{total}] 处理: {Path(image_path).name}")
        
        try:
            result = process_image(
                image_path=image_path,
                query=query,
                output_path=None,  # 不保存中间文件
                example_name=None,  # 批量处理时不显示详细任务信息
                generate_summary=True,
                enable_refinement=enable_refinement,
                verbose=False,  # 批量处理时静默模式
                sample_output_dir=None,  # 不需要样本目录
                orchestrator=shared_orchestrator,  # 复用模型
                refiner=shared_refiner  # 复用模型
            )
            
            # 构建输出项：原始 item + predict 字段
            item_copy = item.copy()
            if result and 'final_answer' in result:
                item_copy['predict'] = result['final_answer']
                success_count += 1
                if tqdm:
                    answer_preview = result['final_answer'][:20] + "..." if len(result['final_answer']) > 20 else result['final_answer']
                    pbar.set_postfix({
                        '成功': success_count, 
                        '失败': fail_count, 
                        '跳过': skip_count
                    })
                else:
                    print(f"   ✓ 答案: {result['final_answer']}")
            else:
                # 处理失败：result为空或没有final_answer
                print(f"\n   ⚠️  处理失败 (图片: {image_path})")
                print(f"      返回结果: {result}")
                item_copy['predict'] = ""
                fail_count += 1
                if tqdm:
                    pbar.set_postfix({'成功': success_count, '失败': fail_count, '跳过': skip_count})
            
            results.append(item_copy)
                    
        except Exception as e:
            # 失败时也保留原始数据，predict 设为空
            print(f"\n   ❌ 处理失败 (图片: {image_path}): {e}")
            import traceback
            traceback.print_exc()
            
            item_copy = item.copy()
            item_copy['predict'] = ""
            results.append(item_copy)
            fail_count += 1
            if tqdm:
                pbar.set_postfix({'成功': success_count, '失败': fail_count, '跳过': skip_count})
            continue
    
    if tqdm:
        pbar.close()
    
    # 保存汇总结果到单个 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n{'='*70}")
    print(f"📊 处理完成统计:")
    print(f"   ✓ 成功: {success_count}")
    print(f"   ✗ 失败: {fail_count}")
    print(f"   ⚠️  跳过: {skip_count}")
    print(f"   📁 总计: {success_count + fail_count + skip_count}/{total}")
    print(f"   💾 结果已保存到: {output_file}")
    print(f"{'='*70}\n")
    
    return results


def main():
    """主函数 - 支持命令行参数和默认示例"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent OCR Pipeline - 智能 OCR 处理流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行默认示例
  python case2/pipeline.py
  
  # 从 OCRBench.json 批量处理（处理前10个）
  python case2/pipeline.py --json OCRBench.json --limit 10 --output result.json
  
  # 处理单个图片
  python case2/pipeline.py --image path/to/image.jpg --query "识别文字"
  
  # 禁用最终答案生成（只做 OCR）
  python case2/pipeline.py --json OCRBench.json --no-refine --output result.json
        """
    )
    
    parser.add_argument('--json', type=str, help='从 JSON 文件批量处理（如 OCRBench.json）')
    parser.add_argument('--limit', type=int, help='限制处理数量')
    parser.add_argument('--image', type=str, help='单个图片路径')
    parser.add_argument('--query', type=str, help='查询问题（与 --image 配合使用）')
    parser.add_argument('--output', type=str, default='predictions.json', help='输出文件路径（默认: predictions.json）')
    parser.add_argument('--no-refine', action='store_true', help='禁用最终答案生成')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" " * 15 + "Multi-Agent OCR Pipeline")
    print(" " * 10 + "自动判断图片类型并调用对应的 Agent")
    print("="*70)
    print("\n工作流程：")
    print("  1. Target Detection - 自动判断图片类型（手写体/印刷体）")
    print("  2. Phi3.5-Vision - 分析图片质量和复杂度")
    print("  3. Prompt Generator - 生成结构化执行计划")
    print("  4. Agent Execution - 自动选择并执行对应的 Agent")
    if not args.no_refine:
        print("  5. Layout Selection - 选择相关布局区域（复杂场景）")
        print("  6. Answer Refinement - Phi3.5-Vision 生成最终答案")
    print("="*70)
    
    results = []
    
    # 模式 1: 从 JSON 文件批量处理
    if args.json:
        results = process_from_json(
            json_path=args.json,
            output_file=args.output,
            limit=args.limit,
            enable_refinement=not args.no_refine
        )
    
    # 模式 2: 处理单个图片
    elif args.image:
        if not args.query:
            print("❌ 错误: 使用 --image 时必须指定 --query")
            return
        
        from pathlib import Path
        image_name = Path(args.image).stem
        output_path = str(Path(args.output) / f"{image_name}_result.json")
        
        result = process_image(
            image_path=args.image,
            query=args.query,
            output_path=output_path,
            example_name=f"自定义: {image_name}",
            enable_refinement=not args.no_refine
        )
        if result:
            results.append(result)
    
    # 模式 3: 运行默认示例
    else:
        print("\n💡 运行默认示例（使用 --help 查看更多选项）\n")
    
    # 示例 1: 复杂印刷体文档
    # result1 = process_image(
    #     image_path="OCRBench_Images/docVQA/val/documents/flpp0227_16.png",
    #         query="Which company has vacancies to the post of general manager and operating engineer?",
    #     output_path="case2_output/example1_result.json",
    #         example_name="示例 1 - 复杂文档",
    #         enable_refinement=not args.no_refine
    # )
    # if result1:
    #     results.append(result1)
    
    
    # 打印摘要（对于单个图片处理）
    if results and args.image:
        print_summary(results)
    
    if args.json:
        print("\n✓ Pipeline 执行完成！")
        print(f"📁 结果已保存到: {args.output}")
    elif args.image:
        print("\n✓ Pipeline 执行完成！")
        print(f"📁 结果已保存到: {args.output}")
    print()


if __name__ == "__main__":
    main()

