#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果摘要器：从Pipeline输出的JSON中提取关键信息
用于生成给大模型的简洁提示信息
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter


class ResultSummarizer:
    """提取Pipeline执行结果的关键信息"""
    
    def __init__(self, max_ocr_text_length: int = 500):
        """
        Args:
            max_ocr_text_length: OCR文本摘要的最大长度
        """
        self.max_ocr_text_length = max_ocr_text_length
    
    def summarize_from_file(self, json_path: str) -> Dict[str, Any]:
        """从JSON文件中提取关键信息摘要"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self.summarize(data)
    
    def summarize(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取关键信息摘要
        
        重点提取：
        1. 基本信息：图片路径、分辨率、分类结果
        2. OCR结果：如果有布局检测，提取每个区域的识别结果；否则提取整体识别结果
        
        Args:
            result_data: Pipeline输出的完整JSON数据
            
        Returns:
            关键信息字典
        """
        exec_results = result_data.get('execution_results', {})
        layout_result = exec_results.get('layout_result')
        
        summary = {
            # 基本信息
            "image_path": result_data.get('image_path', ''),
            "resolution": self._get_resolution(result_data),
            "user_query": result_data.get('query', ''),
            "classification": self._extract_classification(result_data),
            
            # OCR识别结果
            "ocr_results": self._extract_ocr_results(result_data, layout_result),
            
            # 执行信息（可选）
            "agent_sequence": self._get_agent_sequence(result_data),
            "total_time": result_data.get('total_time', 0)
        }
        
        return summary
    
    def format_as_prompt(self, summary: Dict[str, Any]) -> str:
        """
        将摘要格式化为适合大模型的提示信息
        
        Args:
            summary: 关键信息字典
            
        Returns:
            格式化的提示文本
        """
        prompt_parts = []
        
        # 基本信息
        prompt_parts.append("=" * 80)
        prompt_parts.append("图像OCR识别结果摘要")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        
        # 图像路径和分辨率
        prompt_parts.append(f"📷 图像路径: {summary['image_path']}")
        prompt_parts.append(f"📐 分辨率: {summary['resolution']}")
        prompt_parts.append("")
        
        # 用户查询
        if summary.get('user_query'):
            prompt_parts.append(f"❓ 用户查询: {summary['user_query']}")
            prompt_parts.append("")
        
        # 文本类型分类
        cls = summary['classification']
        prompt_parts.append(f"🔍 文本类型: {cls['label']} (置信度: {cls['confidence']:.1%})")
        prompt_parts.append("")
        
        # Agent执行序列
        if summary.get('agent_sequence'):
            prompt_parts.append(f"⚙️  执行序列: {' → '.join(summary['agent_sequence'])}")
            prompt_parts.append("")
        
        # OCR识别结果
        ocr_results = summary['ocr_results']
        prompt_parts.append("=" * 80)
        prompt_parts.append("📝 OCR识别结果")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        
        if ocr_results['type'] == 'layout_based':
            # 基于布局检测的结果
            is_merged = ocr_results.get('merged', False)
            if is_merged:
                merge_stats = ocr_results.get('merge_stats', {})
                original_count = merge_stats.get('original_regions', 0)
                prompt_parts.append(f"✓ 使用布局检测和智能整合")
                prompt_parts.append(f"  原始区域: {original_count} → 整合后: {ocr_results['total_regions']} 个区域")
            else:
                prompt_parts.append(f"✓ 使用布局检测，共识别 {ocr_results['total_regions']} 个区域")
            prompt_parts.append("")
            
            for region in ocr_results['regions']:
                prompt_parts.append(f"【区域 {region['region_id']}】")
                prompt_parts.append(f"  类型: {region['label']}")
                prompt_parts.append(f"  置信度: {region['confidence']}")
                prompt_parts.append(f"  坐标: {region['coordinate']}")
                
                # 如果有标题，显示标题
                if region.get('title'):
                    prompt_parts.append(f"  标题: {region['title']}")
                
                # 如果有子区域数量，显示
                if region.get('children_count'):
                    prompt_parts.append(f"  包含子区域: {region['children_count']}")
                
                prompt_parts.append(f"  识别文字:")
                if region['text']:
                    # 在提示文本中，如果文字太长，可以适当截断以便阅读
                    # 但保留前500个字符，足够查看主要内容
                    text_for_display = region['text']
                    if len(text_for_display) > 500:
                        text_for_display = text_for_display[:500] + "\n    ... (完整内容见summary.json)"
                    # 添加缩进
                    indented_text = '\n    '.join(text_for_display.split('\n'))
                    prompt_parts.append(f"    {indented_text}")
                else:
                    prompt_parts.append(f"    (未识别到文字)")
                prompt_parts.append("")
            
            if ocr_results.get('visualized_image'):
                prompt_parts.append(f"🖼️  可视化图片: {ocr_results['visualized_image']}")
                prompt_parts.append("")
        
        else:
            # 整图识别结果
            prompt_parts.append("✓ 整图识别")
            prompt_parts.append("")
            prompt_parts.append("识别文字:")
            prompt_parts.append(ocr_results['text'])
            prompt_parts.append("")
        
        # 性能信息
        if summary.get('total_time'):
            prompt_parts.append("=" * 80)
            prompt_parts.append(f"⏱️  总耗时: {summary['total_time']:.2f}秒")
        
        return "\n".join(prompt_parts)
    
    def _get_resolution(self, data: Dict) -> str:
        """获取图片分辨率"""
        quality = data.get('task_plan', {}).get('quality_analysis', {})
        return quality.get('resolution', 'unknown')
    
    def _get_agent_sequence(self, data: Dict) -> list:
        """获取Agent执行序列"""
        exec_results = data.get('execution_results', {})
        agents_executed = exec_results.get('agents_executed', [])
        return [a.get('agent_name', '') for a in agents_executed]
    
    def _extract_classification(self, data: Dict) -> Dict:
        """提取文本类型分类信息"""
        cls = data.get('classification', {})
        return {
            "label": cls.get('label', 'unknown'),
            "confidence": cls.get('confidence', 0.0),
            "probabilities": cls.get('probabilities', {})
        }
    
    def _extract_ocr_results(self, data: Dict, layout_result: Dict) -> Dict:
        """
        提取OCR识别结果
        
        如果有布局检测，提取每个区域的详细信息；
        否则提取整体识别结果。
        """
        exec_results = data.get('execution_results', {})
        
        # 检查是否使用了布局检测
        if layout_result and layout_result.get('detected_regions', 0) > 0:
            return self._extract_layout_based_ocr(layout_result, exec_results)
        else:
            return self._extract_whole_image_ocr(exec_results)
    
    def _extract_layout_based_ocr(self, layout_result: Dict, exec_results: Dict) -> Dict:
        """提取基于布局检测的OCR结果 - 完整保留原始识别内容"""
        
        # 优先使用整合后的 merged_blocks（如果存在）
        merged_blocks = layout_result.get('merged_blocks', [])
        if merged_blocks:
            return self._extract_from_merged_blocks(merged_blocks, layout_result)
        
        # 否则使用原始的 boxes
        boxes = layout_result.get('boxes', [])
        
        # 从agents_executed中获取OCR agent的输出
        agents_executed = exec_results.get('agents_executed', [])
        ocr_agent_output = None
        
        for agent in agents_executed:
            if 'OCRAgent' in agent.get('agent_name', ''):
                ocr_agent_output = agent.get('output', '')
                break
        
        # 解析OCR输出，完整提取每个区域的识别结果
        regions = []
        
        if ocr_agent_output and isinstance(ocr_agent_output, str):
            # OCR输出格式：区域 X: label (置信度: Y)\n位置: [coords]\n文字内容:\n...
            # 按分隔符分割
            region_blocks = ocr_agent_output.split('------------------------------------------------------------')
            
            # 第一个块包含标题和区域1，需要特殊处理
            # 其余块依次对应区域2, 3, 4...
            
            for i, box in enumerate(boxes):
                block = None
                
                if i == 0 and region_blocks:
                    # 区域1在第一个块中
                    block = region_blocks[0]
                elif i < len(region_blocks):
                    # 其他区域在对应的块中（注意索引偏移）
                    block = region_blocks[i]
                
                if not block:
                    continue
                
                # 完整提取文字内容（不做任何删减）
                text_content = self._extract_full_text_from_block(block)
                
                region_info = {
                    "region_id": i + 1,
                    "label": box.get('label', 'unknown'),
                    "confidence": round(box.get('score', 0), 3),
                    "coordinate": box.get('coordinate', []),
                    "text": text_content
                }
                regions.append(region_info)
        
        return {
            "type": "layout_based",
            "total_regions": layout_result.get('detected_regions', 0),
            "regions": regions,
            "visualized_image": layout_result.get('visualized_image_path', '')
        }
    
    def _extract_whole_image_ocr(self, exec_results: Dict) -> Dict:
        """提取整图OCR结果（无布局检测）"""
        final_output = exec_results.get('final_output', '')
        
        return {
            "type": "whole_image",
            "text": final_output[:self.max_ocr_text_length] if len(final_output) > self.max_ocr_text_length else final_output
        }
    
    def _extract_full_text_from_block(self, block: str) -> str:
        """
        从OCR输出块中完整提取文字内容
        保留原始识别结果的所有内容，不做任何删减
        """
        if not block:
            return ""
        
        # 查找"文字内容:"标记
        marker = "文字内容:"
        if marker not in block:
            marker = "文字内容："
        
        if marker not in block:
            # 如果没有"文字内容:"标记，返回空
            return ""
        
        # 提取"文字内容:"之后的所有内容
        parts = block.split(marker, 1)
        if len(parts) < 2:
            return ""
        
        # 获取文字内容部分，去除首尾空白
        text_content = parts[1].strip()
        
        return text_content
    
    def _extract_from_merged_blocks(self, merged_blocks: list, layout_result: Dict) -> Dict:
        """
        从整合后的 merged_blocks 中提取信息
        
        merged_blocks 格式:
        [
          {
            "block_id": 1,
            "bbox": [x1, y1, x2, y2],
            "title": "标题文字",
            "labels": ["text", "paragraph_title"],
            "text": "整合后的完整文字",
            "children": [...]
          }
        ]
        
        Args:
            merged_blocks: 整合后的块列表
            layout_result: 原始布局结果（用于获取可视化图片路径）
            
        Returns:
            OCR结果字典
        """
        regions = []
        
        for block in merged_blocks:
            block_id = block.get('block_id')
            bbox = block.get('bbox', [])
            title = block.get('title', '')
            labels = block.get('labels', [])
            text = block.get('text', '')
            children_count = len(block.get('children', []))
            
            # 计算平均置信度（从children中）
            children = block.get('children', [])
            if children:
                avg_confidence = sum(c.get('confidence', 0) for c in children) / len(children)
            else:
                avg_confidence = 0.0
            
            # 构建区域信息
            region_info = {
                "region_id": block_id,
                "label": ", ".join(labels) if labels else "merged_block",
                "confidence": round(avg_confidence, 3),
                "coordinate": bbox,
                "text": text,
                "title": title,
                "children_count": children_count
            }
            
            regions.append(region_info)
        
        return {
            "type": "layout_based",
            "total_regions": len(merged_blocks),
            "regions": regions,
            "visualized_image": layout_result.get('visualized_image_path', ''),
            "merged": True,  # 标记这是整合后的结果
            "merge_stats": layout_result.get('merge_stats', {})
        }
    
    def _extract_text_from_block(self, block: str) -> str:
        """
        从OCR输出块中提取纯文字内容
        只提取实际识别的文字，跳过所有解释性内容
        """
        if not block:
            return ""
        
        lines = block.split('\n')
        text_lines = []
        in_text_section = False
        stop_extraction = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # 识别文字内容开始标记
            if '文字内容' in line_stripped:
                in_text_section = True
                continue
            
            # 遇到这些标题就停止提取（都是解释性内容）
            if any(stop_word in line_stripped for stop_word in [
                '**中文翻译', '中文翻译：', '**翻译',
                '**识别要点', '识别要点：', 
                '**识别结果显示', '识别结果显示：',
                '**识别说明', '识别说明：',
                '**说明：', '说明：',
                '这是一个', '这看起来', '从识别结果', '从内容可以看出',
                '您可以', '需要注意', '如果您'
            ]):
                stop_extraction = True
                break
            
            # 跳过元数据行
            if any(skip in line_stripped for skip in [
                '区域', '位置:', '置信度:', '识别结果如下', '根据OCR', '根据识别'
            ]):
                continue
            
            # 提取实际文字（在文字内容section中，且未停止）
            if in_text_section and not stop_extraction and line_stripped:
                # 跳过空行、分隔线、列表标记行
                if (line_stripped.startswith('---') or 
                    line_stripped.startswith('```') or
                    line_stripped == '**' or
                    (line_stripped.startswith('-') and len(line_stripped) < 50) or
                    (line_stripped[0].isdigit() and '. ' in line_stripped[:3])):  # 跳过编号列表
                    continue
                
                # 移除markdown格式，保留文字
                clean_line = line_stripped.replace('**', '').replace('```', '').strip()
                
                # 移除常见的前缀
                for prefix in ['识别结果显示图片中的文字是', '识别结果显示图片中只有一个单词：', 
                               '识别结果显示图片中', '识别结果只有一个单词：', '识别结果：',
                               '识别结果显示', '识别结果如下：']:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix):].strip()
                        # 移除可能存在的引号
                        clean_line = clean_line.strip('"').strip("'").strip()
                        break
                
                # 截断解释性后缀（如"，置信度为..."）
                for suffix_marker in ['，置信度', '。置信度', '，表示', '。表示', '，这是', '，意思是']:
                    if suffix_marker in clean_line:
                        clean_line = clean_line.split(suffix_marker)[0].strip()
                        break
                
                # 再次清理可能残留的引号
                clean_line = clean_line.strip('"').strip("'").strip()
                
                if clean_line:
                    text_lines.append(clean_line)
                    
                    # 限制提取的行数（前3-5行通常是主要内容）
                    if len(text_lines) >= 5:
                        break
        
        # 合并文字
        if not text_lines:
            return ""
        
        full_text = ' '.join(text_lines)
        
        # 限制长度
        if len(full_text) > 200:
            full_text = full_text[:200] + '...'
        
        return full_text
    
    
    def save_summary(self, summary: Dict, output_path: str):
        """保存摘要到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"✓ 摘要已保存到: {output_path}")


def test_summarizer():
    """测试摘要器"""
    # 测试文件
    test_file = "case2_output/example1_result.json"
    
    if not Path(test_file).exists():
        print(f"测试文件不存在: {test_file}")
        return
    
    print("=" * 80)
    print("测试 ResultSummarizer")
    print("=" * 80)
    
    # 创建摘要器
    summarizer = ResultSummarizer(max_ocr_text_length=800)
    
    # 提取摘要
    print("\n1. 提取关键信息...")
    summary = summarizer.summarize_from_file(test_file)
    
    # 保存摘要JSON
    summary_json_path = "case2_output/example1_summary.json"
    summarizer.save_summary(summary, summary_json_path)
    
    # 格式化为提示文本
    print("\n2. 格式化为大模型提示文本...")
    prompt = summarizer.format_as_prompt(summary)
    
    # 保存提示文本
    prompt_txt_path = "case2_output/example1_prompt.txt"
    with open(prompt_txt_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"✓ 提示文本已保存到: {prompt_txt_path}")
    
    # 打印提示文本
    print("\n" + "=" * 80)
    print("生成的提示文本:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    # 统计信息
    print(f"\n摘要统计:")
    print(f"- 原始JSON大小: {Path(test_file).stat().st_size / 1024:.2f} KB")
    print(f"- 摘要JSON大小: {Path(summary_json_path).stat().st_size / 1024:.2f} KB")
    print(f"- 提示文本大小: {Path(prompt_txt_path).stat().st_size / 1024:.2f} KB")
    print(f"- 压缩比例: {Path(summary_json_path).stat().st_size / Path(test_file).stat().st_size * 100:.1f}%")


if __name__ == "__main__":
    test_summarizer()

