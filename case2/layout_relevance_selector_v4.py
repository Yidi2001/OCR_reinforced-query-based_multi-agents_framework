
"""
layout_relevance_selector_v4.py
- Query is read from summary JSON (user_query)
- Output is JSON (selected layouts / packs)
- Auto-calculate token budget based on image size
python case2/layout_relevance_selector_v4.py --input case2_output/example1_result_summary.json --budget auto --out evidence.json
"""

import json
import math
import re
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import sys

# 导入 token budget calculator
sys.path.insert(0, str(Path(__file__).parent))
from token_budget_calculator import TokenBudgetCalculator

def get_query_from_summary(data: Dict[str, Any]) -> str:
    if "user_query" in data and isinstance(data["user_query"], str):
        return data["user_query"].strip()
    raise ValueError("user_query not found in summary JSON")

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
    """
    使用确定性哈希函数（MD5）生成文本嵌入向量
    这样可以保证每次运行结果一致
    """
    vec = [0.0]*dim
    for w in re.findall(r"\w+", text.lower()):
        # 使用 MD5 哈希（确定性）而不是 Python 内置 hash（随机化）
        hash_val = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
        vec[hash_val % dim] += 1.0
    return vec

def is_full_ocr_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in [
        "识别图片中的所有文字",
        "识别所有文字",
        "提取全部文字",
        "all text",
        "full ocr"
    ])

def select_blocks_by_relevance(regions, query, token_budget, per_block_max_tokens=200):
    q_emb = hash_embed(query)
    scored = []
    for r in regions:
        # 只使用 text 字段来计算与 query 的相似度
        text  = r.get("text")  or ""
        normalized_text = normalize_text(text)

        # 用纯文本内容计算相似度分数
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

def select_blocks_by_reading_order(regions, token_budget, per_block_max_tokens=200):
    sorted_regions = sorted(regions, key=lambda r: (r["coordinate"][1], r["coordinate"][0]))
    packs, current, used = [], [], 0
    for r in sorted_regions:
        text = normalize_text(r.get("text",""))
        t = approx_token_len(text)
        if t > per_block_max_tokens:
            text = text[:per_block_max_tokens*4]
            t = per_block_max_tokens
        if used + t > token_budget and current:
            packs.append(current)
            current, used = [], 0
        current.append({
            "region_id": r.get("region_id"),
            "text": text,
            "bbox": r.get("coordinate"),
            "label": r.get("label"),
            "confidence": r.get("confidence"),
            "tokens": t
        })
        used += t
    if current:
        packs.append(current)
    return packs

def run_selection(summary, token_budget=900):
    query = get_query_from_summary(summary)
    regions = summary["ocr_results"]["regions"]
    
    # 保留原始 summary 中的元数据字段
    base_result = {
        "image_path": summary.get("image_path"),
        "resolution": summary.get("resolution"),
        "user_query": query,
        "classification": summary.get("classification"),
    }
    
    # 保留其他有用的元数据
    if "agent_sequence" in summary:
        base_result["agent_sequence"] = summary["agent_sequence"]
    if "total_time" in summary:
        base_result["total_time"] = summary["total_time"]
    
    # 根据查询类型添加选择结果
    if is_full_ocr_query(query):
        packs = select_blocks_by_reading_order(regions, token_budget)
        base_result.update({
            "mode": "full_ocr",
            "num_packs": len(packs),
            "packs": packs
        })
    else:
        selected = select_blocks_by_relevance(regions, query, token_budget)
        base_result.update({
            "mode": "query_relevance",
            "num_selected": len(selected),
            "blocks": selected
        })
    
    return base_result

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="根据相关性选择布局区域，支持自动计算 token 预算"
    )
    ap.add_argument("--input", required=True, help="输入的 summary JSON 文件")
    ap.add_argument("--budget", default="auto", 
                   help="Token 预算。可以是数字（如 900）或 'auto'（自动计算）")
    ap.add_argument("--out", required=True, help="输出的 JSON 文件")
    ap.add_argument("--max-output", type=int, default=2000,
                   help="预期的最大输出 token 数（用于自动计算预算）")
    ap.add_argument("--show-budget", action="store_true",
                   help="显示详细的 token 预算信息")
    args = ap.parse_args()

    # 读取 summary
    with open(args.input, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 确定 token budget
    if args.budget == "auto":
        # 自动计算
        image_path = summary.get("image_path", "")
        calculator = TokenBudgetCalculator(num_crops=4)
        
        if args.show_budget:
            calculator.print_budget_info(image_path, max_output_tokens=args.max_output)
        
        token_budget = calculator.get_text_budget(image_path, max_output_tokens=args.max_output)
        print(f"✓ 自动计算的 token 预算: {token_budget}")
    else:
        # 手动指定
        try:
            token_budget = int(args.budget)
            print(f"✓ 使用手动指定的 token 预算: {token_budget}")
        except ValueError:
            print(f"❌ 错误: --budget 必须是数字或 'auto'")
            sys.exit(1)

    # 运行选择
    result = run_selection(summary, token_budget=token_budget)

    # 保存结果
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 结果已保存到: {args.out}")
    print(f"  - 选择的区域数: {result.get('num_selected', len(result.get('packs', [])))}")
    print(f"  - Token 预算: {token_budget}")
