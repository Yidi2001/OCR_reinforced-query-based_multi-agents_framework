#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 HAE_result_merged.json 判分：判断每条记录的 predict 与 answers 是否意思一致。
- answers 可能是字符串或字符串列表（多标准答案）。
- 规范化比较：去除首尾空白、忽略大小写，一致则判对。
"""

import json
import argparse
from pathlib import Path


def normalize(s, case_insensitive=True):
    """规范化：去空白；默认转小写以判断意思一致。"""
    if s is None:
        return ""
    s = str(s).strip()
    return s.lower() if case_insensitive else s


def get_answer_list(answers):
    """将 answers 统一为列表。"""
    if answers is None:
        return []
    if isinstance(answers, list):
        return [str(a).strip() for a in answers]
    return [str(answers).strip()]


def is_consistent(predict, answers, case_insensitive=True):
    """
    判断 predict 与 answers 是否意思一致。
    一致：规范化后的 predict 与任一规范化后的 answer 相同。
    """
    pred_norm = normalize(predict, case_insensitive)
    ans_list = get_answer_list(answers)
    if not ans_list:
        return pred_norm == ""
    for a in ans_list:
        if normalize(a, case_insensitive) == pred_norm:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="判分：predict 与 answers 是否一致")
    parser.add_argument("input", nargs="?", default="HAE_result_merged.json", help="输入的合并结果 JSON")
    parser.add_argument("-o", "--output", default="", help="可选：输出带 score 字段的 JSON 路径")
    parser.add_argument("--no-lower", action="store_true", help="不忽略大小写（严格字符串匹配）")
    args = parser.parse_args()

    if args.no_lower:
        def normalize(s):
            if s is None:
                return ""
            return str(s).strip()

    path = Path(args.input)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        data = [data]

    total = 0
    correct = 0
    by_type = {}

    for item in data:
        pred = item.get("predict", "")
        ans = item.get("answers", item.get("answer", ""))
        match = is_consistent(pred, ans)
        item["_score"] = 1 if match else 0
        total += 1
        if match:
            correct += 1
        t = item.get("type", "unknown")
        by_type.setdefault(t, {"total": 0, "correct": 0})
        by_type[t]["total"] += 1
        if match:
            by_type[t]["correct"] += 1

    acc = correct / total if total else 0
    print("=" * 60)
    print("判分结果：predict 与 answers 是否意思一致")
    print("=" * 60)
    print(f"总条数: {total}")
    print(f"一致数: {correct}")
    print(f"一致率: {acc:.2%} ({correct}/{total})")
    print()
    print("按 type 统计:")
    for t, v in sorted(by_type.items(), key=lambda x: -x[1]["total"]):
        a = v["correct"] / v["total"] if v["total"] else 0
        print(f"  {t}: {v['correct']}/{v['total']} = {a:.2%}")
    print("=" * 60)

    if args.output:
        for item in data:
            item["score"] = item.pop("_score", 0)
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = Path(__file__).resolve().parent / out_path
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"已写入带 score 的结果: {out_path}")


if __name__ == "__main__":
    main()
