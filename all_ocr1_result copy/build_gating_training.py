#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 phi3.5_pipeline_new_scored.json (pipeline) 与 test_scored.json (baseline) 构建门控训练集：

1. gating_label0_baseline.json: 走 baseline
   - pipeline 错、baseline 对 (125)
   - 两个都对
   label=0

2. gating_label1_pipeline.json: 走 pipeline
   - pipeline 对、baseline 错 (78)
   - 两个都错
   label=1
"""
import json
from pathlib import Path

DIR = Path(__file__).parent
FILE1 = DIR / "phi3.5_pipeline_new_scored.json"  # pipeline
FILE2 = DIR / "test_scored.json"                # baseline
OUT_BASELINE = DIR / "gating_label0_baseline.json"
OUT_PIPELINE = DIR / "gating_label1_pipeline.json"


def main():
    with open(FILE1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(FILE2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    # 两文件顺序一致，按 index 匹配以覆盖全部（id 可能重复）
    n = min(len(data1), len(data2))

    label0_items = []  # 走 baseline
    label1_items = []  # 走 pipeline

    for i in range(n):
        a, b = data1[i], data2[i]
        r1 = a.get("result", a.get("score", -1))
        r2 = b.get("result", b.get("score", -1))

        # 使用 pipeline 的 item 作为主结构，补充 baseline 的 predict（若需要）
        item_base = {k: v for k, v in a.items() if k != "test_result" and k != "test_predict"}
        item_base["pipeline_result"] = r1
        item_base["baseline_result"] = r2

        if r1 == 0 and r2 == 1:
            # pipeline 错、baseline 对 → 走 baseline
            item_base["gating_label"] = 0
            label0_items.append(item_base)
        elif r1 == 1 and r2 == 0:
            # pipeline 对、baseline 错 → 走 pipeline
            item_base["gating_label"] = 1
            label1_items.append(item_base)
        elif r1 == 1 and r2 == 1:
            # 两个都对 → 走 baseline
            item_base["gating_label"] = 0
            label0_items.append(item_base)
        elif r1 == 0 and r2 == 0:
            # 两个都错 → 走 pipeline
            item_base["gating_label"] = 1
            label1_items.append(item_base)

    with open(OUT_BASELINE, "w", encoding="utf-8") as f:
        json.dump(label0_items, f, ensure_ascii=False, indent=2)
    with open(OUT_PIPELINE, "w", encoding="utf-8") as f:
        json.dump(label1_items, f, ensure_ascii=False, indent=2)

    both_right = sum(1 for i in label0_items if i["pipeline_result"] == 1 and i["baseline_result"] == 1)
    both_wrong = sum(1 for i in label1_items if i["pipeline_result"] == 0 and i["baseline_result"] == 0)
    p_wrong_b_right = sum(1 for i in label0_items if i["pipeline_result"] == 0 and i["baseline_result"] == 1)
    p_right_b_wrong = sum(1 for i in label1_items if i["pipeline_result"] == 1 and i["baseline_result"] == 0)

    print(f"gating_label0_baseline.json ({len(label0_items)} 条) → {OUT_BASELINE}")
    print(f"   - pipeline错baseline对: {p_wrong_b_right}")
    print(f"   - 两个都对: {both_right}")
    print(f"gating_label1_pipeline.json ({len(label1_items)} 条) → {OUT_PIPELINE}")
    print(f"   - pipeline对baseline错: {p_right_b_wrong}")
    print(f"   - 两个都错: {both_wrong}")


if __name__ == "__main__":
    main()
