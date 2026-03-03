#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较 phi3.5_pipeline_new_scored.json 与 test_scored.json，
输出：
1. first_0_second_1.json: 第一个 result=0，第二个 result=1
2. first_1_second_0.json: 第一个 result=1，第二个 result=0
"""
import json
from pathlib import Path

DIR = Path(__file__).parent
FILE1 = DIR / "phi3.5_pipeline_new_scored.json"
FILE2 = DIR / "test_scored.json"
OUT1 = DIR / "first_0_second_1.json"   # pipeline 错，test 对
OUT2 = DIR / "first_1_second_0.json"   # pipeline 对，test 错

def main():
    with open(FILE1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(FILE2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    by_id1 = {item["id"]: item for item in data1 if "id" in item}
    by_id2 = {item["id"]: item for item in data2 if "id" in item}
    common_ids = set(by_id1) & set(by_id2)

    first_0_second_1 = []
    first_1_second_0 = []

    for iid in common_ids:
        a, b = by_id1[iid], by_id2[iid]
        r1 = a.get("result", a.get("score", -1))
        r2 = b.get("result", b.get("score", -1))

        item_out = {**a, "test_result": r2, "test_predict": b.get("predict", "")}

        if r1 == 0 and r2 == 1:
            first_0_second_1.append(item_out)
        elif r1 == 1 and r2 == 0:
            first_1_second_0.append(item_out)

    with open(OUT1, "w", encoding="utf-8") as f:
        json.dump(first_0_second_1, f, ensure_ascii=False, indent=2)
    with open(OUT2, "w", encoding="utf-8") as f:
        json.dump(first_1_second_0, f, ensure_ascii=False, indent=2)

    print(f"first_0_second_1 (pipeline 错, test 对): {len(first_0_second_1)} → {OUT1}")
    print(f"first_1_second_0 (pipeline 对, test 错): {len(first_1_second_0)} → {OUT2}")


if __name__ == "__main__":
    main()
