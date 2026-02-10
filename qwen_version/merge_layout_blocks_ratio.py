# merge_layout_blocks_ratio_dedup_v2.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import re
from difflib import SequenceMatcher


# -----------------------------
# Geometry
# -----------------------------
def area(b: List[float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def intersect(a: List[float], b: List[float]) -> List[float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return [max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)]

def iou(a: List[float], b: List[float]) -> float:
    inter = area(intersect(a, b))
    if inter <= 0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0

def contain_ratio(parent: List[float], child: List[float]) -> float:
    inter = area(intersect(parent, child))
    ca = area(child)
    return 0.0 if ca <= 0 else inter / ca

def union_bbox(boxes: List[List[float]]) -> List[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]

def box_h(b: List[float]) -> float:
    return max(0.0, b[3] - b[1])

def box_w(b: List[float]) -> float:
    return max(0.0, b[2] - b[0])

def x_overlap_ratio(a: List[float], b: List[float]) -> float:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    denom = max(1.0, min(ax2 - ax1, bx2 - bx1))
    return inter / denom


# -----------------------------
# Text normalization & similarity
# -----------------------------
NOISE_PATTERNS = [
    r"未检测到任何印刷体文字",
    r"^\s*[-_—–]+\s*$",
    r"^\s*\W+\s*$",
]
def norm_line(s: str) -> str:
    s = norm_text(s).lower()
    s = s.replace("：", ":").replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def dedup_lines(text: str, sim_thr: float = 0.92) -> str:
    """
    Deduplicate near-duplicate lines after merging.
    """
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln and not is_noise(ln)]

    kept = []
    kept_norm = []
    for ln in lines:
        nln = norm_line(ln)
        dup = False
        for kn in kept_norm:
            if SequenceMatcher(None, nln, kn).ratio() >= sim_thr:
                dup = True
                break
        if not dup:
            kept.append(ln)
            kept_norm.append(nln)
    return "\n".join(kept)

def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def is_noise(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    for p in NOISE_PATTERNS:
        if re.search(p, s):
            return True
    if len(norm_text(s)) <= 2:
        return True
    return False

def seq_sim(a: str, b: str) -> float:
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def collapse_for_containment(s: str) -> str:
    """
    用于“文本包含”判断：去空白/标点差异/断词符号，让超集-子集更容易匹配
    """
    s = (s or "").lower()
    s = s.replace("：", ":").replace("（", "(").replace("）", ")")
    # 去掉连字符断行造成的差异，例如 "con-\nfidential" -> "confidential"
    s = re.sub(r"-\s*\n\s*", "", s)
    # 统一换行/空白
    s = re.sub(r"\s+", "", s)
    # 去掉常见无意义符号（保留字母数字）
    s = re.sub(r"[^0-9a-z]+", "", s)
    return s

_token_re = re.compile(r"[0-9]+|[a-zA-Z]+")

def token_set(s: str) -> set:
    """
    用 token 集合做“覆盖率”判断（比 SequenceMatcher 更适合超集/子集）
    """
    s = (s or "").lower()
    # 保留数字、英文单词（对你这份英文OCR很有效；中文也不会出错，只是token少）
    return set(_token_re.findall(s))

def coverage_score(big: str, small: str) -> float:
    """
    small 的 token 有多少被 big 覆盖： |T(small) ∩ T(big)| / |T(small)|
    """
    Ts = token_set(small)
    if not Ts:
        return 0.0
    Tb = token_set(big)
    return len(Ts & Tb) / max(1, len(Ts))

def text_contains(big: str, small: str, cov_thr: float = 0.92) -> bool:
    """
    判定 big 是否“包含/覆盖” small：
    1) collapsed substring（强）
    2) token coverage（稳健，抗OCR噪声）
    """
    nb = collapse_for_containment(big)
    ns = collapse_for_containment(small)
    if ns and nb and ns in nb:
        return True
    return coverage_score(big, small) >= cov_thr


# -----------------------------
# Data
# -----------------------------
@dataclass
class Region:
    region_id: int
    label: str
    confidence: float
    coordinate: List[float]
    text: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Region":
        return Region(
            region_id=int(d.get("region_id", -1)),
            label=str(d.get("label", "")),
            confidence=float(d.get("confidence", 0.0)),
            coordinate=[float(x) for x in d.get("coordinate", [0, 0, 0, 0])],
            text=str(d.get("text", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# Reading order
# -----------------------------
def median(vals: List[float], default: float = 12.0) -> float:
    vals = [v for v in vals if v > 0]
    if not vals:
        return default
    vals.sort()
    return vals[len(vals) // 2]

def reading_order(regs: List[Region]) -> List[Region]:
    if not regs:
        return []
    line_h = median([box_h(r.coordinate) for r in regs], 12.0)
    y_thr = 0.6 * line_h
    regs = sorted(regs, key=lambda r: (r.coordinate[1], r.coordinate[0]))
    lines: List[List[Region]] = []
    for r in regs:
        if not lines:
            lines.append([r])
            continue
        if abs(r.coordinate[1] - lines[-1][0].coordinate[1]) <= y_thr:
            lines[-1].append(r)
        else:
            lines.append([r])

    out: List[Region] = []
    for ln in lines:
        out.extend(sorted(ln, key=lambda r: r.coordinate[0]))
    return out


# -----------------------------
# Step 1: region-level dedup (near-identical boxes)
# -----------------------------
def dedup_regions(regions: List[Region], iou_thr: float = 0.90, sim_thr: float = 0.95) -> List[Region]:
    regions = sorted(regions, key=lambda r: (-r.confidence, -len(norm_text(r.text))))
    kept: List[Region] = []
    for r in regions:
        dup = False
        for k in kept:
            if iou(r.coordinate, k.coordinate) >= iou_thr:
                if r.label == k.label or seq_sim(r.text, k.text) >= sim_thr:
                    dup = True
                    break
        if not dup:
            kept.append(r)
    return reading_order(kept)


# -----------------------------
# Step 2: merge text lines into paragraphs (ratio thresholds)
# -----------------------------
def merge_text_lines_ratio(
    regions: List[Region],
    page_w: float,
    label_whitelist: Tuple[str, ...] = ("text",),
    x_align_ratio: float = 0.03,
    y_gap_ratio: float = 0.9,
) -> List[Region]:
    regs = reading_order(regions)
    if not regs:
        return []

    line_h = median([box_h(r.coordinate) for r in regs if r.label in label_whitelist], 12.0)
    y_gap_thr = y_gap_ratio * line_h
    x_align_thr = x_align_ratio * page_w

    out: List[Region] = []
    cur: Optional[Region] = None

    for r in regs:
        if r.label not in label_whitelist:
            if cur is not None:
                out.append(cur)
                cur = None
            out.append(r)
            continue

        if cur is None:
            cur = r
            continue

        gap = r.coordinate[1] - cur.coordinate[3]
        left_ok = abs(r.coordinate[0] - cur.coordinate[0]) <= x_align_thr

        if 0 <= gap <= y_gap_thr and left_ok:
            cur.text = (cur.text.rstrip() + "\n" + r.text.lstrip()).strip()
            cur.coordinate = union_bbox([cur.coordinate, r.coordinate])
            cur.confidence = max(cur.confidence, r.confidence)
        else:
            out.append(cur)
            cur = r

    if cur is not None:
        out.append(cur)

    return reading_order(out)


# -----------------------------
# Step 3: anchor-based block merge (ratio thresholds)
# -----------------------------
ANCHOR_LABELS = ("paragraph_title", "header")
BLOCK_MEMBER_LABELS = ("text", "paragraph_title", "header", "number", "footer")

def merge_blocks_by_anchor_ratio(
    regions: List[Region],
    page_h: float,
    block_gap_ratio: float = 0.10,
    min_x_overlap: float = 0.25,
) -> List[Dict[str, Any]]:
    regs = [r for r in regions if r.label in BLOCK_MEMBER_LABELS]
    regs = reading_order(regs)
    max_down_gap = block_gap_ratio * page_h

    anchors = [r for r in regs if r.label in ANCHOR_LABELS]
    used = set()
    blocks: List[Dict[str, Any]] = []
    bid = 1

    for a in anchors:
        if a.region_id in used:
            continue
        group = [a]
        used.add(a.region_id)
        _, ay1, _, ay2 = a.coordinate

        for r in regs:
            if r.region_id in used:
                continue
            _, ry1, _, ry2 = r.coordinate
            if ry2 < ay1:
                continue
            if ry1 - ay2 > max_down_gap:
                continue
            if x_overlap_ratio(a.coordinate, r.coordinate) < min_x_overlap:
                continue
            group.append(r)
            used.add(r.region_id)

        group = reading_order(group)
        bbox = union_bbox([g.coordinate for g in group])
        labels = sorted({g.label for g in group})

        title = norm_text(a.text)[:200] if norm_text(a.text) else None

        blocks.append({
            "block_id": bid,
            "bbox": bbox,
            "title": title,
            "labels": list(labels),
            "children": [g.to_dict() for g in group],
            "text": "",  # later rebuild
        })
        bid += 1

    # leftover as singleton blocks
    for r in regs:
        if r.region_id in used:
            continue
        blocks.append({
            "block_id": bid,
            "bbox": r.coordinate,
            "title": norm_text(r.text)[:200] if r.label in ANCHOR_LABELS else None,
            "labels": [r.label],
            "children": [r.to_dict()],
            "text": "",
        })
        bid += 1

    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return blocks


# -----------------------------
# Post-dedup: remove "super-set merged boxes" inside each block
# -----------------------------
def child_rank(c: Dict[str, Any]) -> Tuple[float, float, int]:
    """
    Keep preference:
      - higher confidence
      - smaller area (more atomic)
      - longer text
    """
    conf = float(c.get("confidence", 0.0))
    a = area(c.get("coordinate", [0, 0, 0, 0]))
    tlen = len(norm_text(c.get("text", "")))
    return (conf, -a, tlen)

def dedup_children_superset(
    children: List[Dict[str, Any]],
    contain_thr: float = 0.80,
    iou_thr: float = 0.30,
    cov_thr: float = 0.92,
) -> List[Dict[str, Any]]:
    """
    核心：删除“拼接超集框”
    若 A bbox 包含 B（或IoU较高），且 A 文本包含 B 文本（substring/coverage），则判定 A 是冗余超集。
    处理策略：
      - 若 A 覆盖了 >=2 个子框内容（常见拼接框），优先删 A
      - 否则按 child_rank 保留更优者
    """
    # 清理空/噪声
    cs = []
    for c in children:
        t = c.get("text", "")
        if not norm_text(t) or is_noise(t):
            continue
        cs.append(c)

    n = len(cs)
    if n <= 1:
        return cs

    # 预计算
    coords = [c.get("coordinate", [0, 0, 0, 0]) for c in cs]
    texts = [c.get("text", "") for c in cs]

    # 标记要删的 index
    remove = set()

    # 统计每个框“包含了多少别人的文本”（用于识别拼接框）
    covers_count = [0] * n

    for i in range(n):
        if i in remove:
            continue
        for j in range(n):
            if i == j or j in remove:
                continue

            bi, bj = coords[i], coords[j]
            ti, tj = texts[i], texts[j]

            spatial_close = (iou(bi, bj) > iou_thr) or (contain_ratio(bi, bj) > contain_thr) or (contain_ratio(bj, bi) > contain_thr)
            if not spatial_close:
                continue

            # 文本包含/覆盖（超集-子集）
            if text_contains(ti, tj, cov_thr=cov_thr):
                # i 覆盖 j
                covers_count[i] += 1
            if text_contains(tj, ti, cov_thr=cov_thr):
                covers_count[j] += 1

    # 第二遍：真正删谁
    for i in range(n):
        if i in remove:
            continue
        for j in range(i + 1, n):
            if j in remove:
                continue

            bi, bj = coords[i], coords[j]
            ti, tj = texts[i], texts[j]

            spatial_close = (iou(bi, bj) > iou_thr) or (contain_ratio(bi, bj) > contain_thr) or (contain_ratio(bj, bi) > contain_thr)
            if not spatial_close:
                continue

            i_contains_j = text_contains(ti, tj, cov_thr=cov_thr)
            j_contains_i = text_contains(tj, ti, cov_thr=cov_thr)

            if not (i_contains_j or j_contains_i):
                continue

            # 规则：如果某个是“拼接超集”（覆盖>=2个），优先删它
            if i_contains_j and covers_count[i] >= 2 and covers_count[j] < 2:
                remove.add(i)
                continue
            if j_contains_i and covers_count[j] >= 2 and covers_count[i] < 2:
                remove.add(j)
                continue

            # 否则按 rank 选一个保留（更高置信度、更原子）
            ri = child_rank(cs[i])
            rj = child_rank(cs[j])

            # 如果 i 是 j 的超集：默认删超集（更大、更冗余）
            if i_contains_j and not j_contains_i:
                # 如果超集反而明显更可信且子集很差，可保留超集（保守）
                if ri >= rj and cs[i].get("confidence", 0) > cs[j].get("confidence", 0) + 0.25:
                    remove.add(j)
                else:
                    remove.add(i)
            elif j_contains_i and not i_contains_j:
                if rj >= ri and cs[j].get("confidence", 0) > cs[i].get("confidence", 0) + 0.25:
                    remove.add(i)
                else:
                    remove.add(j)
            else:
                # 两边互相“包含”（很少见，通常是非常相似），保留 rank 高的
                if ri >= rj:
                    remove.add(j)
                else:
                    remove.add(i)

    kept = [cs[i] for i in range(n) if i not in remove]
    kept.sort(key=lambda r: (r["coordinate"][1], r["coordinate"][0], r["coordinate"][3], r["coordinate"][2]))
    return kept


def rebuild_block(block: Dict[str, Any]) -> Dict[str, Any]:
    children = block.get("children", [])
    if not children:
        block["text"] = ""
        return block

    block["bbox"] = union_bbox([c["coordinate"] for c in children])
    block["labels"] = sorted(list({c.get("label", "") for c in children if c.get("label", "")}))

    # title：取第一个 anchor 的 text
    title = None
    for c in children:
        if c.get("label") in ("paragraph_title", "header"):
            title = norm_text(c.get("text", ""))[:200] or None
            break
    block["title"] = title if title is not None else block.get("title")

    # text：用 children 逐条拼接（每个 child 作为一个段），然后做“段落级去重”
    segments = [norm_text(c.get("text", "")) for c in children if norm_text(c.get("text", ""))]
    # 段落去重：如果某段被另一段覆盖，则删掉被覆盖段（或删超集，下面更偏“删超集”已在 children 去重做了）
    # 这里再补一层：删掉“被覆盖的重复段”
    kept_segments: List[str] = []
    for seg in segments:
        redundant = False
        for kept in kept_segments:
            if text_contains(kept, seg, cov_thr=0.92) and len(seg) > 20:
                redundant = True
                break
        if not redundant:
            kept_segments.append(seg)

    merged = "\n".join(kept_segments)
    # 行级再去一次（对电话号码重复等有用）
    block["text"] = dedup_lines(merged, sim_thr=0.92)
    return block


def cross_block_dedup(
    blocks: List[Dict[str, Any]],
    contain_thr: float = 0.90,
    cov_thr: float = 0.92,
) -> List[Dict[str, Any]]:
    """
    跨 block 去掉“被更大 block 包含的无意义小块”
    例如你的 block3: "Contact:" 被 block2 的 bbox 覆盖，且文本也被 block2 覆盖 => 删除 block3
    """
    keep = [True] * len(blocks)

    texts = [b.get("text", "") for b in blocks]
    bboxes = [b.get("bbox", [0, 0, 0, 0]) for b in blocks]

    for i in range(len(blocks)):
        if not keep[i]:
            continue
        for j in range(len(blocks)):
            if i == j or not keep[i]:
                continue
            # 如果 i 是小块，j 是大块
            if area(bboxes[j]) <= area(bboxes[i]):
                continue
            # 空间包含
            if contain_ratio(bboxes[j], bboxes[i]) < contain_thr:
                continue
            # 文本包含
            if text_contains(texts[j], texts[i], cov_thr=cov_thr):
                keep[i] = False
                break

    out = [blocks[i] for i in range(len(blocks)) if keep[i] and norm_text(blocks[i].get("text", ""))]
    out.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    for k, b in enumerate(out, 1):
        b["block_id"] = k
    return out


def postprocess_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in blocks:
        b["children"] = dedup_children_superset(
            b.get("children", []),
            contain_thr=0.80,
            iou_thr=0.30,
            cov_thr=0.92,
        )
        b = rebuild_block(b)
        if norm_text(b.get("text", "")):
            out.append(b)

    out.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    for i, b in enumerate(out, 1):
        b["block_id"] = i

    # 跨块再去重（干掉像 Contact: 这种“悬空小块”）
    out = cross_block_dedup(out, contain_thr=0.90, cov_thr=0.92)
    return out


# -----------------------------
# Main pipeline
# -----------------------------
def merge_layout_regions_ratio_dedup_v2(doc: Dict[str, Any]) -> Dict[str, Any]:
    regions_raw = doc.get("ocr_results", {}).get("regions", [])
    regions = [Region.from_dict(r) for r in regions_raw]

    # page size
    res = doc.get("resolution", "")
    if isinstance(res, str) and "x" in res:
        w_s, h_s = res.lower().split("x", 1)
        page_w, page_h = float(w_s), float(h_s)
    else:
        max_x = max((r.coordinate[2] for r in regions), default=1.0)
        max_y = max((r.coordinate[3] for r in regions), default=1.0)
        page_w, page_h = max_x, max_y

    # drop noise regions
    regions = [r for r in regions if not is_noise(r.text)]

    # region dedup
    regions = dedup_regions(regions, iou_thr=0.90, sim_thr=0.95)

    # merge lines
    regions = merge_text_lines_ratio(
        regions,
        page_w=page_w,
        label_whitelist=("text",),
        x_align_ratio=0.03,
        y_gap_ratio=0.9,
    )

    # merge blocks
    blocks = merge_blocks_by_anchor_ratio(
        regions,
        page_h=page_h,
        block_gap_ratio=0.10,
        min_x_overlap=0.25,
    )

    # dedup (children + block text + cross-block)
    blocks = postprocess_blocks(blocks)

    out = dict(doc)
    out.setdefault("ocr_results", {})
    out["ocr_results"]["merged_blocks"] = blocks
    out["ocr_results"]["merged_total_blocks"] = len(blocks)
    out["ocr_results"]["merge_params"] = {
        "page_w": page_w,
        "page_h": page_h,
        "x_align_ratio": 0.03,
        "y_gap_ratio": 0.9,
        "block_gap_ratio": 0.10,
        "min_x_overlap": 0.25,
        "dedup_child_contain_thr": 0.80,
        "dedup_child_iou_thr": 0.30,
        "dedup_text_cov_thr": 0.92,
        "cross_block_contain_thr": 0.90,
    }
    return out


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 3:
        print("Usage: python merge_layout_blocks_ratio_dedup_v2.py input.json output.json")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        doc = json.load(f)

    merged = merge_layout_regions_ratio_dedup_v2(doc)

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("Done. merged_total_blocks =", merged["ocr_results"]["merged_total_blocks"])
