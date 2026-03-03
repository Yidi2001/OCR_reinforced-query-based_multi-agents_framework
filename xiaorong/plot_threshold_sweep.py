#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
门控阈值超参数实验折线图（硕士论文用，中文）
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 数据
thresholds = [0, 0.4, 0.5, 0.6, 1.0]
scores = [561, 611, 606, 592, 542]

# 中文字体：优先用系统 Noto CJK，否则按名称在 matplotlib 缓存里找
_font_paths = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
]
def get_chinese_font():
    for path in _font_paths:
        import os
        if os.path.isfile(path):
            return path
    for name in ['Noto Sans CJK SC', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK JP']:
        for f in fm.fontManager.ttflist:
            if name.lower() in f.name.lower():
                return f.fname
    return None

chinese_font = get_chinese_font()
if chinese_font:
    fm.fontManager.addfont(chinese_font)
    # 用路径加载时，需取 font 的 name 用于 rcParams
    try:
        prop = fm.FontProperties(fname=chinese_font)
        font_name = prop.get_name()
        plt.rcParams['font.sans-serif'] = [font_name]
    except Exception:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 折线：蓝色实线 + 实心三角标记（参照论文图风格）
ax.plot(thresholds, scores, '^-', color='#2563eb', linewidth=2, markersize=8,
        markerfacecolor='#2563eb', markeredgecolor='#2563eb')

# 中文标签
ax.set_xlabel('K', fontsize=12)
ax.set_ylabel('得分', fontsize=12)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(530, 630)
ax.set_xticks(thresholds)
ax.tick_params(axis='both', labelsize=10)
# 浅灰虚线网格
ax.grid(True, linestyle='--', color='#d1d5db', alpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 每个数据点上方标注数值（与参考图一致）
for x, y in zip(thresholds, scores):
    ax.annotate(str(y), xy=(x, y), xytext=(x, y + 12),
                ha='center', fontsize=10, color='#1e3a8a')

plt.tight_layout()
out_path = 'xiaorong/threshold_sweep_curve.png'
plt.savefig(out_path, bbox_inches='tight', dpi=200, facecolor='white')
print(f'Saved: {out_path}')
plt.close()
