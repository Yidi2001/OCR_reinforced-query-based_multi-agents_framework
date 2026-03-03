#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验表格（硕士论文用，中文）
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 中文字体
_font_paths = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
]
def get_chinese_font():
    for path in _font_paths:
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
    try:
        prop = fm.FontProperties(fname=chinese_font)
        plt.rcParams['font.sans-serif'] = [prop.get_name()]
    except Exception:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 消融实验数据
configs = [
    'phi-pipeline（完整）',
    'no-gate',
    'no-gate，no-字体检测模块',
    'no-gate，no-相关性检测模块',
    'no-gate，no-prompt generator',
]
scores = [611, 561, 542, 540, 545]

# 画表
fig, ax = plt.subplots(figsize=(7, 2.8), dpi=200)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')

table = ax.table(
    cellText=[[c, str(s)] for c, s in zip(configs, scores)],
    colLabels=['配置', '得分'],
    loc='center',
    cellLoc='center',
    colColours=['#e5e7eb', '#e5e7eb'],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# 表头加粗、整体边框风格
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('#d1d5db')
    if i == 0:
        cell.set_text_props(fontweight='bold')
    cell.set_facecolor('white' if i > 0 else '#f3f4f6')

plt.tight_layout()
out_path = 'xiaorong/ablation_table.png'
plt.savefig(out_path, bbox_inches='tight', dpi=200, facecolor='white')
print(f'Saved: {out_path}')
plt.close()
