# 788 条「Baseline 对、Pipeline 错」样本分析

## 一、题型与数据分布

### 1.1 按题型 (type) 分布

| 题型 | 数量 | 占比 | 说明 |
|-----|------|------|------|
| cognition VQA en | 133 | 16.9% | 认知类视觉问答 |
| reasoning VQA en | 83 | 10.5% | 推理类视觉问答 |
| key information extraction cn | 67 | 8.5% | 中文关键信息抽取（发票等） |
| VQA with position en | 47 | 6.0% | **需带 bbox/位置的 VQA** |
| APP agent en | 45 | 5.7% | 界面/APP 问答 |
| text recognition en | 45 | 5.7% | 纯文字识别 |
| chart parsing en | 40 | 5.1% | 图表解析（转 JSON/dict） |
| diagram QA en | 40 | 5.1% | 示意图/图表推理 |
| math QA en | 38 | 4.8% | 数学题 |
| document classification en | 34 | 4.3% | 文档分类 |
| ASCII art classification en | 30 | 3.8% | ASCII 艺术分类 |
| key information extraction en | 30 | 3.8% | 英文关键信息抽取 |
| science QA en | 29 | 3.7% | 科学问答 |
| 其他 | 147 | 18.7% | 多种零散题型 |

### 1.2 按数据集分布（前 10）

- rico: 52 (6.6%)
- SCID: 49 (6.2%)
- textvqa: 40 (5.1%)
- OneChart: 40 (5.1%)
- RVL-CDIP: 34 (4.3%)
- ASCII art: 30 (3.8%)
- mmsi: 29 (3.7%)
- CMMU: 26 (3.3%)
- Science QA: 23 (2.9%)
- Captured Doc: 22 (2.8%)

### 1.3 问题关键词

- what: 39.8%
- which: 12.9%
- option: 12.7%
- how many: 7.2%
- number: 6.5%
- chart / table / math: 各约 4–6%

---

## 二、Pipeline 错误类型归纳

基于 788 条样本的题目与答案形式，可归纳为以下几类（有交叉）：

### 类型 A：输出格式不符合要求（约 70+ 条显式需 bbox/json）

- **表现**：语义或内容接近对，但判分要求严格格式（如 `answer` + `bbox`、指定 JSON 结构）。
- **典型**：VQA with position（要带 bbox）、部分 key information extraction（要严格 JSON）。
- **原因**：Baseline 端到端按题生成，容易贴合格式；Pipeline 先 OCR 再 Refiner，Refiner 更偏向“自然句+内容”，格式易丢或错。

### 类型 B：OCR 引入噪声或识别错误（约 58+ 条纯文字识别）

- **表现**：baseline 直接看图识别正确，pipeline 输出错字、漏字、多字。
- **示例**：
  - 答案 `AJSTHI`，baseline 对，pipeline 输出 `AJSTH`（少字）。
  - 答案 `beer`，baseline 对，pipeline 输出 `bcer`（错字）。
- **原因**：Pipeline 依赖 OCR 结果再 Refiner；OCR 错或 Layout 切错区域，Refiner 被错误文本带偏，反而比“只看图”的 baseline 差。

### 类型 C：选错区域 / 注意力在错误位置（大量，尤其是 APP/界面/文档）

- **表现**：页面有多块文字（多 tab、多选项、多字段），pipeline 答成别的区域内容。
- **示例**：
  - 问「Which tab is selected?」答案 `VIDEOS`，pipeline 答成 `harlemshake1`（别的 tab 名）。
  - 问「What is the total number of questions?」答案 `2`，pipeline 答成 `71`（页面其他数字）。
  - 问「Which tab is currently selected?」答案 `RAMEN`，pipeline 答成 `Zaoh Restaurant`（其他区域）。
- **原因**：Layout 切出多区域后，Layout Selection 或 Refiner 更关注了“高 OCR 置信度/显眼”区域，而非与问题真正相关的区域；baseline 整图理解，更容易对准“当前选中的 tab”“题目数量”等。

### 类型 D：数值/计数/数学推理错误（约 109 条涉 number/math）

- **表现**：how many、数字、简单运算，baseline 对，pipeline 错。
- **示例**：
  - 问「how many years were there more than 6 matches?」答案 `4`，baseline 对，pipeline 答 `3`。
  - 问「How many different coloured lines?」答案 `4`，pipeline 答 `3`。
- **原因**：Pipeline 依赖 OCR 数字 + 文本摘要，容易漏数、多数或选错数字来源；图表/表格被切块后，计数和对应关系易乱。Baseline 整图推理更稳。

### 类型 E：图表/表格转结构化（约 111 条 chart/table/dict）

- **表现**：要求把图表转成嵌套 dict/JSON，baseline 结构对、字段对，pipeline 结构错或字段混淆。
- **示例**：chart parsing 题要求固定嵌套格式，pipeline 输出层级或 key 与参考答案不一致，被判错。
- **原因**：多步 OCR + 区域选择 + Refiner，容易在“谁是谁”的对应关系上出错（例如标题、轴、系列名混淆），或格式不符合严格 schema。

### 类型 F：Pipeline 输出为空或极短（约 218 条）

- **表现**：`pipeline_predict` 为空或不足 5 个字符。
- **原因**：可能 Refiner 未生成、截断、或前面步骤（如 Layout/OCR）失败导致没有有效输入给 Refiner；baseline 始终有整图，不易出现“无输出”。

### 类型 G：格式细节导致判错（如标点、空格）

- **表现**：语义对但判分严格，如电话号 `(212) 555-1212` vs `(212)555-1212`。
- **示例**：contact number 题，baseline 输出 `1-800-531-6154` 判对，pipeline 输出 `(212)555-1212` 可能因格式/多选一未匹配被判错。
- **原因**：Pipeline 从 OCR 直接抄数字，空格和括号与标准答案不完全一致。

---

## 三、为什么 Pipeline 反而比 Baseline 差？

### 3.1 信息损失与噪声

- **Baseline**：整图进 VLM，保留布局、相对位置、视觉焦点。
- **Pipeline**：  
  布局检测 → 分区域 OCR → 摘要/选区域 → Refiner。  
  每一步都会：丢失部分空间关系、引入 OCR 错/漏、或选错区域，Refiner 收到的是“不完整或带噪的文本”，容易答偏。

### 3.2 任务类型不匹配

- **适合 Pipeline 的**：长文档、多块文字、需要“先找再答”的题（如 1050 条里 baseline 错、pipeline 对）。
- **不适合 Pipeline 的**：  
  - 强格式（bbox、严格 JSON）；  
  - 强推理（计数、数学、图表对应）；  
  - 强“整图理解”（当前选中的 tab、哪个是错误选项）；  
  - 纯识别且图中文字清晰（OCR 反而引入错误）。

### 3.3 多步误差累积

- Layout 切错 → 区域不对；  
- OCR 在某块错 → 摘要里就是错的；  
- Layout Selection 选错块 → Refiner 基于错误块作答；  
- Refiner 再有一点格式/表述偏差 → 判分错。  
Baseline 只有“看图→生成”一步，没有中间步骤的累积误差。

### 3.4 输出格式与评分标准

- 很多题要求严格格式（JSON、bbox、指定 key）。  
- Baseline 更常按“题面要求”直接生成格式；Pipeline 的 Refiner 更偏“用 OCR 内容组句”，格式控制弱，易被判错。

---

## 四、典型示例（Baseline 对 / Pipeline 错）

| 类型 | 问题概要 | 正确答案 | Baseline 预测 | Pipeline 预测 |
|-----|----------|----------|----------------|----------------|
| 选错区域 | Which tab is selected? | VIDEOS | VIDEOS tab is selected | harlemshake1 |
| 选错区域 | Which tab is currently selected? | RAMEN | RAMEN tab | Zaoh Restaurant |
| 数字错 | What is the total number of questions? | 2 | 2 | 71 |
| 格式/多选 | Contact number to add coworkers? | (212) 555-1212 等 | 1-800-531-6154 | (212)555-1212 |
| 识别错 | what is written in the image? | beer | beer | bcer |
| 识别错 | what is written in the image? | AJSTHI | AJSTHIA | AJSTH |
| 推理错 | how many years >6 matches? | 4 | 3 years: 1986,1999,… | 3 |
| 推理错 | How many coloured lines? | 4 | 4 | 3 |
| 格式 | 发票代码+金额 JSON | 指定 JSON | 正确 JSON | 代码/金额错位 |
| 格式 | 带 bbox 的 VQA | answer+bbox | 正确 bbox | 缺 bbox 或错 |

---

## 五、总结与建议

### 5.1 788 条的主要共性

1. **格式敏感题**：要 bbox、严格 JSON 的题，Pipeline 更容易格式不符。  
2. **强空间/语义对应**：如“当前选中的 tab”“第 2 个错误选项”“题目数量”，Pipeline 易选错区域或数字来源。  
3. **纯识别题**：图中字清晰时，OCR 错误会拉低 Pipeline，Baseline 直接看图更稳。  
4. **图表/计数/数学**：依赖整图结构和推理的题，Pipeline 多步处理易丢信息或算错。  
5. **空输出**：约 218 条 Pipeline 输出极短或空，多为中间步骤异常或 Refiner 未正确生成。

### 5.2 改进方向（简要）

- **按题型路由**：对“纯识别、强格式、强推理”类题目考虑走 Baseline 或简化 Pipeline（少用/不用 Layout 与多块 OCR）。  
- **格式后处理**：对需 bbox/JSON 的题，在 Refiner 后加格式校验与修正。  
- **区域选择与 query 对齐**：用 query 与区域内容做更强对齐，减少“答非所问”的区域被选中。  
- **鲁棒性**：检测 Pipeline 中间步骤失败或输出过短时，回退到 Baseline。

以上分析基于 `phi_correct_pipeline_wrong_items.json` 中的 788 条样本。
