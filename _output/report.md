# 中文 TaskMatch（方案 B）验证报告

- 生成时间：2026-08-20 18:05:39.762701

## 1. 总体统计

| 指标 | 数值 |
|---|---|
| 输入岗位数 | 21 |
| 抽取出的候选任务句数量 | 110 |
| 成功匹配的句子数量（有≥1个超阈值匹配） | 110 |
| 句子级匹配记录数 | 1097 |
| 产生的唯一 Task ID 数量 | 201 |
| 平均每个岗位匹配 Task 数 | 45.57 |
| 空结果岗位数量 | 0 |
| 任务库规模（唯一 Task） | 18831 |
| 中文任务库完成度（task_zh 非空） | 18831/18831 |
| 阈值 | 0.65 |
| top_k | 10 |
| 嵌入模型 | BAAI/bge-m3 |

## 2. 句子级 top-1 得分分布

| 区间 | 句子数 |
|---|---|
| <0.60 | 0 |
| 0.60-0.65 | 0 |
| 0.65-0.70 | 19 |
| 0.70-0.75 | 41 |
| 0.75-0.80 | 28 |
| 0.80-0.85 | 3 |
| 0.85-0.90 | 19 |
| >=0.90 | 0 |

## 3. 抽样展示（5 个不同句子 × 各自 top-2 匹配 = 10 条）

| 中文原句 | 匹配到的中文 O*NET task | 英文原始 task | Task ID | score |
|---|---|---|---|---|
| 与客户/前方销售人员接触，了解并确定客户需求，完成业务需求和相关资料的收集 | 与销售团队合作，了解客户需求，促进公司产品的销售，并提供销售支持。 | Collaborate with sales teams to understand customer requirem | 9680 | 0.8654 |
| 与客户/前方销售人员接触，了解并确定客户需求，完成业务需求和相关资料的收集 | 与客户、销售人员或营销人员沟通以确定客户需求。 | Communicate with customers, sales staff, or marketing staff  | 18957 | 0.8618 |
| 负责项目现场的组织和管理工作 | 管理和指导项目现场的施工、运营或维护活动。 | Manage and direct the construction, operations, or maintenan | 20490 | 0.8572 |
| 负责项目现场的组织和管理工作 | 管理和指导项目现场的施工、运营或维护活动。 | Manage and direct the construction, operations, or maintenan | 20490 | 0.8572 |
| 组织和指导本区域销售人员做好营销策略和营销计划 | 制定销售区域的营销或战略计划。 | Develop marketing or strategic plans for sales territories. | 17660 | 0.8372 |
| 制定所辖区域或产品方向的销售团队建设和人员发展计划，依照销售管理办法和销售政策，制定并执行本区域销售人员的培训、工作考核 | 规划和指导人员配置、培训和绩效评估，以制定和控制销售和服务计划。 | Plan and direct staffing, training, and performance evaluati | 11 | 0.8348 |
| 组织和指导本区域销售人员做好营销策略和营销计划 | 指导营销或销售人员的招聘、培训或绩效评估，并监督他们的日常活动。 | Direct the hiring, training, or performance evaluations of m | 954 | 0.8169 |
| 制定所辖区域或产品方向的销售团队建设和人员发展计划，依照销售管理办法和销售政策，制定并执行本区域销售人员的培训、工作考核 | 指导营销或销售人员的招聘、培训或绩效评估，并监督他们的日常活动。 | Direct the hiring, training, or performance evaluations of m | 954 | 0.8112 |
| 负责与系统开发、测试等小组沟通，确保各个协作小组对系统设计文档的充分准确理解 | 与其他工作人员协调和咨询，以设计、布局或详细说明组件和系统，并解决设计或其他问题。 | Coordinate with and consult other workers to design, lay out | 1448 | 0.8077 |
| 负责与系统开发、测试等小组沟通，确保各个协作小组对系统设计文档的充分准确理解 | 指导开发团队之间的设计讨论。 | Guide design discussions between development teams. | 16207 | 0.7949 |

## 4. 与 achievement exhibition.csv 的对比（形态层面，不做 ID join）

> 注意：示例文件 `岗位编号`（如 27201501001）与 `test set.csv` 的 `编号`（如 002364201402001）不是同一套编号体系，因此不能按 ID 对齐，仅对比任务数量分布与形态。

| 指标 | 示例文件(achievement exhibition) | 本方案输出 |
|---|---|---|
| 岗位数 | 21 | 21 |
| 每岗位任务数均值 | 10.24 | 45.57 |
| 每岗位任务数中位数 | 10 | 44 |
| 唯一 Task ID 数 | 136 | 201 |
| 任务ID总数 | 215 | 957 |

形态说明：示例文件每岗位 5-19 个任务 ID（均值约 10.24），本方案每岗位 45.57 个；两者任务数量级可比，但本方案因阈值过滤偏保守、任务库为中文翻译版，且示例文件可能直接来自论文标注结果，数量差异属预期。

## 5. 方法、模型与参数

- 方法：中文分句（按 `。；1、2、3、（一）（二）` 等规则拆分）→ 章节过滤（优先岗位职责/工作职责/职位描述/工作内容，排除岗位要求/任职要求/任职资格）→ 中文字符数过滤（8-120）→ 关键词过滤非任务句 → bge-m3 嵌入 → 归一化后点积求余弦相似度 → top-k 取候选 → 阈值截断。
- 模型：`BAAI/bge-m3`（多语言，中文/英文/跨语言均可）。
- 阈值：`0.65`（可通过 `--threshold` 调整）；top_k=`10`。

## 6. 当前限制

- 翻译源：环境中的 DEEPSEEK_API_KEY 无效（401）、Ollama 本地服务不可用，无受支持的翻译 API；本次采用无密钥的 Google 免费翻译端点完成全部 18831 条 task_zh 生成。该端点非官方、无 SLA，若需正式复现，建议换用受支持的翻译 API 或人工校对翻译。
- 若 `onet_tasks_zh.csv` 中 `task_zh` 为空，匹配将自动回退到英文库（跨语言演示模式），此时结果仅为技术演示。
- 分句与关键词过滤为规则实现，对格式多变的广告会漏抽或误抽；没有经过人工标注校验。
- 阈值 0.65 较宽松，且每句保留 top-10 候选，导致每岗位任务数偏多（均值 45.6）；可调高阈值或调低 top_k 收紧。
- 无中文人工标注集，**不报告 Precision/Recall/F1**；以上仅为未经人工验证的匹配统计。

## 7. 下一步人工验证方案

1. 从 21 个岗位中抽取 300-500 条候选句，由 2 名标注员对照 O*NET 任务语义标注：
   - 该句是否为有效任务句（task / not-task）；
   - 该句语义对应的正确 O*NET Task ID（可多选）。
2. 计算：task 分类准确率、Top-1/Top-10 命中率、Precision/Recall/F1。
3. 依据标注结果调整：阈值、top_k、分句规则、非任务句过滤词表、分节标签。
4. 校验翻译质量：抽查 100 条 task_zh，修正误译后再重跑匹配。

## 8. 输出文件清单

| 文件 | 说明 |
|---|---|
| `onet_tasks_zh.csv` | 去重任务库：task_id, task_en, task_zh, onet_soc_code, title, dwa_id, dwa_title, gwa |
| `chinese_job_sentences.csv` | 中文候选任务句：job_id, sentence_id, sentence, source_section |
| `taskmatch_sentence_level.csv` | 句子级匹配：job_id, sentence_id, original_sentence, matched_task_id, matched_task_zh, matched_task_en, score, rank |
| `taskmatch_job_level.csv` | 岗位级聚合：job_id, task_ids, task_count, avg_score |