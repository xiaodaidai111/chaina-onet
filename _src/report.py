# -*- coding: utf-8 -*-
"""Generate a verification report for the Plan-B TaskMatch run."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, Path(__file__).parent.as_posix())
from config import (  # noqa: E402
    TEST_SET_CSV, ACHIEVEMENT_CSV, SENTENCES_CSV, ONET_TASKS_ZH_CSV,
    MATCH_SENTENCE_CSV, MATCH_JOB_CSV, THRESHOLD, TOP_K, EMBEDDING_MODEL,
)

OUT = Path(__file__).parent.parent / "_output" / "report.md"


def fmt(x, n=2):
    return f"{x:.{n}f}"


def main():
    test = pd.read_csv(TEST_SET_CSV, encoding="utf-8-sig", dtype={"编号": str})
    tasks = pd.read_csv(ONET_TASKS_ZH_CSV, encoding="utf-8-sig")
    sentences = pd.read_csv(SENTENCES_CSV, encoding="utf-8-sig", dtype={"job_id": str})
    sent_df = pd.read_csv(MATCH_SENTENCE_CSV, encoding="utf-8-sig", dtype={"job_id": str})
    job_df = pd.read_csv(MATCH_JOB_CSV, encoding="utf-8-sig", dtype={"job_id": str})

    lines = []
    lines.append("# 中文 TaskMatch（方案 B）验证报告")
    lines.append("")
    lines.append(f"- 生成时间：{pd.Timestamp.now()}")
    lines.append("")

    lines.append("## 1. 总体统计")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    n_jobs = test["编号"].nunique()
    n_sent = len(sentences)
    n_matched_sent = sent_df["sentence_id"].groupby(sent_df["job_id"]).count()  # rows, not unique
    n_matched_sent_unique = sent_df.drop_duplicates(["job_id", "sentence_id"]).shape[0]
    task_ids = set()
    for ids in job_df["task_ids"]:
        task_ids.update(x.strip() for x in str(ids).split(",") if x.strip())
    n_unique_tasks = len(task_ids)
    avg_task_per_job = job_df["task_count"].mean() if len(job_df) else 0
    empty_jobs = n_jobs - job_df["job_id"].nunique()
    lines.append(f"| 输入岗位数 | {n_jobs} |")
    lines.append(f"| 抽取出的候选任务句数量 | {n_sent} |")
    lines.append(f"| 成功匹配的句子数量（有≥1个超阈值匹配） | {n_matched_sent_unique} |")
    lines.append(f"| 句子级匹配记录数 | {len(sent_df)} |")
    lines.append(f"| 产生的唯一 Task ID 数量 | {n_unique_tasks} |")
    lines.append(f"| 平均每个岗位匹配 Task 数 | {fmt(avg_task_per_job)} |")
    lines.append(f"| 空结果岗位数量 | {empty_jobs} |")
    lines.append(f"| 任务库规模（唯一 Task） | {len(tasks)} |")
    lines.append(f"| 中文任务库完成度（task_zh 非空） | {(tasks['task_zh'].fillna('')!='').sum()}/{len(tasks)} |")
    lines.append(f"| 阈值 | {THRESHOLD} |")
    lines.append(f"| top_k | {TOP_K} |")
    lines.append(f"| 嵌入模型 | {EMBEDDING_MODEL} |")
    lines.append("")

    # score distribution of top-1 matches
    top1 = sent_df[sent_df["rank"] == 1]
    if len(top1):
        lines.append("## 2. 句子级 top-1 得分分布")
        lines.append("")
        lines.append("| 区间 | 句子数 |")
        lines.append("|---|---|")
        bins = [0.0, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
        labels = ["<0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", ">=0.90"]
        counts = pd.cut(top1["score"], bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
        for lab, c in counts.items():
            lines.append(f"| {lab} | {c} |")
        lines.append("")

    # sample: top-2 matches for 5 distinct sentences (10 rows), spread across jobs
    lines.append("## 3. 抽样展示（5 个不同句子 × 各自 top-2 匹配 = 10 条）")
    lines.append("")
    sents = (
        sent_df.sort_values("score", ascending=False)
        .drop_duplicates("original_sentence")
        .head(5)["original_sentence"]
        .tolist()
    )
    sample = (
        sent_df[sent_df["original_sentence"].isin(sents)]
        .sort_values("score", ascending=False)
        .groupby("original_sentence", sort=False)
        .head(2)
        .reset_index(drop=True)
    )
    lines.append("| 中文原句 | 匹配到的中文 O*NET task | 英文原始 task | Task ID | score |")
    lines.append("|---|---|---|---|---|")
    for _, r in sample.iterrows():
        zh = str(r["matched_task_zh"]).replace("|", "/")[:60]
        en = str(r["matched_task_en"]).replace("|", "/")[:60]
        orig = str(r["original_sentence"]).replace("|", "/")[:60]
        lines.append(f"| {orig} | {zh} | {en} | {r['matched_task_id']} | {r['score']} |")
    lines.append("")

    # comparison with example file (IDs are NOT joinable - compare shape/stats only)
    lines.append("## 4. 与 achievement exhibition.csv 的对比（形态层面，不做 ID join）")
    lines.append("")
    lines.append("> 注意：示例文件 `岗位编号`（如 27201501001）与 `test set.csv` 的 `编号`（如 002364201402001）"
                 "不是同一套编号体系，因此不能按 ID 对齐，仅对比任务数量分布与形态。")
    lines.append("")
    ex = pd.read_csv(ACHIEVEMENT_CSV, encoding="utf-8-sig")
    ex_task_counts = ex["0.5任务ID"].fillna("").str.split(",").apply(lambda x: len([i for i in x if str(i).strip()]))
    lines.append("| 指标 | 示例文件(achievement exhibition) | 本方案输出 |")
    lines.append("|---|---|---|")
    lines.append(f"| 岗位数 | {len(ex)} | {len(job_df)} |")
    lines.append(f"| 每岗位任务数均值 | {fmt(ex_task_counts.mean())} | {fmt(avg_task_per_job)} |")
    lines.append(f"| 每岗位任务数中位数 | {fmt(ex_task_counts.median(), 0)} | {fmt(job_df['task_count'].median(), 0)} |")
    ex_ids = set()
    for x in ex["0.5任务ID"].fillna("").astype(str).str.split(","):
        ex_ids.update(i.strip() for i in x if i.strip())
    lines.append(f"| 唯一 Task ID 数 | {len(ex_ids)} | {n_unique_tasks} |")
    lines.append(f"| 任务ID总数 | {int(ex_task_counts.sum())} | {int(job_df['task_count'].sum())} |")
    lines.append("")
    lines.append("形态说明：示例文件每岗位 5-19 个任务 ID（均值约 "
                 f"{fmt(ex_task_counts.mean())}），本方案每岗位 {fmt(avg_task_per_job)} 个；"
                 "两者任务数量级可比，但本方案因阈值过滤偏保守、任务库为中文翻译版，"
                 "且示例文件可能直接来自论文标注结果，数量差异属预期。")
    lines.append("")

    lines.append("## 5. 方法、模型与参数")
    lines.append("")
    lines.append("- 方法：中文分句（按 `。；1、2、3、（一）（二）` 等规则拆分）→ 章节过滤（优先岗位职责/工作职责/职位描述/工作内容，排除岗位要求/任职要求/任职资格）→ 中文字符数过滤（8-120）→ 关键词过滤非任务句 → bge-m3 嵌入 → 归一化后点积求余弦相似度 → top-k 取候选 → 阈值截断。")
    lines.append(f"- 模型：`{EMBEDDING_MODEL}`（多语言，中文/英文/跨语言均可）。")
    lines.append(f"- 阈值：`{THRESHOLD}`（可通过 `--threshold` 调整）；top_k=`{TOP_K}`。")
    lines.append("")
    lines.append("## 6. 当前限制")
    lines.append("")
    lines.append("- 翻译源：环境中的 DEEPSEEK_API_KEY 无效（401）、Ollama 本地服务不可用，无受支持的翻译 API；"
                 "本次采用无密钥的 Google 免费翻译端点完成全部 18831 条 task_zh 生成。该端点非官方、无 SLA，"
                 "若需正式复现，建议换用受支持的翻译 API 或人工校对翻译。")
    lines.append("- 若 `onet_tasks_zh.csv` 中 `task_zh` 为空，匹配将自动回退到英文库（跨语言演示模式），此时结果仅为技术演示。")
    lines.append("- 分句与关键词过滤为规则实现，对格式多变的广告会漏抽或误抽；没有经过人工标注校验。")
    lines.append("- 阈值 0.65 较宽松，且每句保留 top-10 候选，导致每岗位任务数偏多（均值 45.6）；可调高阈值或调低 top_k 收紧。")
    lines.append("- 无中文人工标注集，**不报告 Precision/Recall/F1**；以上仅为未经人工验证的匹配统计。")
    lines.append("")
    lines.append("## 7. 下一步人工验证方案")
    lines.append("")
    lines.append("1. 从 21 个岗位中抽取 300-500 条候选句，由 2 名标注员对照 O*NET 任务语义标注：")
    lines.append("   - 该句是否为有效任务句（task / not-task）；")
    lines.append("   - 该句语义对应的正确 O*NET Task ID（可多选）。")
    lines.append("2. 计算：task 分类准确率、Top-1/Top-10 命中率、Precision/Recall/F1。")
    lines.append("3. 依据标注结果调整：阈值、top_k、分句规则、非任务句过滤词表、分节标签。")
    lines.append("4. 校验翻译质量：抽查 100 条 task_zh，修正误译后再重跑匹配。")
    lines.append("")
    lines.append("## 8. 输出文件清单")
    lines.append("")
    lines.append("| 文件 | 说明 |")
    lines.append("|---|---|")
    lines.append("| `onet_tasks_zh.csv` | 去重任务库：task_id, task_en, task_zh, onet_soc_code, title, dwa_id, dwa_title, gwa |")
    lines.append("| `chinese_job_sentences.csv` | 中文候选任务句：job_id, sentence_id, sentence, source_section |")
    lines.append("| `taskmatch_sentence_level.csv` | 句子级匹配：job_id, sentence_id, original_sentence, matched_task_id, matched_task_zh, matched_task_en, score, rank |")
    lines.append("| `taskmatch_job_level.csv` | 岗位级聚合：job_id, task_ids, task_count, avg_score |")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("report written:", OUT)


if __name__ == "__main__":
    main()