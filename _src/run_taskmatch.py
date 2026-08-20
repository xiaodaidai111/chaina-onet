# -*- coding: utf-8 -*-
"""
Plan-B TaskMatch: Chinese job sentences vs O*NET tasks via embedding similarity.

  - library:  BAAI/bge-m3 (multilingual); embedded side is task_zh when fully
              populated, otherwise task_en (cross-lingual demo fallback).
  - queries:  candidate Chinese sentences from chinese_job_sentences.csv
  - scoring:  cosine similarity via dot product on L2-normalized embeddings
  - top_k:    keep top-k matches per sentence, then apply threshold

Outputs:
  taskmatch_sentence_level.csv  (job_id, sentence_id, original_sentence,
                                 matched_task_id, matched_task_zh, matched_task_en,
                                 score, rank)
  taskmatch_job_level.csv       (job_id, task_ids, task_count, avg_score)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, Path(__file__).parent.as_posix())
from config import ONET_TASKS_ZH_CSV, SENTENCES_CSV, MATCH_SENTENCE_CSV, MATCH_JOB_CSV, EMBEDDING_MODEL, EMBEDDING_MODEL_CACHE, TOP_K, THRESHOLD, BATCH_SIZE  # noqa: E402


def load_model(name, cache_dir):
    from sentence_transformers import SentenceTransformer
    if name and Path(name).is_dir():
        print(f"loading embedding model from local dir {name} ...", flush=True)
        return SentenceTransformer(name)
    print(f"loading embedding model {name} ...", flush=True)
    model = SentenceTransformer(name, cache_folder=str(cache_dir))
    return model


def embed_batch(model, texts, batch_size):
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    return vecs.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--model", default=None, help="model name or local dir; defaults to local cache dir if present")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--force", action="store_true", help="re-embed library even if cached")
    ap.add_argument("--tasks-csv", default=str(ONET_TASKS_ZH_CSV))
    ap.add_argument("--sentences-csv", default=str(SENTENCES_CSV))
    args = ap.parse_args()

    if not args.model:
        local_dir = Path(EMBEDDING_MODEL_CACHE)
        if (local_dir / "config.json").exists() and (local_dir / "pytorch_model.bin").exists():
            args.model = str(local_dir)
            print(f"using local embedding model dir: {local_dir}")
        else:
            args.model = EMBEDDING_MODEL

    tasks = pd.read_csv(args.tasks_csv, encoding="utf-8-sig")
    tasks["task_id"] = tasks["task_id"].astype(str)
    tasks["task_en"] = tasks["task_en"].fillna("").astype(str)
    tasks["task_zh"] = tasks["task_zh"].fillna("").astype(str)
    tasks["title"] = tasks["title"].fillna("").astype(str)
    tasks["onet_soc_code"] = tasks["onet_soc_code"].fillna("").astype(str)

    filled_zh = (tasks["task_zh"] != "").sum()
    use_zh = filled_zh == len(tasks)
    lib_texts = tasks["task_zh"].tolist() if use_zh else tasks["task_en"].tolist()
    print(f"task library rows={len(tasks)}  task_zh filled={filled_zh}/{len(tasks)}")
    if not use_zh:
        print("WARNING: task_zh is incomplete -> using task_en as the library "
              "(cross-lingual demo mode). Provide a complete onet_tasks_zh.csv "
              "for proper Chinese-vs-Chinese Plan-B matching.")

    cache_key = "zh" if use_zh else "en"
    cache_file = Path(EMBEDDING_MODEL_CACHE) / f"task_emb_{cache_key}.npy"
    if cache_file.exists() and not args.force:
        task_vec = np.load(cache_file)
        if task_vec.shape[0] != len(tasks):
            print(f"cached embeddings stale ({task_vec.shape[0]} != {len(tasks)}), re-embedding")
            task_vec = None
        else:
            print(f"loaded cached library embeddings: {cache_file}")
    else:
        task_vec = None
    if task_vec is None:
        model = load_model(args.model, EMBEDDING_MODEL_CACHE)
        task_vec = embed_batch(model, lib_texts, args.batch_size)
        Path(EMBEDDING_MODEL_CACHE).mkdir(parents=True, exist_ok=True)
        np.save(cache_file, task_vec)
        print(f"library embeddings cached: {cache_file}")

    sentences = pd.read_csv(args.sentences_csv, encoding="utf-8-sig")
    sentences["job_id"] = sentences["job_id"].astype(str)
    print(f"candidate sentences: {len(sentences)}")

    # cache query embeddings
    query_file = Path(EMBEDDING_MODEL_CACHE) / "query_emb.npy"
    if query_file.exists() and not args.force:
        q_vec = np.load(query_file)
        if q_vec.shape[0] != len(sentences):
            print(f"cached query embeddings stale ({q_vec.shape[0]} != {len(sentences)}), re-embedding")
            q_vec = None
        else:
            print(f"loaded cached query embeddings: {query_file}")
    else:
        q_vec = None
    if q_vec is None:
        model = load_model(args.model, EMBEDDING_MODEL_CACHE)
        q_vec = embed_batch(model, sentences["sentence"].tolist(), args.batch_size)
        np.save(query_file, q_vec)
        print(f"query embeddings cached: {query_file}")

    # cosine similarity via dot product on normalized vectors
    sim = q_vec @ task_vec.T  # (n_sentences, n_tasks)

    sent_rows = []
    n_matched_sentences = 0
    for i, row in sentences.iterrows():
        scores = sim[i]
        top_idx = np.argsort(-scores)[: args.top_k]
        matched = [(idx, float(scores[idx])) for idx in top_idx if float(scores[idx]) >= args.threshold]
        if matched:
            n_matched_sentences += 1
        for rank, (idx, score) in enumerate(matched, start=1):
            t = tasks.iloc[idx]
            sent_rows.append({
                "job_id": row["job_id"],
                "sentence_id": int(row["sentence_id"]),
                "original_sentence": row["sentence"],
                "matched_task_id": str(t["task_id"]),
                "matched_task_zh": t["task_zh"] or t["task_en"],
                "matched_task_en": t["task_en"],
                "score": round(score, 4),
                "rank": rank,
            })

    sent_df = pd.DataFrame(sent_rows)
    sent_df.to_csv(MATCH_SENTENCE_CSV, index=False, encoding="utf-8-sig")
    print(f"sentence-level rows: {len(sent_df)}  matched sentences: {n_matched_sentences}/{len(sentences)}")

    # job-level aggregation
    job_rows = []
    for job_id, grp in sent_df.groupby("job_id", sort=False):
        # keep best rank per task id, order by score desc
        best = grp.sort_values("score", ascending=False).drop_duplicates("matched_task_id")
        task_ids = ",".join(best["matched_task_id"].tolist())
        task_count = len(best)
        avg_score = round(float(grp["score"].mean()), 4)
        job_rows.append({"job_id": job_id, "task_ids": task_ids, "task_count": task_count, "avg_score": avg_score})

    job_df = pd.DataFrame(job_rows)
    job_df.to_csv(MATCH_JOB_CSV, index=False, encoding="utf-8-sig")
    print(f"job-level rows: {len(job_df)}")

    # quick stats
    all_task_ids = set()
    for ids in job_df["task_ids"]:
        all_task_ids.update(str(x).strip() for x in ids.split(",") if str(x).strip())
    print(f"unique matched Task IDs: {len(all_task_ids)}")
    print(f"avg tasks per job: {job_df['task_count'].mean():.2f}")
    empty_jobs = (job_df["task_ids"].fillna("") == "").sum() + (len(sentences["job_id"].unique()) - len(job_df))
    print(f"jobs with empty results: {empty_jobs}/{sentences['job_id'].nunique()}")


if __name__ == "__main__":
    main()