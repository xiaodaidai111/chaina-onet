# -*- coding: utf-8 -*-
"""
Pluggable English->Chinese translation for O*NET task statements.

Translator backends (order of preference set by --backend):
  - google  : Google free translate endpoint (no key; unofficial; used by default)
  - deepseek: DeepSeek chat API via env var DEEPSEEK_API_KEY
  - openai  : OpenAI-compatible chat API via env vars OPENAI_API_KEY / OPENAI_BASE_URL
  - file    : read previously translated task_zh from an existing onet_tasks_zh.csv

Output: onet_tasks_zh.csv with columns
  task_id, task_en, task_zh, onet_soc_code, title, dwa_id, dwa_title, gwa

Translation is resumable: completed translations are saved after every batch.
Run with --resume to continue an interrupted run.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TASK_DEDUP_CSV, ONET_TASKS_ZH_CSV  # noqa: E402

GOOGLE_URL = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=zh-CN&dt=t&q={q}"


class BaseTranslator:
    name = "base"

    def translate(self, text: str) -> str:
        raise NotImplementedError

    @classmethod
    def describe(cls) -> str:
        return cls.name


class GoogleTranslator(BaseTranslator):
    name = "google"

    def translate(self, text: str) -> str:
        q = urllib.parse.quote(text, safe="")
        url = GOOGLE_URL.format(q=q)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return "".join(seg[0] for seg in data[0]).strip()


class ChatTranslator(BaseTranslator):
    """OpenAI-compatible chat API (DeepSeek / OpenAI / other)."""

    name = "chat"

    def __init__(self, api_key_env: str, base_url: str | None = None):
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(f"env var {api_key_env} is not set")
        self.base_url = base_url
        self.model = "deepseek-chat" if "DEEPSEEK" in api_key_env else "gpt-4o-mini"
        self.name = api_key_env.replace("_API_KEY", "").lower()

    def translate(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate English O*NET job task "
                    "statements into fluent Simplified Chinese. Output ONLY the Chinese translation.",
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url or "https://api.deepseek.com/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


def make_translator(backend: str) -> BaseTranslator:
    if backend == "google":
        return GoogleTranslator()
    if backend == "deepseek":
        return ChatTranslator("DEEPSEEK_API_KEY")
    if backend == "openai":
        return ChatTranslator("OPENAI_API_KEY", os.environ.get("OPENAI_BASE_URL"))
    if backend == "file":
        return None  # handled separately
    raise ValueError(f"unknown backend: {backend}")


def translate_with_retry(translator: BaseTranslator, text: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            return translator.translate(text)
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} after {wait}s ({type(exc).__name__})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"translation failed after {max_retries} retries: {text[:60]}")


def translate_all(backend: str, workers: int, resume: bool, limit: int | None):
    df = pd.read_csv(TASK_DEDUP_CSV, encoding="utf-8-sig")
    # columns for the zh library (fixed schema)
    out_cols = ["task_id", "task_en", "task_zh", "onet_soc_code", "title", "dwa_id", "dwa_title", "gwa"]
    df = df.rename(columns={"Task ID": "task_id", "Task": "task_en", "GWA": "gwa"})
    df["task_id"] = df["task_id"].astype(str)
    if limit:
        df = df.head(limit)

    for c in out_cols:
        if c not in df.columns:
            df[c] = ""

    # resume: load previously translated rows
    if resume and ONET_TASKS_ZH_CSV.exists():
        prev = pd.read_csv(ONET_TASKS_ZH_CSV, encoding="utf-8-sig")
        prev["task_id"] = prev["task_id"].astype(str)
        zh_map = dict(zip(prev["task_id"], prev.get("task_zh", pd.Series()).fillna("")))
        done = {tid for tid, z in zh_map.items() if z}
        df["task_zh"] = df["task_id"].map(zh_map).fillna("")
        print(f"resume: {len(done)} tasks already translated")

    # fill from an existing file if available (backend=file)
    if backend == "file":
        if not ONET_TASKS_ZH_CSV.exists():
            print("ERROR: backend=file but no existing onet_tasks_zh.csv found")
            sys.exit(1)
        prev = pd.read_csv(ONET_TASKS_ZH_CSV, encoding="utf-8-sig")
        prev["task_id"] = prev["task_id"].astype(str)
        if "task_zh" not in prev.columns:
            print("ERROR: existing onet_tasks_zh.csv has no task_zh column")
            sys.exit(1)
        df = df.set_index("task_id")
        for c in out_cols:
            if c in prev.columns:
                df[c] = prev.set_index("task_id")[c].reindex(df.index)
        df = df.reset_index()
        df.to_csv(ONET_TASKS_ZH_CSV, index=False, encoding="utf-8-sig")
        print(f"file backend: filled from existing {ONET_TASKS_ZH_CSV}")
        return

    translator = make_translator(backend)
    todo = df[df["task_zh"].fillna("") == ""]
    print(f"backend={translator.name}  total={len(df)}  to_translate={len(todo)}")

    results = {}

    def work(row):
        if not row["task_en"] or pd.isna(row["task_en"]):
            return row["task_id"], ""
        return row["task_id"], translate_with_retry(translator, str(row["task_en"]))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, row): row for _, row in todo.iterrows()}
        count = 0
        for fut in as_completed(futs):
            tid, zh = fut.result()
            results[tid] = zh
            count += 1
            if count % 200 == 0:
                print(f"  translated {count}/{len(todo)}", flush=True)
                _checkpoint(df, results, out_cols)

    df.loc[df["task_id"].isin(results), "task_zh"] = df["task_id"].map(results)
    df["task_zh"] = df["task_zh"].fillna("")
    df[out_cols].to_csv(ONET_TASKS_ZH_CSV, index=False, encoding="utf-8-sig")
    filled = (df["task_zh"] != "").sum()
    print(f"saved {ONET_TASKS_ZH_CSV}: {filled}/{len(df)} tasks translated")


def _checkpoint(df: pd.DataFrame, results: dict, out_cols: list):
    tmp = df.copy()
    tmp.loc[tmp["task_id"].isin(results), "task_zh"] = tmp["task_id"].map(results)
    tmp["task_zh"] = tmp["task_zh"].fillna("")
    tmp[out_cols].to_csv(ONET_TASKS_ZH_CSV, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(description="Translate O*NET tasks to Chinese (pluggable backend)")
    ap.add_argument("--backend", default="google", choices=["google", "deepseek", "openai", "file"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="translate only first N tasks (testing)")
    args = ap.parse_args()
    translate_all(args.backend, args.workers, args.resume, args.limit)


if __name__ == "__main__":
    main()