# -*- coding: utf-8 -*-
"""Shared configuration for the Chinese TaskMatch pipeline."""
from pathlib import Path

BASE = Path(r"D:/创业/职位招聘——蒋")
SRC = BASE / "_src"
OUT = BASE / "_output"

# ---- inputs ----
TEST_SET_CSV = BASE / "test set.csv"
ACHIEVEMENT_CSV = BASE / "achievement exhibition.csv"
TASK_DWA_CSV = SRC / "Task_DWA.csv"

# ---- task library ----
TASK_DEDUP_CSV = OUT / "onet_tasks_dedup.csv"
ONET_TASKS_ZH_CSV = OUT / "onet_tasks_zh.csv"

# ---- sentence extraction ----
SENTENCES_CSV = OUT / "chinese_job_sentences.csv"

# ---- matching ----
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL_CACHE = SRC / "models"
MATCH_SENTENCE_CSV = OUT / "taskmatch_sentence_level.csv"
MATCH_JOB_CSV = OUT / "taskmatch_job_level.csv"

# ---- parameters ----
MIN_CHAR_LEN = 8       # minimum Chinese character count for a candidate sentence
MAX_CHAR_LEN = 120     # maximum Chinese character count for a candidate sentence
TOP_K = 10             # top-k O*NET tasks per sentence
THRESHOLD = 0.65       # similarity threshold (configurable via CLI)
BATCH_SIZE = 32        # embedding batch size

# sections that are preferred task sources
TASK_SECTIONS = ["岗位职责", "工作职责", "职位描述", "工作内容", "主要职责", "职责描述", "职位概要"]
# sections that are usually NOT tasks (requirements, qualifications, benefits)
NON_TASK_SECTIONS = ["岗位要求", "任职要求", "任职资格", "职位要求", "福利待遇", "薪资待遇", "岗位要求及福利", "岗位描述及要求"]

# noise patterns to strip from raw text
NOISE_PATTERNS = [
    "更多数据，详见马克数据网",
    "来源：百度搜索马克数据网",
    "来自马-克-数-据-官网",
    "关注公众号马克数据网",
    "马 克 数 据 网",
    "马克数据网",
    "马 克 团 队",
    "来源：马克团队",
    "来源：马 克 团 队",
]