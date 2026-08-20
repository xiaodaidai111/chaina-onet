# -*- coding: utf-8 -*-
"""
Chinese job-ad sentence extraction for Plan-B TaskMatch.

Pipeline per ad:
  1. clean noise (HTML tags, boilerplate, duplicated punctuation, spaces)
  2. split into sections by Chinese section headers
  3. keep preferred task sections; drop non-task sections
  4. split each kept section into candidate sentences (Chinese-aware)
  5. filter by Chinese-character count (8..120) and non-task keyword rules

Output: chinese_job_sentences.csv  (job_id, sentence_id, sentence, source_section)
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, Path(__file__).parent.as_posix())
from config import TEST_SET_CSV, SENTENCES_CSV, TASK_SECTIONS, NON_TASK_SECTIONS, NOISE_PATTERNS, MIN_CHAR_LEN, MAX_CHAR_LEN  # noqa: E402

CJK = re.compile(r"[\u4e00-\u9fff]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ENT_RE = re.compile(r"&[a-zA-Z#0-9]+;")
DUP_PUNCT_RE = re.compile(r"([。；;，,!！？?：:、])\1+")
SPACE_RE = re.compile(r"\s+")

SECTION_RE = re.compile(
    "(" + "|".join(sorted(TASK_SECTIONS + NON_TASK_SECTIONS, key=len, reverse=True)) + r")\s*[:：]?"
)
# numbered / phase list markers that begin a new candidate sentence
LIST_MARKER_RE = re.compile(r"(?=[（(]?(?:\d+|[一二三四五六七八九十]+)[）)]?[、.．)）]|第一阶段|第二阶段|第三阶段|第四阶段|第[一二三四五六七八九十]+阶段)")
SENT_END_RE = re.compile(r"[。；;！？!]")
LEADING_MARKER_RE = re.compile(r"^(?:[（(]?(?:\d+|[一二三四五六七八九十]+)[）)]?[、.．)）]|第[一二三四五六七八九十]+阶段)[:：]?")

# non-task / requirement phrase filters (candidate sentence level)
DROP_RE = [
    re.compile(r"优先"),
    re.compile(r"熟悉"),
    re.compile(r"熟练(掌握|使用|运用)"),
    re.compile(r"精通"),
    re.compile(r"沟通能力"),
    re.compile(r"表达能力"),
    re.compile(r"协作能力"),
    re.compile(r"写作能力"),
    re.compile(r"团队精神"),
    re.compile(r"团队意识"),
    re.compile(r"责任心"),
    re.compile(r"事业心"),
    re.compile(r"工作激情"),
    re.compile(r"适应(?:经常)?出差"),
    re.compile(r"出差"),
    re.compile(r"学历"),
    re.compile(r"本科|硕士|博士|研究生|大专"),
    re.compile(r"毕业"),
    re.compile(r"证书"),
    re.compile(r"资格证"),
    re.compile(r"年龄"),
    re.compile(r"性别"),
    re.compile(r"福利"),
    re.compile(r"待遇"),
    re.compile(r"薪资|工资|薪酬"),
    re.compile(r"五险一金"),
    re.compile(r"社会保险"),
    re.compile(r"培训机会|晋升|上升空间|发展前景"),
    re.compile(r"汇报对象"),
    re.compile(r"有(?:过)?\d+\s*年.*经验"),
    re.compile(r"经验者"),
    re.compile(r"专业(?:背景|对口|相关)"),
]


def clean_text(raw: str) -> str:
    text = str(raw)
    for noise in NOISE_PATTERNS:
        text = text.replace(noise, "")
    text = HTML_ENT_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = DUP_PUNCT_RE.sub(r"\1", text)
    text = text.replace("\u3000", " ")
    text = SPACE_RE.sub(" ", text)
    text = text.replace("(五险一金", "五险一金").strip(" 。;；,，、:：")
    return text.strip()


def split_sections(text: str):
    """Return list of (section_label_or_None, segment_text)."""
    parts = SECTION_RE.split(text)
    # parts[0] is preamble, then alternating label, text
    sections = [(None, parts[0])]
    i = 1
    while i + 1 < len(parts):
        sections.append((parts[i], parts[i + 1]))
        i += 2
    if i < len(parts):
        sections.append((parts[i], ""))
    return sections


def split_sentences(segment: str):
    """Split a section segment into candidate sentences."""
    # split on sentence-ending punctuation first
    chunks = [c for c in SENT_END_RE.split(segment) if c and c.strip()]
    out = []
    for chunk in chunks:
        # then split on numbered / phase list markers
        pieces = LIST_MARKER_RE.split(chunk)
        for piece in pieces:
            piece = LEADING_MARKER_RE.sub("", piece.strip())
            piece = piece.strip(" 。;；,，、:：")
            if piece:
                out.append(piece)
    return out


def is_task_like(s: str) -> bool:
    n = len(CJK.findall(s))
    if n < MIN_CHAR_LEN or n > MAX_CHAR_LEN:
        return False
    for pat in DROP_RE:
        if pat.search(s):
            return False
    return True


def extract_all():
    df = pd.read_csv(TEST_SET_CSV, encoding="utf-8-sig", dtype={"编号": str})
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"编号": "job_id", "内容": "raw_text"})
    df["job_id"] = df["job_id"].astype(str).str.strip()
    df["raw_text"] = df["raw_text"].fillna("").astype(str)

    rows = []
    entered_non_task = False
    for job_id, raw in zip(df["job_id"], df["raw_text"]):
        text = clean_text(raw)
        sections = split_sections(text)
        entered_non_task = False
        sid = 0
        for label, seg in sections:
            seg = seg.strip(" 。;；,，、:：")
            if not seg:
                continue
            if label is None:
                # preamble or unknown segment: keep only before any non-task section
                if entered_non_task:
                    continue
                src = "未标注"
            elif label in NON_TASK_SECTIONS:
                entered_non_task = True
                continue
            else:
                src = label
            for s in split_sentences(seg):
                s = s.strip(" 。;；,，、:：")
                if not is_task_like(s):
                    continue
                sid += 1
                rows.append((job_id, sid, s, src))

    out = pd.DataFrame(rows, columns=["job_id", "sentence_id", "sentence", "source_section"])
    out.to_csv(SENTENCES_CSV, index=False, encoding="utf-8-sig")
    print(f"jobs={df['job_id'].nunique()}  candidate_sentences={len(out)}")
    print("saved:", SENTENCES_CSV)
    print(out["source_section"].value_counts().to_string())


if __name__ == "__main__":
    extract_all()