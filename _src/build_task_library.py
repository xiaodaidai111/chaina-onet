# -*- coding: utf-8 -*-
"""Build a deduplicated O*NET task library from Task_DWA.csv."""
import pandas as pd

SRC = r"D:/创业/职位招聘——蒋/_src/Task_DWA.csv"
OUT = r"D:/创业/职位招聘——蒋/_output/onet_tasks_dedup.csv"

df = pd.read_csv(SRC, encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]
print("raw rows:", len(df))

df["Task ID"] = df["Task ID"].astype(str).str.strip()
df["Task"] = df["Task"].astype(str).str.strip()

# A Task ID + Task pair may be tagged with multiple DWAs.
# Aggregate DWA ID / DWA Title / GWA into |-joined strings per unique task.
def agg(s):
    seen = []
    for x in s.dropna().astype(str):
        x = x.strip()
        if x not in seen and x != "nan" and x != "":
            seen.append(x)
    return "|".join(seen)

grouped = (
    df.groupby(["Task ID", "Task"], as_index=False)
    .agg({
        "O*NET-SOC Code": "first",
        "Title": "first",
        "DWA ID": agg,
        "DWA Title": agg,
        "GWA": agg,
    })
)

grouped = grouped.rename(columns={
    "O*NET-SOC Code": "onet_soc_code",
    "DWA ID": "dwa_id",
    "DWA Title": "dwa_title",
})

print("unique tasks:", len(grouped))
print("unique Task IDs:", grouped["Task ID"].nunique())

grouped.to_csv(OUT, index=False, encoding="utf-8-sig")
print("saved:", OUT)

# quick stats on field completeness
for col in ["onet_soc_code", "title", "dwa_id", "dwa_title", "GWA"]:
    if col not in grouped.columns:
        print(col, "missing column")
        continue
    print(col, "non-null:", grouped[col].notna().sum())