# -*- coding: utf-8 -*-
"""Download Task_DWA.csv from the JAAT GitHub repository."""
import requests

URL = "https://raw.githubusercontent.com/Job-Ad-Research-at-QSB-LUC/JAAT/main/JAAT/data/Task_DWA.csv"
OUT = r"D:/创业/职位招聘——蒋/_src/Task_DWA.csv"

r = requests.get(URL, timeout=180)
print("status:", r.status_code, "bytes:", len(r.content))
if r.status_code == 200:
    with open(OUT, "wb") as f:
        f.write(r.content)
    print("saved to:", OUT)