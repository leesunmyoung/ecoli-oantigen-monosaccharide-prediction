#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from collections import Counter

def split_hits(x):
    if pd.isna(x) or x in ["", "-"]:
        return []
    for sep in ["+", ",", ";"]:
        if sep in x:
            return [i.split("(")[0] for i in x.split(sep)]
    return [x.split("(")[0]]

def parse(path):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    cols = [c for c in df.columns if "hmmer" in c.lower() or "diamond" in c.lower()]
    fams = []
    for _, r in df.iterrows():
        for c in cols:
            fams += split_hits(r[c])
    return Counter(fams)

rows = []
root = Path("results/dbcan_runs")

for d in root.iterdir():
    ov = d / "dbcanoverview.txt"
    if not ov.exists():
        continue
    c = parse(ov)
    row = {"accession": d.name}
    row.update(c)
    rows.append(row)

df = pd.DataFrame(rows).fillna(0)
df.to_csv("results/dbcan_feature_table.tsv", sep="\t", index=False)

print("DONE", df.shape)
