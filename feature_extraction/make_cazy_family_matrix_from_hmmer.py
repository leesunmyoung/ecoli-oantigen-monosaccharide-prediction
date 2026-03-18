#!/usr/bin/env python3
from pathlib import Path
import re
from collections import defaultdict

# -----------------------------
# Paths (프로젝트 구조에 맞춤)
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.resolve()

STATS_FILE = PROJECT_DIR / "results" / "ecoli_subtype_stats.tsv"
DBCAN_BASE = PROJECT_DIR / "results" / "dbcan"
OUT_FILE = PROJECT_DIR / "results" / "ecoli_cazy_family_count_matrix.tsv"

# CAZy family 패턴 (GT2, GH13, CE4, PL1, CBM50 등)
FAM_RE = re.compile(r"\b(GT|GH|CE|PL|CBM)\d+\b", re.IGNORECASE)

def extract_family(hmm_name: str):
    """
    HMM 이름에서 CAZy family 토큰을 추출합니다.
    예: 'GT2.hmm' -> 'GT2', 'GH13_1' -> 'GH13'
    """
    s = hmm_name.replace(".hmm", "").replace(".HMM", "")
    m = FAM_RE.search(s)
    if not m:
        return None
    return m.group(0).upper()

def parse_hmmer_out(hmmer_path: Path):
    """
    dbCAN hmmer.out(대개 hmmscan domtblout 형식)을 파싱하여
    family별로 hit된 query protein set을 반환합니다.
    """
    fam_to_queries = defaultdict(set)

    with open(hmmer_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.rstrip("\n").split()
            if len(parts) < 4:
                continue

            # domtblout 기준:
            # parts[0] = target(hmm), parts[3] = query(protein)
            hmm = parts[0]
            query = parts[3]

            fam = extract_family(hmm)
            if fam:
                fam_to_queries[fam].add(query)

    return fam_to_queries

def main():
    if not STATS_FILE.exists():
        raise FileNotFoundError(f"Not found: {STATS_FILE}")
    if not DBCAN_BASE.exists():
        raise FileNotFoundError(f"Not found: {DBCAN_BASE}")

    # 1) stats에서 subtype-acc 읽기
    subtype_acc = []
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        i_subtype = header.index("subtype")
        i_acc = header.index("selected_accession")

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            subtype = parts[i_subtype]
            acc = parts[i_acc]
            if not acc or acc == "NA":
                continue
            subtype_acc.append((subtype, acc))

    # 2) 각 accession의 hmmer.out을 읽어서 family 카운트 계산
    all_families = set()
    row_counts = []  # (subtype, acc, counts_dict)

    processed = 0
    skipped = 0

    for subtype, acc in subtype_acc:
        hmmer_path = DBCAN_BASE / acc / "hmmer.out"
        if not hmmer_path.exists():
            skipped += 1
            continue

        fam_to_queries = parse_hmmer_out(hmmer_path)
        counts = {fam: len(qset) for fam, qset in fam_to_queries.items()}

        all_families.update(counts.keys())
        row_counts.append((subtype, acc, counts))
        processed += 1

    # 3) 매트릭스 출력 (rows=subtype, cols=family)
    families_sorted = sorted(all_families, key=lambda x: (re.sub(r"\d+", "", x), int(re.search(r"\d+", x).group())))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        out.write("subtype\tselected_accession")
        for fam in families_sorted:
            out.write(f"\t{fam}")
        out.write("\n")

        for subtype, acc, counts in row_counts:
            out.write(f"{subtype}\t{acc}")
            for fam in families_sorted:
                out.write(f"\t{counts.get(fam, 0)}")
            out.write("\n")

    print(f"[DONE] matrix rows={processed}, skipped(no hmmer.out)={skipped}")
    print(f"[DONE] families={len(families_sorted)}")
    print(f"[OUT]  {OUT_FILE}")

if __name__ == "__main__":
    main()
