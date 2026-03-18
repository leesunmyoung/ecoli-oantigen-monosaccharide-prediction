#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import re

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.resolve()

STATS_FILE = PROJECT_DIR / "results" / "ecoli_subtype_stats.tsv"
OLOC_DIR   = PROJECT_DIR / "results" / "oantigen_locus"
DBCAN_DIR  = PROJECT_DIR / "results" / "dbcan"

OUT_TSV    = PROJECT_DIR / "results" / "oantigen_locus_feature_table.tsv"

# ----- dbCAN family pattern -----
FAM_RE = re.compile(r"\b(GT|GH|CE|PL|CBM)\d+\b", re.IGNORECASE)

# ----- key genes from product -----
KEYGENES = ["wzx", "wzy", "wzm", "wzt"]

# ----- product tokenization -----
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP = {
    "protein","hypothetical","putative","family","domain","like","subunit",
    "enzyme","biosynthesis","biosynthetic","pathway","uncharacterized",
    "probable","predicted","associated","component","membrane","transport",
    "system","dependent","related","possible","unknown"
}

def load_subtype_to_acc() -> dict[str, str]:
    if not STATS_FILE.exists():
        raise FileNotFoundError(f"Not found: {STATS_FILE}")

    with open(STATS_FILE, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        i_sub = header.index("subtype")
        i_acc = header.index("selected_accession")

        m = {}
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            sub = parts[i_sub]
            acc = parts[i_acc]
            if acc and acc != "NA":
                m[sub] = acc
        return m

def find_one(patterns: list[str], root: Path) -> Path | None:
    for pat in patterns:
        hits = list(root.rglob(pat))
        if hits:
            return hits[0]
    return None

def load_oantigen_gene_tables() -> dict[str, Path]:
    """
    returns: subtype -> gene_table_path
    expects: results/oantigen_locus/<subtype>/*.oantigen_genes.*.tsv
    """
    if not OLOC_DIR.exists():
        raise FileNotFoundError(f"Not found: {OLOC_DIR}")

    tables = {}
    for sub_dir in OLOC_DIR.iterdir():
        if not sub_dir.is_dir():
            continue
        tsv = find_one(["*.oantigen_genes.*.tsv"], sub_dir)
        if tsv:
            tables[sub_dir.name] = tsv
    return tables

def parse_oantigen_genes_tsv(tsv: Path):
    """
    returns:
      protein_ids: set[str]
      products: list[str]
      locus_gene_count: int
    """
    protein_ids = set()
    products = []
    n = 0
    with open(tsv, "r", encoding="utf-8", errors="replace") as f:
        header = next(f).rstrip("\n").split("\t")
        # expected columns
        col_pi = header.index("protein_id")
        col_pr = header.index("product")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(col_pi, col_pr):
                continue
            pid = parts[col_pi].strip()
            prod = parts[col_pr].strip()
            if pid:
                protein_ids.add(pid)
            products.append(prod)
            n += 1
    return protein_ids, products, n

def parse_dbcan_hmmer_out(hmmer_out: Path):
    """
    Parses hmmer.out to map query_id -> set(families)
    Very tolerant parsing:
    - Find all family tokens GTxx/GHxx/CE/PL/CBM in each line
    - Determine query id by taking a plausible column
    """
    query_to_fams = defaultdict(set)

    with open(hmmer_out, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fams = set(m.group(0).upper() for m in FAM_RE.finditer(line))
            if not fams:
                continue

            cols = line.rstrip("\n").split()
            # dbCAN hmmer.out formats vary; query is often at cols[2] or cols[3]
            # We'll try several candidates and pick the one that looks like an ID (has '_' or '|', or is long).
            cand_idxs = [2, 3, 0, 1, 4]
            q = None
            for idx in cand_idxs:
                if idx < len(cols):
                    c = cols[idx]
                    if len(c) >= 6:
                        q = c
                        break
            if q is None:
                continue

            for fam in fams:
                query_to_fams[q].add(fam)

    return query_to_fams

def match_locus_proteins_to_dbcan(protein_ids: set[str], query_to_fams: dict[str, set[str]]):
    """
    protein_id from GFF (e.g., WP_..., or something else)
    hmmer query IDs may contain protein_id as substring or match exactly.
    We'll do:
      1) exact match
      2) substring match (protein_id in query string)
    """
    matched = 0
    locus_fams = Counter()

    # exact matches
    for pid in protein_ids:
        if pid in query_to_fams:
            for fam in query_to_fams[pid]:
                locus_fams[fam] += 1
            matched += 1

    # substring matches for remaining
    remaining = [pid for pid in protein_ids if pid not in query_to_fams]
    if remaining:
        # build list of queries once
        queries = list(query_to_fams.keys())
        for pid in remaining:
            found_any = False
            for q in queries:
                if pid in q:
                    for fam in query_to_fams[q]:
                        locus_fams[fam] += 1
                    found_any = True
            if found_any:
                matched += 1

    return locus_fams, matched

def product_features(products: list[str], top_tokens: list[str]) -> dict[str, int]:
    """
    - key genes presence/count based on product text
    - top token counts (global top tokens computed later)
    """
    feats = {}

    text_all = " ".join(products).lower()

    for kg in KEYGENES:
        feats[f"has_{kg}"] = 1 if re.search(rf"\b{kg}\b", text_all) else 0

    # some broad keyword counts
    kw_list = [
        "glycosyltransferase",
        "acetyltransferase",
        "sulfotransferase",
        "dehydrogenase",
        "epimerase",
        "mutase",
        "isomerase",
        "kinase",
        "ligase",
        "transferase",
        "polymerase",
        "flippase",
        "transporter",
    ]
    for kw in kw_list:
        feats[f"kw_{kw}"] = sum(1 for p in products if kw in p.lower())

    # token counts
    token_counts = Counter()
    for p in products:
        toks = [t.lower() for t in WORD_RE.findall(p)]
        toks = [t for t in toks if t not in STOP and len(t) >= 3]
        token_counts.update(toks)

    for tok in top_tokens:
        feats[f"prod_tok_{tok}"] = token_counts.get(tok, 0)

    return feats

def main():
    subtype_to_acc = load_subtype_to_acc()
    subtype_to_table = load_oantigen_gene_tables()

    # First pass: collect product tokens globally to choose top-N tokens
    global_tok = Counter()
    subtype_products = {}
    subtype_protein_ids = {}
    subtype_gene_count = {}

    for subtype, tsv in subtype_to_table.items():
        pids, products, n = parse_oantigen_genes_tsv(tsv)
        subtype_products[subtype] = products
        subtype_protein_ids[subtype] = pids
        subtype_gene_count[subtype] = n

        for p in products:
            toks = [t.lower() for t in WORD_RE.findall(p)]
            toks = [t for t in toks if t not in STOP and len(t) >= 3]
            global_tok.update(toks)

    # choose top 50 tokens for product features (can change to 20/100 as needed)
    top_tokens = [t for t, c in global_tok.most_common(50)]

    # Second pass: build features per subtype
    rows = []
    all_keys = set(["subtype", "accession", "locus_gene_count", "dbcan_locus_matched_proteins"])

    # cache parsed dbcan per accession
    dbcan_cache = {}

    for subtype, products in subtype_products.items():
        acc = subtype_to_acc.get(subtype, "NA")

        feats = {}
        feats["subtype"] = subtype
        feats["accession"] = acc
        feats["locus_gene_count"] = subtype_gene_count.get(subtype, 0)

        # product-derived features
        pf = product_features(products, top_tokens)
        feats.update(pf)

        # dbCAN locus family counts (if accession exists and hmmer.out exists)
        locus_matched = 0
        if acc != "NA":
            acc_dbcan_dir = DBCAN_DIR / acc
            hmmer_out = acc_dbcan_dir / "hmmer.out"
            if hmmer_out.exists():
                if acc not in dbcan_cache:
                    dbcan_cache[acc] = parse_dbcan_hmmer_out(hmmer_out)
                query_to_fams = dbcan_cache[acc]

                locus_fams, locus_matched = match_locus_proteins_to_dbcan(subtype_protein_ids[subtype], query_to_fams)

                # add family counts
                # also aggregate into 5 big classes: GT/GH/CE/PL/CBM
                big = Counter()
                for fam, cnt in locus_fams.items():
                    feats[f"locus_{fam}"] = cnt
                    big[fam[:2].upper()] += cnt

                for cls in ["GT", "GH", "CE", "PL", "CBM"]:
                    feats[f"locus_{cls}_total"] = big.get(cls, 0)

        feats["dbcan_locus_matched_proteins"] = locus_matched

        all_keys.update(feats.keys())
        rows.append(feats)

    # write table
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["subtype", "accession", "locus_gene_count", "dbcan_locus_matched_proteins"]

    # put key gene flags early
    for kg in KEYGENES:
        k = f"has_{kg}"
        if k in all_keys and k not in cols:
            cols.append(k)

    # keyword counts early
    for k in sorted([k for k in all_keys if k.startswith("kw_")]):
        if k not in cols:
            cols.append(k)

    # locus big class totals
    for cls in ["GT", "GH", "CE", "PL", "CBM"]:
        k = f"locus_{cls}_total"
        if k in all_keys and k not in cols:
            cols.append(k)

    # locus family counts
    for k in sorted([k for k in all_keys if k.startswith("locus_") and k not in cols]):
        cols.append(k)

    # product tokens
    for k in sorted([k for k in all_keys if k.startswith("prod_tok_")]):
        cols.append(k)

    # remaining
    for k in sorted(all_keys):
        if k not in cols:
            cols.append(k)

    with open(OUT_TSV, "w", encoding="utf-8") as out:
        out.write("\t".join(cols) + "\n")
        for feats in rows:
            out.write("\t".join(str(feats.get(c, 0)) for c in cols) + "\n")

    print(f"[DONE] wrote: {OUT_TSV}")
    print(f"[INFO] subtypes processed: {len(rows)}")
    print(f"[INFO] product top tokens used: {len(top_tokens)}")
    print("[NOTE] dbCAN locus matching quality depends on protein_id vs hmmer query id consistency.")
    print("Check column: dbcan_locus_matched_proteins (0이면 매칭 실패 가능성이 큽니다).")

if __name__ == "__main__":
    main()
