#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path
import pandas as pd

# 선택: product token feature를 TF-IDF로 만들고 싶을 때
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None


CAZY_PREFIXES = ("GH", "GT", "PL", "CE", "AA", "CBM")


def normalize_gene_name(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def parse_hmmer_out(hmmer_path: Path) -> pd.DataFrame:
    if not hmmer_path.exists():
        return pd.DataFrame(columns=["query_id", "target_hmm", "evalue"])

    df = pd.read_csv(hmmer_path, sep="\t", dtype=str).fillna("")
    df.columns = [str(c).strip() for c in df.columns]

    # 컬럼명은 hmmer 파일에서 확정됨
    out = pd.DataFrame({
        "query_id": df["Gene ID"].astype(str).str.strip(),
        "target_hmm": df["HMM Profile"].astype(str).str.strip(),
        "evalue": pd.to_numeric(df["E Value"], errors="coerce"),
    }).dropna(subset=["evalue"])

    return out

def extract_cazy_family(target_hmm: str) -> str:
    """
    target_hmm에서 CAZy family 라벨을 추출합니다.
    예: GH13.hmm -> GH13
        GT2_XXX -> GT2
    """
    if target_hmm is None:
        return ""
    s = str(target_hmm)

    # 흔한 케이스: GH13.hmm, GT2.hmm, CBM50.hmm
    m = re.search(r"\b(" + "|".join(CAZY_PREFIXES) + r")\s*[_-]?\s*(\d+)\b", s)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    # 접두만 있는 경우(드묾): GH, GT 등
    for p in CAZY_PREFIXES:
        if s.startswith(p):
            return p
    return ""


def best_hit_per_gene(hmmer_df: pd.DataFrame) -> pd.DataFrame:
    """
    query_id 별로 evalue 최소 hit만 남깁니다.
    """
    if hmmer_df.empty:
        return hmmer_df
    hmmer_df = hmmer_df.copy()
    hmmer_df["cazy_family"] = hmmer_df["target_hmm"].map(extract_cazy_family)
    hmmer_df = hmmer_df[hmmer_df["cazy_family"] != ""]
    if hmmer_df.empty:
        return hmmer_df
    hmmer_df = hmmer_df.sort_values(["query_id", "evalue"], ascending=[True, True])
    return hmmer_df.groupby("query_id", as_index=False).first()

def load_oantigen_genes(genes_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(genes_tsv, sep="\t", dtype=str).fillna("")
    df.columns = [str(c).strip() for c in df.columns]

    if "protein_id" not in df.columns:
        raise RuntimeError(f"'protein_id' column not found in {genes_tsv}. Columns={list(df.columns)}")

    out = pd.DataFrame()
    out["gene_id"] = df["protein_id"].astype(str).str.strip()   #WP_...로 dbCAN과 매칭
    out["product"] = df["product"].astype(str) if "product" in df.columns else ""
    out["accession"] = df["accession"].astype(str).str.strip() if "accession" in df.columns else ""
    return out

def build_features(
    locus_root: Path,
    dbcan_root: Path,
    out_tsv: Path,
    token_mode: str = "tfidf",
    tfidf_max_features: int = 2000,
    tfidf_min_df: int = 2
):
    """
    locus_root: results/oantigen_locus
    dbcan_root: results/dbcan
    """
    feature_rows = []
    product_corpus = []   # O-type별 product concat
    otypes_for_corpus = []

    # 1) accession별 dbCAN best-hit map 사전 구축(필요한 accession만 lazy-load로도 가능)
    # 여기서는 lazy-load: O-type을 돌며 accession이 나타날 때마다 파싱 캐시합니다.
    hmmer_cache = {}  # accession -> best_hit_df(query_id, cazy_family)

    def get_besthit_for_accession(acc: str) -> pd.DataFrame:
        if acc in hmmer_cache:
            return hmmer_cache[acc]
        hmmer_path = dbcan_root / acc / "hmmer.out"
        df = parse_hmmer_out(hmmer_path)
        df = best_hit_per_gene(df)
        hmmer_cache[acc] = df
        return df

    # 2) O-type별 loop
    for otype_dir in sorted(locus_root.glob("*")):
        if not otype_dir.is_dir():
            continue
        otype = otype_dir.name

        # genes_tsv 후보 탐색: 보통 *.oantigen_genes.tsv 형태지만, 유연하게 처리
        gene_candidates = sorted(otype_dir.glob("*.oantigen_genes*.tsv"))
        if not gene_candidates:
            gene_candidates = sorted(otype_dir.glob("*oantigen_genes*.tsv"))

        if not gene_candidates:
            # print(f"[WARN] no genes file under {otype_dir}")
            continue

        # 대표 genome 1개만 사용(현재 폴더당 1개면 사실상 이게 대표)
        genes_tsv = gene_candidates[0]
        print(f"[DEBUG] otype={otype} genes_tsv={genes_tsv.name}")

        genes_df = load_oantigen_genes(genes_tsv)

        gene_count = int((genes_df["gene_id"] != "").sum())

        # wzx/wzy presence (대/소문자 + 변형 고려)
        products = " ".join(genes_df["product"].astype(str).tolist()).lower()
        genes_join = " ".join(genes_df["gene_id"].astype(str).tolist()).lower()

        def has_kw(kw: str) -> int:
            return 1 if (kw in products or kw in genes_join) else 0

        wzx = has_kw("wzx")
        wzy = has_kw("wzy")
        wzz = has_kw("wzz")
        wzm = has_kw("wzm")
        wzt = has_kw("wzt")

        # 3) dbCAN 기반 locus CAZy count
        # genes_df에 accession이 비어있을 수 있으므로, 비어있으면 "유일 accession"을 별도로 추론해야 합니다.
        # 여기서는: genes_df['accession']에 값이 있으면 그 값들로, 없으면 otype 전체를 'unknown'으로 처리합니다.
        cazy_counts = {}
        locus_cazy_total = 0

        acc_list = sorted({a for a in genes_df["accession"].tolist() if a.strip()})
        if acc_list:
            # 여러 accession이 섞여 있다면 gene 단위로 join
            for acc in acc_list:
                besthit = get_besthit_for_accession(acc)
                if besthit.empty:
                    continue
                # 해당 accession에 속한 gene_id만
                acc_gene_ids = set(genes_df.loc[genes_df["accession"] == acc, "gene_id"].tolist())
                bh = besthit[besthit["query_id"].isin(acc_gene_ids)]
                if bh.empty:
                    continue
                for fam, n in bh["cazy_family"].value_counts().items():
                    cazy_counts[fam] = cazy_counts.get(fam, 0) + int(n)
                    locus_cazy_total += int(n)
        else:
            # accession 정보가 없다면, dbCAN과 연결 불가 → 0으로 두되, 추후 개선 포인트로 남겨둠
            pass

        # 4) O-type별 product 텍스트 (token feature용)
        product_text = " ; ".join([p for p in genes_df["product"].tolist() if p.strip()])

        row = {
            "O_type": otype,
            "locus_gene_count": gene_count,
            "has_wzx": wzx,
            "has_wzy": wzy,
            "has_wzz": wzz,
            "has_wzm": wzm,
            "has_wzt": wzt,
            "locus_cazy_total_hits": locus_cazy_total,
            "product_text": product_text,  # 토큰화 전 원문 저장(추적/디버깅용)
        }

        # CAZy family count들을 wide로 만들기 위해 dict를 같이 저장
        row["_cazy_dict"] = cazy_counts

        feature_rows.append(row)
        product_corpus.append(product_text)
        otypes_for_corpus.append(otype)

    if not feature_rows:
        raise RuntimeError(f"No locus features built. Check locus_root={locus_root}")

    base_df = pd.DataFrame(feature_rows)

    # 5) CAZy dict를 wide 컬럼으로 확장
    all_fams = sorted({fam for d in base_df["_cazy_dict"].tolist() for fam in d.keys()})
    for fam in all_fams:
        base_df[f"cazy_{fam}_count"] = base_df["_cazy_dict"].map(lambda d: int(d.get(fam, 0)))
    base_df = base_df.drop(columns=["_cazy_dict"])

    # 6) product token features (TF-IDF 권장)
    if token_mode == "none":
        feat_df = base_df.drop(columns=["product_text"], errors="ignore")
    elif token_mode == "tfidf":
        if TfidfVectorizer is None:
            raise RuntimeError("scikit-learn not available. Install scikit-learn or use --token-mode none")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
            max_features=tfidf_max_features,
            min_df=tfidf_min_df,
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(base_df["product_text"].fillna(""))
        tfidf_cols = [f"prod_tfidf__{t}" for t in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf_cols)
        feat_df = pd.concat([base_df.drop(columns=["product_text"]), tfidf_df], axis=1)
    else:
        raise ValueError(f"Unknown token_mode={token_mode}")

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] wrote: {out_tsv} (rows={len(feat_df)}, cols={feat_df.shape[1]})")


def main():
    ap = argparse.ArgumentParser()

    # 기본값(default) 지정: 옵션 없이 실행 가능
    project_root = Path(__file__).resolve().parents[1]
    ap.add_argument("--locus-root", default=project_root / "results/oantigen_locus", type=Path)
    ap.add_argument("--dbcan-root", default=project_root / "results/dbcan", type=Path)
    ap.add_argument("--out", default=project_root / "results/features/oantigen_locus_features.tsv", type=Path)

    ap.add_argument("--token-mode", default="tfidf", choices=["tfidf", "none"])
    ap.add_argument("--tfidf-max-features", type=int, default=2000)
    ap.add_argument("--tfidf-min-df", type=int, default=2)

    args = ap.parse_args()

    build_features(
        locus_root=args.locus_root,
        dbcan_root=args.dbcan_root,
        out_tsv=args.out,
        token_mode=args.token_mode,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
    )
    print(f"[PARAM] locus_root={args.locus_root}")
    print(f"[PARAM] dbcan_root={args.dbcan_root}")
    print(f"[PARAM] out={args.out}")
    print(f"[PARAM] token_mode={args.token_mode}, tfidf_max_features={args.tfidf_max_features}, tfidf_min_df={args.tfidf_min_df}")

if __name__ == "__main__":
    main()
