"""
monosaccharide_matrix_ecodab.py

ecodab_residue_count_matrix_base.tsv → monosaccharide category count matrix.

This file's column names already have stereo/linkage prefixes (a/b, D/L) removed,
so we classify the "core" form directly.

Column header  : subtype
Output file    : monosaccharide_matrix_ecodab.tsv

Mapping rules (same logic as monosaccharide_matrix.py, but applied to bare core names):
  Special standalones    : Gro, GroA, Ribitol→ribitol, Pyr, IdoA, Col
  digit+e prefix (8eLeg) : strip digit+e → reclassify → Leg_mod
  Hep (any form)         : Hep_mod
  ara4dHex               : Hex_mod
  Tal…                   : Tal_mod
  Neu…                   : Neu_mod  (plain Neu → Neu)
  Pse…                   : Pse_mod
  GalNAc + suffix        : GalNAc_mod   |  plain GalNAc → GalNAc
  GalA   + suffix        : GalA_mod     |  plain GalA   → GalA
  GlcA   + suffix        : GlcA_mod     |  plain GlcA   → GlcA
  ManNAc + suffix        : ManNAc_mod   |  plain ManNAc → ManNAc
  RhaNAc + suffix        : RhaNAc_mod   |  plain RhaNAc → RhaNAc
  FucNAc (plain)         : FucNAc       |  FucNAc+suffix / FucNAm → FucNAc_mod
  QuiNAc (plain)         : QuiNAc       |  QuiNAc+suffix → QuiNAc_mod
  Qui+digit+NAc          : QuiNAc       |  Qui+other → Qui
  GlcNAc + suffix        : GlcNAc_mod   |  plain GlcNAc → GlcNAc
  Galf   + suffix        : Galf_mod     |  plain Galf   → Galf
  Fruf   + suffix        : Fruf_mod     |  plain Fruf   → Fruf
  Fucf…                  : Fuc_mod
  Ribf / Rib…            : Rib
  Xul…                   : Xul
  Fuc    + suffix        : Fuc_mod      |  plain Fuc    → Fuc
  Rha    + suffix        : Rha_mod      |  plain Rha    → Rha
  Gal    + suffix        : Gal_mod      |  plain Gal    → Gal
  Glc    + suffix        : Glc_mod      |  plain Glc    → Glc
  Man    + suffix        : Man_mod      |  plain Man    → Man
  fallback               : core (as-is)
"""

import re
import csv
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core classifier  (input = column name already stripped of a/b D/L prefix)
# ---------------------------------------------------------------------------

def classify_core(core: str) -> str:
    """Map a bare monosaccharide core name to its category."""

    # --- special standalones ---
    _standalone = {
        'Gro': 'Gro', 'GroA': 'GroA',
        'Ribitol': 'ribitol',
        'Pyr': 'Pyr', 'IdoA': 'IdoA', 'Col': 'Col',
    }
    if core in _standalone:
        return _standalone[core]

    # --- digit+e prefix (e.g. 8eLeg5Ac7Ac → Leg5Ac7Ac) ---
    m = re.match(r'^(\d+e)(.*)', core)
    if m:
        return classify_core(m.group(2))

    # --- digit-only prefix that encodes deoxy position on Hep (e.g. 6dManHep2Ac) ---
    if re.search(r'Hep', core):
        return 'Hep_mod'

    # --- ara4dHex ---
    if re.match(r'^ara\d*d[Hh]ex', core):
        return 'Hex_mod'

    # --- Leg ---
    if core.startswith('Leg'):
        return 'Leg_mod'

    # --- Tal ---
    if core.startswith('Tal'):
        return 'Tal_mod'

    # --- Neu ---
    if core.startswith('Neu'):
        return 'Neu_mod' if len(core) > 3 else 'Neu'

    # --- Pse ---
    if core.startswith('Pse'):
        return 'Pse_mod'

    # --- GalNAc ---
    if core.startswith('GalNAc'):
        return 'GalNAc_mod' if len(core) > 6 else 'GalNAc'

    # --- GalA ---
    if core.startswith('GalA'):
        return 'GalA_mod' if len(core) > 4 else 'GalA'

    # --- GlcA ---
    if core.startswith('GlcA'):
        return 'GlcA_mod' if len(core) > 4 else 'GlcA'

    # --- ManNAc ---
    if core.startswith('ManNAc'):
        return 'ManNAc_mod' if len(core) > 6 else 'ManNAc'

    # --- RhaNAc ---
    if core.startswith('RhaNAc'):
        return 'RhaNAc_mod' if len(core) > 6 else 'RhaNAc'

    # --- FucNAc / FucNAm  (must check before generic Fuc) ---
    if core.startswith('FucNAc'):
        return 'FucNAc_mod' if len(core) > 6 else 'FucNAc'
    if core.startswith('FucNAm'):
        return 'FucNAc_mod'   # FucNAm always counts as modified

    # --- QuiNAc (plain or suffixed) ---
    if core.startswith('QuiNAc'):
        return 'QuiNAc_mod' if len(core) > 6 else 'QuiNAc'

    # --- Qui + NAc-digit suffix → QuiNAc; other Qui → Qui ---
    if core.startswith('Qui'):
        suffix = core[3:]
        if not suffix:
            return 'Qui'
        if re.match(r'^\d+NAc', suffix):
            return 'QuiNAc'
        return 'Qui'

    # --- GlcNAc ---
    if core.startswith('GlcNAc'):
        return 'GlcNAc_mod' if len(core) > 6 else 'GlcNAc'

    # --- Galf ---
    if core.startswith('Galf'):
        return 'Galf_mod' if len(core) > 4 else 'Galf'

    # --- Fruf ---
    if core.startswith('Fruf'):
        return 'Fruf_mod' if len(core) > 4 else 'Fruf'

    # --- Fucf (furanose Fuc, always modified form) ---
    if core.startswith('Fucf'):
        return 'Fuc_mod'

    # --- Ribf / Rib ---
    if core.startswith('Ribf') or core.startswith('Rib'):
        return 'Rib'

    # --- Xul ---
    if core.startswith('Xul'):
        return 'Xul'

    # --- Fuc ---
    if core.startswith('Fuc'):
        return 'Fuc_mod' if len(core) > 3 else 'Fuc'

    # --- Rha ---
    if core.startswith('Rha'):
        return 'Rha_mod' if len(core) > 3 else 'Rha'

    # --- Gal ---
    if core.startswith('Gal'):
        return 'Gal_mod' if len(core) > 3 else 'Gal'

    # --- Glc ---
    if core.startswith('Glc'):
        return 'Glc_mod' if len(core) > 3 else 'Glc'

    # --- Man ---
    if core.startswith('Man'):
        return 'Man_mod' if len(core) > 3 else 'Man'

    # --- fallback ---
    return core


# ---------------------------------------------------------------------------
# Build matrix
# ---------------------------------------------------------------------------

def build_matrix(input_path: str, output_path: str):
    with open(input_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames

        # First column is the sample/subtype identifier
        id_col = fieldnames[0]   # 'subtype'
        residue_cols = [c for c in fieldnames if c != id_col]

        # Build column → category mapping
        col_to_cat = {col: classify_core(col) for col in residue_cols}

        # Collect unique categories in order of first appearance
        seen_cats, cat_set = [], set()
        for col in residue_cols:
            cat = col_to_cat[col]
            if cat not in cat_set:
                seen_cats.append(cat)
                cat_set.add(cat)

        # Each row is kept independently; duplicate subtype names get _1, _2, ... suffix
        # First pass: count total occurrences per subtype
        all_raw = []
        subtype_total = {}
        for row in reader:
            sid = row[id_col]
            subtype_total[sid] = subtype_total.get(sid, 0) + 1
            counts = defaultdict(int)
            for col in residue_cols:
                val = row.get(col, '0').strip()
                try:
                    n = int(val)
                except ValueError:
                    n = 0
                if n:
                    counts[col_to_cat[col]] += n
            all_raw.append((sid, counts))

        # Second pass: assign unique labels
        occurrence = {}
        labeled_rows = []
        for sid, counts in all_raw:
            occurrence[sid] = occurrence.get(sid, 0) + 1
            label = f"{sid}_{occurrence[sid]}" if subtype_total[sid] > 1 else sid
            labeled_rows.append((label, counts))

    # Write output TSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([id_col] + seen_cats)
        for label, counts in labeled_rows:
            writer.writerow([label] + [counts.get(cat, 0) for cat in seen_cats])

    print(f"Done. {len(labeled_rows)} rows × {len(seen_cats)} monosaccharide categories")
    print(f"Output: {output_path}")

    print("\n--- Column → Category mapping ---")
    for col, cat in col_to_cat.items():
        print(f"  {col:35s} → {cat}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys, os

    if len(sys.argv) == 3:
        inp, out = sys.argv[1], sys.argv[2]
    else:
        inp = 'data/ecodab_residue_count_matrix_base.tsv' #216 subtyoes
        out = 'data/simplify_ecodab_monosaccharide.tsv'  #monosaccharide and it's modification categories
        if not os.path.exists(inp):
            base = os.path.dirname(os.path.abspath(__file__))
            inp = os.path.join(base, inp)
            out = os.path.join(base, out)

    build_matrix(inp, out)
