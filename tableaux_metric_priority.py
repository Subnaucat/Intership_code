"""
tableaux_metric_priority.py — Tableaux croisés metric_priority × paramètres
=============================================================================
Génère, pour chaque paramètre d'entrée, un tableau croisé montrant la
distribution (en effectif ET en pourcentage) de metric_priority.

Définition des valeurs :
  -1 → Inconnu / non spécifié
   0 → EFX par défaut (priorité basse)
   1 → Priorité SFX
   2 → Priorité EFX
   3 → Priorité NCOB
   4 → Priorité ECOB

Les paramètres analysés :
  - nb_cameras      : nombre de caméras (qgc[2])
  - nb_cont         : nombre de contaminants
  - mag_star        : magnitude de l'étoile (tranches)
  - mag_cont_mean   : magnitude moyenne des contaminants (tranches)
  - ang_dist        : distance angulaire étoile/contaminant (tranches)
  - depth_prim      : profondeur du transit (tranches)
  - signal_type     : type de signal (planète / EB)

Pour les paramètres continus, le découpage en tranches est automatique
(quantiles, configurable via --bins).

Usage :
  python tableaux_metric_priority.py --folder /chemin/vers/fichiers
  python tableaux_metric_priority.py --folder . --bins 4 --output tableaux.xlsx
"""

import os, glob, json, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────

MP_LABELS = {
    -1: "-1 Inconnu",
     0: " 0 EFX défaut",
     1: " 1 SFX",
     2: " 2 EFX",
     3: " 3 NCOB",
     4: " 4 ECOB",
}

PARAM_LABELS = {
    "nb_cameras":    "cameras nb",
    "nb_cont":       "Contaminants nb",
    "mag_star":      "Star magnitude",
    "mag_cont_mean": "Cont. magnitude (moy.)",
    "delta_mag": "Magnitude difference",
    "ang_dist":      "Ang. distance  [arcsec]",
    "depth_prim":    "Transit depth [ppm]",
    "signal_type":   "Signal type",
}

NUMERIC_PARAMS = ["nb_cameras", "nb_cont", "mag_star",
                  "mag_cont_mean","delta_mag", "ang_dist", "depth_prim"]
ALL_PARAMS     = NUMERIC_PARAMS + ["signal_type"]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _f(val, absent=-1):
    try:
        v = float(val)
        return np.nan if (v == absent and absent != -1) else v
    except (TypeError, ValueError):
        return np.nan

def _mean_list(lst):
    vals = [float(x) for x in lst
            if x is not None and float(x) != -1]
    return float(np.mean(vals)) if vals else np.nan


# ─────────────────────────────────────────────────────────────────
# Chargement — FORMAT A uniquement (metric_priority dans donnees)
# ─────────────────────────────────────────────────────────────────

def extract_records(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] {filepath}: {e}")
        return []

    # Détecter FORMAT A (mask) uniquement
    produit = data.get("produit", [])
    donnees = data.get("donnees", [])
    if not donnees:
        return []

    # FORMAT B : donnees[0] est une liste, metric_priority absent → ignorer
    if isinstance(donnees[0], list):
        return []
    if not any(("nominal" in str(p) or "extended" in str(p)) for p in produit):
        # Pas clairement FORMAT A non plus → vérifier metric_priority
        if donnees and donnees[0].get("metric_priority") is None:
            return []

    params = data.get("parametres", [{}])[0]
    qgc    = params.get("qgc", [[], [], -1])
    depth  = _f(params.get("depth_prim_ldc", params.get("depth_prim", np.nan)), absent=None)

    # depth : on garde -1 comme valeur réelle (= absent = valeur dans les données)
    # Mais pour les autres paramètres on garde les vraies valeurs
    def fp(key, fallback=np.nan):
        v = params.get(key, fallback)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    nb_cameras    = fp("qgc", np.nan)  # on va utiliser qgc[2]
    try:
        nb_cameras = float(qgc[2])
        if nb_cameras == -1:
            nb_cameras = np.nan
    except Exception:
        nb_cameras = np.nan

    nb_cont       = fp("nb_cont");       nb_cont       = np.nan if nb_cont == -1 else nb_cont
    mag_star      = fp("mag_star");      mag_star      = np.nan if mag_star == -1 else mag_star
    ang_dist      = fp("Ang_dist");      ang_dist      = np.nan if ang_dist == -1 else ang_dist
    depth_prim    = fp("depth_prim_ldc") if "depth_prim_ldc" in params else fp("depth_prim")
    if depth_prim == -1: depth_prim = np.nan
    signal_type   = params.get("Signal_type", "unknown")
    mag_cont_mean = _mean_list(params.get("mag_sec", []))

    sim  = Path(filepath).stem
    rows = []
    for obs in donnees:
        mp = obs.get("metric_priority", -1)
        if mp is None:
            mp = -1
        rows.append({
            "sim":            sim,
            "nb_cameras":     nb_cameras,
            "nb_cont":        nb_cont,
            "mag_star":       mag_star,
            "mag_cont_mean":  mag_cont_mean,
            "delta_mag": mag_cont_mean-mag_star,
            "ang_dist":       ang_dist,
            "depth_prim":     depth_prim,
            "signal_type":    signal_type,
            "metric_priority": int(mp),
        })
    return rows


def load_all(folder):
    files = sorted(glob.glob(os.path.join(folder, "product_param_sim*.json")))
    if not files:
        raise FileNotFoundError(f"Aucun fichier dans : {folder}")
    print(f"  {len(files)} fichier(s) trouvé(s).")
    rows = []
    skipped = 0
    for fp in files:
        r = extract_records(fp)
        if r:
            rows.extend(r)
        else:
            skipped += 1
    df = pd.DataFrame(rows)
    print(f"  {len(df)} observations chargées "
          f"({df['sim'].nunique()} simulations, {skipped} fichiers ignorés [FORMAT B]).")
    return df


# ─────────────────────────────────────────────────────────────────
# Construction des tableaux croisés
# ─────────────────────────────────────────────────────────────────

def all_mp_values(df):
    """Retourne les valeurs de metric_priority présentes, triées."""
    return sorted(df["metric_priority"].unique())


def bin_param(df, param, n_bins=5):
    """
    Retourne une Series avec les étiquettes de tranches pour un paramètre continu.
    Si le paramètre est discret (≤ n_bins valeurs uniques), retourne tel quel.
    """
    col = df[param].dropna()
    if col.nunique() <= n_bins:
        return df[param].astype(str)
    try:
        cuts = pd.qcut(df[param], q=n_bins, duplicates="drop")
        return cuts.astype(str)
    except Exception:
        return df[param].astype(str)


def crosstab_for_param(df, param, n_bins=5):
    """
    Retourne (ct_count, ct_pct) :
      - ct_count : effectifs (lignes = tranches du paramètre, colonnes = metric_priority)
      - ct_pct   : pourcentages ligne (somme = 100% par tranche)
    Les colonnes sont renommées avec MP_LABELS.
    """
    mp_vals = all_mp_values(df)
    col_names = [MP_LABELS.get(v, str(v)) for v in mp_vals]

    if param == "signal_type":
        group_col = df[param].fillna("NaN")
    else:
        group_col = bin_param(df, param, n_bins)
        group_col = group_col.fillna("NaN")

    ct = pd.crosstab(group_col, df["metric_priority"],
                     margins=True, margins_name="TOTAL")

    # Renommer les colonnes metric_priority
    rename_mp = {v: MP_LABELS.get(v, str(v)) for v in mp_vals}
    rename_mp["TOTAL"] = "TOTAL"
    ct = ct.rename(columns=rename_mp)
    ct.index.name = PARAM_LABELS.get(param, param)

    # Pourcentages (hors TOTAL)
    ct_body = ct.drop(index="TOTAL", errors="ignore")
    ct_pct  = ct_body.copy().astype(float)
    row_totals = ct_body.get("TOTAL", ct_body.sum(axis=1))
    for col in ct_body.columns:
        if col != "TOTAL":
            ct_pct[col] = (ct_body[col] / row_totals * 100).round(1)
    ct_pct["TOTAL"] = 100.0

    # Ligne TOTAL pour ct_pct (% de chaque MP sur l'ensemble)
    total_counts = ct.loc["TOTAL"] if "TOTAL" in ct.index else ct.sum()
    grand_total  = total_counts.get("TOTAL", total_counts.sum())
    total_pct_row = {}
    for col in ct_pct.columns:
        if col != "TOTAL":
            total_pct_row[col] = round(total_counts.get(col, 0) / grand_total * 100, 1)
        else:
            total_pct_row[col] = 100.0
    ct_pct = pd.concat([ct_pct,
                         pd.DataFrame(total_pct_row, index=["TOTAL"])])
    ct_pct.index.name = ct.index.name

    return ct, ct_pct


# ─────────────────────────────────────────────────────────────────
# Distribution globale
# ─────────────────────────────────────────────────────────────────

def global_distribution(df):
    """Tableau de distribution globale de metric_priority."""
    counts = df["metric_priority"].value_counts().sort_index()
    total  = counts.sum()
    result = pd.DataFrame({
        "metric_priority": counts.index,
        "Label":           [MP_LABELS.get(v, str(v)) for v in counts.index],
        "Effectif":        counts.values,
        "Pourcentage (%)": (counts.values / total * 100).round(2),
    }).set_index("metric_priority")
    result.loc["TOTAL"] = ["TOTAL", total, 100.0]
    return result


# ─────────────────────────────────────────────────────────────────
# Affichage texte
# ─────────────────────────────────────────────────────────────────

def print_all_tables(df, n_bins):
    print("\n══════════════════════════════════════════════════════════════")
    print("  DISTRIBUTION GLOBALE DE metric_priority")
    print("══════════════════════════════════════════════════════════════")
    print(global_distribution(df).to_string())

    for param in ALL_PARAMS:
        if df[param].dropna().empty:
            continue
        ct, ct_pct = crosstab_for_param(df, param, n_bins)
        label = PARAM_LABELS.get(param, param)
        print(f"\n{'═'*62}")
        print(f"  TABLEAU CROISÉ : metric_priority × {label}")
        print(f"{'═'*62}")
        print("\n── Effectifs ──")
        print(ct.to_string())
        print("\n── Pourcentages (% par ligne) ──")
        # Formater avec le signe %
        pct_display = ct_pct.copy()
        for col in pct_display.columns:
            pct_display[col] = pct_display[col].apply(
                lambda x: f"{x:.1f}%" if not pd.isna(x) else "—")
        print(pct_display.to_string())


# ─────────────────────────────────────────────────────────────────
# Export Excel (multi-onglets)
# ─────────────────────────────────────────────────────────────────

def export_excel(df, n_bins, output_path):
    """Exporte tous les tableaux dans un fichier Excel, un onglet par paramètre."""
    try:
        import openpyxl
    except ImportError:
        print("  [INFO] openpyxl non disponible, export Excel ignoré.")
        print("         Installez-le avec : pip install openpyxl")
        return

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Onglet distribution globale
        dist = global_distribution(df)
        dist.to_excel(writer, sheet_name="Distribution globale")

        for param in ALL_PARAMS:
            if df[param].dropna().empty:
                continue
            ct, ct_pct = crosstab_for_param(df, param, n_bins)
            label = PARAM_LABELS.get(param, param)[:28]  # limite nom onglet Excel

            # Effectifs
            sheet_name_n = f"{label[:14]}_N"
            ct.to_excel(writer, sheet_name=sheet_name_n)

            # Pourcentages
            sheet_name_p = f"{label[:14]}_%"
            pct_display = ct_pct.copy()
            for col in pct_display.columns:
                pct_display[col] = pct_display[col].apply(
                    lambda x: round(x, 1) if not pd.isna(x) else None)
            pct_display.to_excel(writer, sheet_name=sheet_name_p)

            # Mise en forme basique
            wb   = writer.book
            for sname in [sheet_name_n, sheet_name_p]:
                ws = wb[sname]
                # En-tête en gras
                for cell in ws[1]:
                    cell.font = openpyxl.styles.Font(bold=True)
                # Ligne TOTAL en gras
                for row in ws.iter_rows():
                    if row[0].value == "TOTAL":
                        for cell in row:
                            cell.font = openpyxl.styles.Font(bold=True)
                # Largeur auto
                for col_cells in ws.columns:
                    length = max((len(str(c.value or "")) for c in col_cells), default=8)
                    ws.column_dimensions[col_cells[0].column_letter].width = length + 4

    print(f"  → Excel : {output_path}")


# ─────────────────────────────────────────────────────────────────
# Graphiques
# ─────────────────────────────────────────────────────────────────

MP_COLORS = {
    -1: "#AAAAAA",   # gris  — inconnu
     0: "#4C72B0",   # bleu  — EFX défaut
     1: "#55A868",   # vert  — SFX
     2: "#64B5CD",   # cyan  — EFX
     3: "#C44E52",   # rouge — NCOB
     4: "#DD8452",   # orange— ECOB
}


def plot_global(df, out_dir):
    mp_vals = all_mp_values(df)
    counts  = df["metric_priority"].value_counts().sort_index()
    labels  = [MP_LABELS.get(v, str(v)) for v in mp_vals]
    values  = [counts.get(v, 0) for v in mp_vals]
    colors  = [MP_COLORS.get(v, "#888888") for v in mp_vals]
    total   = sum(values)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Barres
    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Population", fontsize=11)
    ax.set_title("Overall distribution of metric_priority", fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)

    # Camembert
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        values, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        startangle=90, pctdistance=0.75)
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Breakdown by percentage (global)", fontsize=12)

    plt.tight_layout()
    path = os.path.join(out_dir, "mp_distribution_globale.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


def plot_stacked_bars(df, param, n_bins, out_dir):
    """100% stacked bars: metric_priority by parameter range."""
    if param == "signal_type":
        group_col = df[param].fillna("NaN")
    else:
        group_col = bin_param(df, param, n_bins).fillna("NaN")

    mp_vals = all_mp_values(df)
    # Tableau de comptage
    ct = pd.crosstab(group_col, df["metric_priority"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(ct)*1.5), 5))
    bottom = np.zeros(len(ct_pct))
    x      = np.arange(len(ct_pct))

    for mp in mp_vals:
        if mp not in ct_pct.columns:
            continue
        vals   = ct_pct[mp].values
        color  = MP_COLORS.get(mp, "#888888")
        label  = MP_LABELS.get(mp, str(mp))
        bars   = ax.bar(x, vals, bottom=bottom, color=color,
                        label=label, edgecolor="white", width=0.6)
        # Annotation si la tranche est assez large
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(xi, b + v/2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if v > 15 else "black")
        bottom += vals

    ax.set_ylim(0, 105)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ct_pct.index], rotation=20, ha="right", fontsize=9)
    ax.legend(title="metric_priority", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_title(f"metric_priority × {PARAM_LABELS.get(param, param)}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"mp_stacked_{param}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def plot_heatmap_pct(df, param, n_bins, out_dir):
    """Heatmap des pourcentages : lignes = tranches, colonnes = metric_priority."""
    _, ct_pct = crosstab_for_param(df, param, n_bins)
    # Enlever TOTAL pour la heatmap
    ct_body = ct_pct.drop(index="TOTAL", columns="TOTAL", errors="ignore")
    if ct_body.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(ct_body.columns)*1.5),
                                    max(3, len(ct_body)*0.7 + 1.5)))
    im = ax.imshow(ct_body.values.astype(float), cmap="YlOrRd",
                   vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ct_body.columns)))
    ax.set_yticks(range(len(ct_body)))
    ax.set_xticklabels(ct_body.columns, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels([str(i) for i in ct_body.index], fontsize=9)
    for i in range(len(ct_body)):
        for j in range(len(ct_body.columns)):
            v = ct_body.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if v > 60 else "black")
    plt.colorbar(im, ax=ax, label="% ")
    ax.set_xlabel("metric_priority", fontsize=10)
    ax.set_ylabel(PARAM_LABELS.get(param, param), fontsize=10)
    ax.set_title(f"% metric_priority × {PARAM_LABELS.get(param, param)}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f"mp_heatmap_{param}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tableaux croisés metric_priority × paramètres PLATO")
    parser.add_argument("--folder",  default=".",                   help="Dossier JSON")
    parser.add_argument("--output",  default="metric_priority.xlsx",help="Fichier Excel de sortie")
    parser.add_argument("--plots",   default="plots_metric_priority")
    parser.add_argument("--bins",    type=int, default=5,
                        help="Nombre de tranches pour les paramètres continus (défaut: 5)")
    parser.add_argument("--no-plots", action="store_true",          help="Désactiver les graphiques")
    args = parser.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    print("\n══════════════════════════════════════════════════════════════")
    print("  ANALYSE metric_priority × PARAMÈTRES PLATO")
    print("══════════════════════════════════════════════════════════════")

    print("\n[1/4] Chargement des données (FORMAT A uniquement)...")
    df = load_all(args.folder)

    if df.empty:
        print("  ⚠  Aucune donnée FORMAT A avec metric_priority trouvée.")
        return

    print(f"\n  Valeurs de metric_priority présentes : "
          f"{sorted(df['metric_priority'].unique())}")

    print("\n[2/4] Affichage des tableaux...")
    print_all_tables(df, args.bins)

    print("\n[3/4] Export Excel...")
    export_excel(df, args.bins, args.output)

    if not args.no_plots:
        print("\n[4/4] Graphiques...")
        plot_global(df, args.plots)
        for param in ALL_PARAMS:
            if df[param].dropna().empty:
                print(f"  [SKIP] {param} — aucune donnée")
                continue
            if df[param].nunique() <= 1:
                print(f"  [SKIP] {param} — une seule valeur, pas de variabilité")
                continue
            plot_stacked_bars(df, param, args.bins, args.plots)
            plot_heatmap_pct(df, param,  args.bins, args.plots)

    print("\nTerminé ✓")


if __name__ == "__main__":
    main()
