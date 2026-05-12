"""
Analyse statistique des simulations PLATO - VERSION EFFICACITÉ
==============================================================
Règle d'absence : toute valeur == -1 est remplacée par NaN (donnée absente).
Cela s'applique à tous les paramètres ET toutes les sorties.

Paramètres d'entrée (extraits de 'parametres') :
  - nb_cameras      : nombre de caméras (qgc[2], -1 → NaN)
  - nb_cont         : nombre de contaminants (-1 → NaN)
  - mag_star        : magnitude de l'étoile cible (-1 → NaN)
  - mag_cont_mean   : magnitude moyenne des contaminants (-1 ignoré dans la moyenne)
  - ang_dist        : distance angulaire étoile/contaminant (-1 → NaN)
  - depth_prim      : profondeur du transit primaire [ppm] (-1 → NaN)
  - signal_type     : type de signal (planète ou EB)

Sorties :
  - n/e/s_mask_efficiency  : valeur réelle, NaN si -1 (absent)
  - sky_nominal_present    : 0/1 (présence dans 'produit')
  - sky_extended_present   : 0/1 (présence dans 'produit')

Usage :
  python analyse_efficacite.py --folder /chemin/vers/fichiers [--output stats.csv]
"""

import os, json, glob, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Helpers ─────────────────────────────────────────────────────────────────

def _to_float(val, absent=-1):
    """Float avec NaN si val == absent ou non convertible."""
    try:
        v = float(val)
        return np.nan if v == absent else v
    except (TypeError, ValueError):
        return np.nan


def _mean_list(lst, absent=-1):
    """Moyenne d'une liste en ignorant les valeurs absentes."""
    vals = []
    for x in lst:
        try:
            v = float(x)
            if v != absent:
                vals.append(v)
        except (TypeError, ValueError):
            pass
    return np.mean(vals) if vals else np.nan


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_records(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] {filepath}: {e}")
        return []

    params  = data.get("parametres", [{}])[0]
    donnees = data.get("donnees",    [])
    produit = data.get("produit",    [])

    qgc           = params.get("qgc", [[], [], -1])
    nb_cameras    = _to_float(qgc[2])
    nb_cont       = _to_float(params.get("nb_cont",       -1))
    mag_star      = _to_float(params.get("mag_star",       -1))
    ang_dist      = _to_float(params.get("Ang_dist",       -1))
    depth_prim    = _to_float(params.get("depth_prim_ldc", -1))
    signal_type   = params.get("Signal_type", "unknown")
    mag_cont_mean = _mean_list(params.get("mag_sec", []))

    has_sky_nominal  = int(any("nominal"  in p for p in produit))
    has_sky_extended = int(any("extended" in p for p in produit))

    sim = Path(filepath).stem
    records = []
    for d in donnees:
        records.append({
            "sim":               sim,
            "nb_cameras":        nb_cameras,
            "nb_cont":           nb_cont,
            "mag_star":          mag_star,
            "mag_cont_mean":     mag_cont_mean,
            "ang_dist":          ang_dist,
            "depth_prim":        depth_prim,
            "signal_type":       signal_type,
            "n_mask_efficiency": _to_float(d.get("n_mask_efficiency", -1)),
            "e_mask_efficiency": _to_float(d.get("e_mask_efficiency", -1)),
            "s_mask_efficiency": _to_float(d.get("s_mask_efficiency", -1)),
            "sky_nominal_present":  has_sky_nominal,
            "sky_extended_present": has_sky_extended,
        })
    return records


def load_all(folder):
    files = sorted(glob.glob(os.path.join(folder, "product_param_sim*.json")))
    if not files:
        raise FileNotFoundError(f"Aucun fichier dans : {folder}")
    print(f"  {len(files)} fichier(s) trouvé(s).")
    rows = []
    for fp in files:
        rows.extend(extract_records(fp))
    df = pd.DataFrame(rows)
    print(f"  {len(df)} lignes chargées.")
    return df


# ── Labels ───────────────────────────────────────────────────────────────────

PARAMS  = ["nb_cameras", "nb_cont", "mag_star", "mag_cont_mean", "ang_dist", "depth_prim"]
OUTPUTS = ["n_mask_efficiency", "e_mask_efficiency", "s_mask_efficiency",
           "sky_nominal_present", "sky_extended_present"]
PARAM_LABELS = {
    "nb_cameras":    "Nb caméras",
    "nb_cont":       "Nb contaminants",
    "mag_star":      "Mag étoile",
    "mag_cont_mean": "Mag cont. (moy.)",
    "ang_dist":      "Distance ang. [arcsec]",
    "depth_prim":    "Profondeur transit [ppm]",
}
OUTPUT_LABELS = {
    "n_mask_efficiency":   "Masque nominal (eff.)",
    "e_mask_efficiency":   "Masque étendu (eff.)",
    "s_mask_efficiency":   "Masque secondaire (eff.)",
    "sky_nominal_present": "Sky disp. nominal",
    "sky_extended_present":"Sky disp. étendu",
}


# ── Stats ────────────────────────────────────────────────────────────────────

def print_stats(df):
    print("\n══════════════════════════════════════════════════════")
    print("  STATISTIQUES — VERSION EFFICACITÉ")
    print("══════════════════════════════════════════════════════")
    print("\n── Paramètres d'entrée (NaN = donnée absente) ───────")
    print(df[PARAMS].describe().to_string())
    print("\n── Sorties (NaN exclus des calculs) ─────────────────")
    print(df[OUTPUTS].describe().to_string())
    print("\n── Répartition signal_type ──────────────────────────")
    print(df.groupby("signal_type")["sim"].nunique().rename("nb_sims").to_string())


def correlation_table(df):
    num = df[PARAMS + OUTPUTS].select_dtypes(include=[np.number])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = num[PARAMS].apply(lambda col: num[OUTPUTS].corrwith(col)).T
    return corr


# ── Graphiques ───────────────────────────────────────────────────────────────

def plot_correlations(corr, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels([OUTPUT_LABELS.get(c, c) for c in corr.columns],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels([PARAM_LABELS.get(r, r) for r in corr.index], fontsize=9)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Corrélation de Pearson")
    ax.set_title("Corrélation paramètres → sorties (efficacité)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "correlations_efficacite.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


def plot_scatter_grid(df, out_dir):
    mask_out = ["n_mask_efficiency", "e_mask_efficiency", "s_mask_efficiency"]
    sigs   = df["signal_type"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sigs)))
    cmap   = dict(zip(sigs, colors))
    fig, axes = plt.subplots(len(PARAMS), len(mask_out),
                             figsize=(4*len(mask_out), 3*len(PARAMS)), squeeze=False)
    for i, param in enumerate(PARAMS):
        for j, out in enumerate(mask_out):
            ax = axes[i][j]
            for sig, grp in df.groupby("signal_type"):
                valid = grp.dropna(subset=[param, out])
                ax.scatter(valid[param], valid[out], alpha=0.3, s=10,
                           color=cmap[sig], label=sig)
            ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=8)
            ax.set_ylabel(OUTPUT_LABELS.get(out, out), fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, markerscale=2)
    fig.suptitle("Paramètres vs Efficacité des masques", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "scatter_efficacite.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def plot_sky_boxplots(df, out_dir):
    for sky_col in ["sky_nominal_present", "sky_extended_present"]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, param in zip(axes.flatten(), PARAMS):
            groups = [df[df[sky_col] == v][param].dropna() for v in [0, 1]]
            ax.boxplot(groups, tick_labels=["Absent", "Présent"])
            ax.set_title(PARAM_LABELS.get(param, param), fontsize=9)
        label = "nominal" if "nominal" in sky_col else "étendu"
        fig.suptitle(f"Paramètres selon présence sky displacement {label}", fontsize=12)
        plt.tight_layout()
        path = os.path.join(out_dir, f"sky_{label}_boxplot.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"  → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse PLATO — efficacité")
    parser.add_argument("--folder",  default=".",                   help="Dossier JSON")
    parser.add_argument("--output",  default="stats_efficacite.csv")
    parser.add_argument("--plots",   default="plots_efficacite")
    args = parser.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    print("\n[1/4] Chargement...")
    df = load_all(args.folder)

    print("\n[2/4] Statistiques descriptives...")
    print_stats(df)

    print("\n[3/4] Corrélations (Pearson, NaN exclus)...")
    corr = correlation_table(df)
    print(corr.round(3).to_string())
    print("\nPar type de signal :")
    print(df.groupby("signal_type")[OUTPUTS].agg(["mean","std","median"]).round(4).to_string())

    print("\n[4/4] Graphiques...")
    plot_correlations(corr, args.plots)
    plot_scatter_grid(df, args.plots)
    plot_sky_boxplots(df, args.plots)

    df.to_csv(args.output, index=False)
    print(f"\n  → CSV : {args.output}\nTerminé ✓")


if __name__ == "__main__":
    main()
