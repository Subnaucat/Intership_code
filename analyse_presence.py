"""
Analyse statistique des simulations PLATO - VERSION PRÉSENCE/ABSENCE
=====================================================================
Règle d'absence : toute valeur == -1 est considérée comme ABSENTE.
  - Paramètres  : -1 → NaN (exclu des statistiques)
  - Masques     : -1 → absent (0), toute autre valeur → présent (1)
  - Sky disp.   : présence/absence dans la liste 'produit'

Usage :
  python analyse_presence.py --folder /chemin/vers/fichiers [--output stats.csv]
"""

import os, json, glob, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Helpers ─────────────────────────────────────────────────────────────────

def _to_float(val, absent=-1):
    try:
        v = float(val)
        return np.nan if v == absent else v
    except (TypeError, ValueError):
        return np.nan


def _mean_list(lst, absent=-1):
    vals = []
    for x in lst:
        try:
            v = float(x)
            if v != absent:
                vals.append(v)
        except (TypeError, ValueError):
            pass
    return np.mean(vals) if vals else np.nan


def _present(val, absent=-1):
    """Retourne 1 si val != absent ET n'est pas None, sinon 0."""
    try:
        return 0 if float(val) == absent else 1
    except (TypeError, ValueError):
        return 0


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
    print(ang_dist)
    depth_prim    = _to_float(params.get("depth_prim", -1))
    signal_type   = params.get("Signal_type", "unknown")
    mag_sec_list = params.get("mag_sec", [])
    
    if mag_sec_list and len(mag_sec_list) > 0:
        mag_cont_prim = _to_float(min(mag_sec_list))
        delta_mag = mag_cont_prim - mag_star if not np.isnan(mag_star) else np.nan
    else:
        delta_mag = np.nan
    has_sky_nominal  = int(any("nominal"  in p for p in produit))
    has_sky_extended = int(any("extended" in p for p in produit))

    sim = Path(filepath).stem
    records = []
    for d in donnees:
        records.append({
            "sim":               sim,
            "nb_cameras":        nb_cameras,
            "nb_cont":           nb_cont,
            "delta_mag":         delta_mag,
            "ang_dist":          ang_dist,
            "depth_prim":        depth_prim,
            "signal_type":       signal_type,
            # Binaire : 1 = présent, 0 = absent (-1)
            "n_mask_present":    _present(d.get("n_mask_efficiency", -1)),
            "e_mask_present":    _present(d.get("e_mask_efficiency", -1)),
            "s_mask_present":    _present(d.get("s_mask_efficiency", -1)),
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

PARAMS  = ["nb_cameras", "nb_cont", "delta_mag", "ang_dist", "depth_prim"]
OUTPUTS = ["n_mask_present", "e_mask_present", "s_mask_present",
           "sky_nominal_present", "sky_extended_present"]
PARAM_LABELS = {
    "nb_cameras":    "Nb caméras",
    "nb_cont":       "Nb contaminants",
    "delta_mag":     "Δ Mag (Cont. - Star)",
    "ang_dist":      "Distance ang. [arcsec]",
    "depth_prim":    "Profondeur transit [ppm]",
}
OUTPUT_LABELS = {
    "n_mask_present":      "Nominal mask ",
    "e_mask_present":      "Extended mask",
    "s_mask_present":      "Secondary mask",
    "sky_nominal_present": "Nominal Sky disp. ",
    "sky_extended_present":"Extended Sky disp.",
}


# ── Stats ────────────────────────────────────────────────────────────────────

def print_stats(df):
    print("\n══════════════════════════════════════════════════════")
    print("  STATISTIQUES — VERSION PRÉSENCE/ABSENCE")
    print("══════════════════════════════════════════════════════")
    print("\n── Paramètres d'entrée (NaN = donnée absente) ───────")
    print(df[PARAMS].describe().to_string())
    print("\n── Rate de présence des sorties (%) ─────────────────")
    rates = df[OUTPUTS].mean() * 100
    for col, rate in rates.items():
        print(f"  {OUTPUT_LABELS[col]:<25} : {rate:.1f}%")
    print("\n── Répartition signal_type ──────────────────────────")
    print(df.groupby("signal_type")["sim"].nunique().rename("nb_sims").to_string())


def presence_rate_by_param(df, n_bins=5):
    results = {}
    for param in PARAMS:
        col = df[param].dropna()
        if col.nunique() <= n_bins:
            grouped = df.groupby(param)[OUTPUTS].mean() * 100
        else:
            bins   = pd.qcut(df[param], q=n_bins, duplicates="drop")
            grouped = df.groupby(bins, observed=True)[OUTPUTS].mean() * 100
        results[param] = grouped
    return results


# ── Graphiques ───────────────────────────────────────────────────────────────

def plot_presence_rates(df, out_dir):
    rates  = df[OUTPUTS].mean() * 100
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([OUTPUT_LABELS[o] for o in OUTPUTS], rates.values,
                  color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Occurrence rate (%)", fontsize=11)
    ax.set_title("Global occurrence rate global per product", fontsize=13)
    ax.axhline(50, color="grey", lw=0.8, linestyle="--", label="50%")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "Rate_presence_global.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


def plot_presence_by_param(param_rates, out_dir):
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for param, table in param_rates.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(table))
        for j, out in enumerate(OUTPUTS):
            if out in table.columns:
                ax.plot(x, table[out].values, marker="o",
                        color=colors[j], label=OUTPUT_LABELS[out], linewidth=2)
        x_labels = [str(idx) for idx in table.index]
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8)
        ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=10)
        ax.set_ylabel("Occurrence rate (%)", fontsize=10)
        ax.set_ylim(-5, 110)
        ax.axhline(50, color="grey", lw=0.8, linestyle="--")
        ax.legend(fontsize=8)
        ax.set_title(f"Availability of the product based on {PARAM_LABELS.get(param, param)}", fontsize=12)
        plt.tight_layout()
        path = os.path.join(out_dir, f"presence_par_{param}.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"  → {path}")


def plot_heatmap(param_rates, out_dir):
    matrix = np.zeros((len(PARAMS), len(OUTPUTS)))
    for i, (param, table) in enumerate(param_rates.items()):
        for j, out in enumerate(OUTPUTS):
            if out in table.columns:
                vals = table[out].values
                if len(vals) > 1:
                    matrix[i, j] = np.corrcoef(range(len(vals)), vals)[0, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(OUTPUTS)))
    ax.set_yticks(range(len(PARAMS)))
    ax.set_xticklabels([OUTPUT_LABELS[o] for o in OUTPUTS], rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in PARAMS], fontsize=9)
    for i in range(len(PARAMS)):
        for j in range(len(OUTPUTS)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(matrix[i,j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Tendency (Pearson sur tranches)")
    ax.set_title("Parameter tendency → presence of the product", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "heatmap_presence.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


def plot_by_signal_type(df, out_dir):
    table = df.groupby("signal_type")[OUTPUTS].mean() * 100
    sigs  = table.index.tolist()
    x     = np.arange(len(OUTPUTS))
    width = 0.8 / len(sigs)
    colors = plt.cm.Set2(np.linspace(0, 1, len(sigs)))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, sig in enumerate(sigs):
        offset = (i - len(sigs)/2 + 0.5) * width
        vals   = [table.loc[sig, o] if o in table.columns else 0 for o in OUTPUTS]
        ax.bar(x + offset, vals, width*0.9, label=sig, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([OUTPUT_LABELS[o] for o in OUTPUTS], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Rate de présence (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(50, color="grey", lw=0.8, linestyle="--")
    ax.legend(title="Signal type", fontsize=8)
    ax.set_title("Availability of product by signal type", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "presence_par_signal_type.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse PLATO — présence/absence")
    parser.add_argument("--folder",  default=".",                  help="Dossier JSON")
    parser.add_argument("--output",  default="stats_presence.csv")
    parser.add_argument("--plots",   default="plots_presence")
    parser.add_argument("--bins",    type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    print("\n[1/5] Chargement...")
    df = load_all(args.folder)

    print("\n[2/5] Statistiques descriptives...")
    print_stats(df)

    print("\n[3/5] Rate de présence par tranche de paramètre...")
    param_rates = presence_rate_by_param(df, n_bins=args.bins)
    for param, table in param_rates.items():
        print(f"\n  ── {PARAM_LABELS.get(param, param)} ──")
        print(table.round(1).to_string())
    print("\n  ── Per sygnal type ──")
    print((df.groupby("signal_type")[OUTPUTS].mean() * 100).round(1).to_string())

    print("\n[4/5] Graphiques...")
    plot_presence_rates(df, args.plots)
    plot_presence_by_param(param_rates, args.plots)
    plot_heatmap(param_rates, args.plots)
    plot_by_signal_type(df, args.plots)

    print("\n[5/5] Export CSV...")
    df.to_csv(args.output, index=False)
    print(f"  → CSV : {args.output}\nTerminé ✓")


if __name__ == "__main__":
    main()
