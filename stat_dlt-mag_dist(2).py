# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:51:40 2026

@author: Zammit
"""
import json
import os
import pandas as pd
import numpy as np

# Configuration
directory = "."  # Dossier contenant tes fichiers JSON
prefix = "product_param_sim"
start_sim = 60
end_sim = 250

all_results = []

for i in range(start_sim, end_sim + 1):
    # Formatage du nom de fichier sim00060, sim00061, etc.
    filename = f"{prefix}{str(i).zfill(5)}.json"
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        continue  # On passe si la simulation n'existe pas

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # --- Extraction des Paramètres ---
        params = data['parametres'][0]
        dist = params['Ang_dist']
        mag_star = params['mag_star']
        # Calcul de la différence de magnitude (delta mag)
        # On utilise le min de mag_sec comme dans ton exemple
        dlt_mag = mag_star - min(params['mag_sec'])
        
        # --- Analyse des Produits ---
        row = {
            "sim_id": i,
            "dist": dist,
            "dlt_mag": dlt_mag,
            "sky_nominal": 0,
            "sky_extended": 0,
            "mask_nominal": 0,
            "mask_extended": 0,
            "mask_secondary": 0
        }
        
        # Logique Sky Displacement (basée sur ton ratio)
        # On vérifie si la simulation a généré ces produits
        qgc_len = len(params['qgc'][0]) #[cite: 1]
        qgc_sub_len = len(params['qgc'][1][-1]) #[cite: 1]
        with open(filepath) as f:
            data = json.load(f)
        produit = data.get("produit",    [])
        has_sky_nominal  = int(any("nominal"  in p for p in produit))
        has_sky_extended = int(any("extended" in p for p in produit))
        if has_sky_nominal != 0:
            row["sky_nominal"] = 1
            
        if has_sky_extended != 0:
            row["sky_extended"] = 1

        # Logique des Masques (on regarde le premier élément de 'donnees')[cite: 1, 2]
        if data['donnees']:
            first_entry = data['donnees'][0]
            if first_entry.get('n_mask_efficiency', -1) != -1:
                row["mask_nominal"] = 1
            if first_entry.get('e_mask_efficiency', -1) != -1:
                row["mask_extended"] = 1
            if first_entry.get('s_mask_efficiency', -1) != -1:
                row["mask_secondary"] = 1
                
        all_results.append(row)
        
    except Exception as e:
        print(f"Erreur lors de la lecture de {filename}: {e}")

# Création du DataFrame final
df = pd.DataFrame(all_results)

# --- Statistiques ---
print("### Aperçu des données ###")
pd.set_option('display.max_columns', None) 
print(df)

print("\n### Taux d'apparition par type de produit ###")
stats_columns = ["sky_nominal", "sky_extended", "mask_nominal", "mask_extended", "mask_secondary"]
print(df[stats_columns].mean() * 100) # Pourcentage d'apparition