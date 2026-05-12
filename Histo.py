# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:23:56 2026

@author: Zammit
"""

import os
import json
import matplotlib.pyplot as plt
from collections import Counter

def analyser_diversite_par_fichier(dossier_source):
    # Cette liste stockera le "compte de valeurs uniques" pour chaque fichier
    # Exemple : [3, 5, 2, 3, 8...]
    diversite_par_fichier = []
    
    if not os.path.exists(dossier_source):
        print(f"Le dossier '{dossier_source}' n'existe pas.")
        return

    for nom_fichier in os.listdir(dossier_source):
        if nom_fichier.endswith('.json'):
            chemin_complet = os.path.join(dossier_source, nom_fichier)
            
            try:
                with open(chemin_complet, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    valeurs_uniques_locales = set()
                    
                    if "donnees" in data:
                        elements = data["donnees"]
                        # On s'assure d'itérer sur une liste
                        if isinstance(elements, dict): elements = [elements]
                        
                        for item in elements:
                            val = item.get("metric_priority")
                            if val is not None:
                                valeurs_uniques_locales.add(val)
                    
                    # On enregistre COMBIEN il y en avait de différentes dans CE fichier
                    diversite_par_fichier.append(len(valeurs_uniques_locales))
                    print(diversite_par_fichier[-1],nom_fichier)
            except Exception as e:
                print(f"Erreur sur {nom_fichier}: {e}")

    if not diversite_par_fichier:
        print("Aucune donnée analysée.")
        return

    # --- Histogramme ---
    plt.figure(figsize=(10, 6))
    
    # On utilise les valeurs réelles pour définir les colonnes (bins)
    max_div = max(diversite_par_fichier)
    min_div = min(diversite_par_fichier)
    
    # Création de l'histogramme
    # bins=range(...) permet d'aligner les barres sur les nombres entiers
    plt.hist(diversite_par_fichier, bins=range(min_div, max_div + 2), 
             color='#69b3a2', edgecolor='white', align='left')

    plt.title('Répartition de la diversité des priorités par fichier', fontsize=13)
    plt.xlabel('Nombre de valeurs "metric_priority" différentes au sein d\'un fichier', fontsize=11)
    plt.ylabel('Nombre de fichiers', fontsize=11)
    
    # Forcer l'affichage de tous les entiers sur l'axe X
    plt.xticks(range(min_div, max_div + 1))
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Utilisation
# analyser_diversite_par_fichier('votre_dossier_json')