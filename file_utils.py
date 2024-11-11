"""
Ce script gère le chargement, nettoyage, et sauvegarde de données issues du glossaire du GIEC et d'un rapport PDF.
Il inclut des fonctions pour extraire du texte, gérer des embeddings et sauvegarder des résultats dans des fichiers JSON et CSV.

Fonctionnalités principales :
- Chargement du glossaire traduit avec termes et définitions
- Extraction de texte d'un fichier PDF
- Sauvegarde de texte nettoyé et de sections indexées
- Chargement et gestion des embeddings du rapport
- Sauvegarde des correspondances dans un fichier CSV

"""

import csv
import json
import os
import pandas as pd
from txt_manipulation import parse_llm_response



# Fonction pour sauvegarder les mentions ou correspondances dans un fichier CSV


def save_to_csv(mentions: list[dict], chemin_csv: str, fieldnames: list[str]) -> None:
    """
    Sauvegarde les mentions ou correspondances dans un fichier CSV.

    Args:
        mentions (list[dict]): Liste des mentions à sauvegarder.
        chemin_csv (str): Chemin vers le fichier CSV.
        fieldnames (list[str]): Liste des noms de colonnes pour le fichier CSV.
    """
    fichier_existe = os.path.exists(chemin_csv)
    with open(chemin_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not fichier_existe:
            writer.writeheader()
        for mention in mentions:
            writer.writerow(mention)
    print(f"Mentions sauvegardées dans le fichier {chemin_csv}")







