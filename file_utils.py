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



# Fonction pour charger les embeddings du rapport
def charger_embeddings_rapport(chemin_rapport_embeddings: str) -> tuple[list, list]:
    """
    Charge les embeddings du rapport ainsi que les sections correspondantes depuis un fichier JSON.

    Args:
        chemin_rapport_embeddings (str): Chemin vers le fichier JSON contenant les embeddings et les sections.

    Returns:
        tuple: Liste des embeddings et des sections de texte du rapport.
    """
    with open(chemin_rapport_embeddings, 'r', encoding='utf-8') as file:
        data = json.load(file)

    sections = [section['text'] for section in data]
    embeddings = [section['embedding'] for section in data]
    titles = [section.get('title', 'Section sans titre')
              for section in data]  # Get section titles or default

    return embeddings, sections, titles

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




# Function to parse all responses and create a DataFrame
def create_final_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse toutes les réponses d'un DataFrame et crée un DataFrame final.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les paragraphes et réponses analysées.

    Returns:
        pd.DataFrame: Un nouveau DataFrame contenant les paragraphes, les réponses binaires et les sujets associés.
    """
    parsed_data = []
    for _, row in df.iterrows():
        paragraph = row['paragraph']
        response = row['climate_related']
        binary_response, subjects_list = parse_llm_response(response)
        parsed_data.append({
            "paragraph": paragraph,
            "binary_response": binary_response,
            "subjects": subjects_list
        })

    return pd.DataFrame(parsed_data)




