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
from llama_index.core import Settings
from pdfminer.high_level import extract_text
from txt_manipulation import parse_llm_response


# Fonction pour charger le glossaire traduit avec termes et définitions
def charger_glossaire(chemin_glossaire: str) -> tuple[list[str], list[str]]:
    """
    Charge le glossaire traduit du GIEC depuis un fichier CSV.

    Args:
        chemin_glossaire (str): Chemin vers le fichier CSV contenant le glossaire.

    Returns:
        tuple: Une liste de termes traduits et une liste des définitions correspondantes.
    """
    glossaire = pd.read_csv(chemin_glossaire)
    # Extraire les termes traduits
    termes = glossaire['Translated_Term'].tolist()
    # Extraire les définitions traduites
    definitions = glossaire['Translated_Definition'].tolist()
    return termes, definitions


# Fonction pour sauvegarder le texte nettoyé
def save_cleaned_text(text: str, filepath: str) -> None:
    """
    Sauvegarde le texte nettoyé dans un fichier texte.

    Args:
        text (str): Texte à sauvegarder.
        filepath (str): Chemin du fichier de destination.
    """
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(text)


# Fonction pour charger le texte brut
def load_text(filepath: str) -> str:
    """
    Charge le texte brut à partir d'un fichier.

    Args:
        filepath (str): Chemin vers le fichier texte.

    Returns:
        str: Contenu du fichier texte.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


# Fonction pour extraire et nettoyer le texte d'un PDF (rapport GIEC)
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrait le texte d'un fichier PDF.

    Args:
        pdf_path (str): Chemin vers le fichier PDF.

    Returns:
        str: Texte extrait du PDF.
    """
    return extract_text(pdf_path)


# Fonction pour sauvegarder les sections indexées dans un fichier JSON
def save_database(sections: list[dict], output_path: str) -> None:
    """
    Sauvegarde les sections indexées dans un fichier JSON.

    Args:
        sections (list[dict]): Liste des sections à sauvegarder.
        output_path (str): Chemin du fichier JSON de destination.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
    print(f"Sections sauvegardées dans {output_path}")


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
    embeddings = [Settings.embed_model.get_text_embedding(section) for section in sections]
    return embeddings, sections


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


# Function to load paragraphs from the "contexte" column of a CSV file
def load_paragraphs_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    paragraphs = df['contexte'].dropna().tolist()
    return paragraphs



# Function to parse all responses and create a DataFrame
def create_final_dataframe(df):
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
