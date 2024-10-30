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
from nltk.tokenize import sent_tokenize
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


# Function to load paragraphs from the "contexte" column of a CSV file
def load_paragraphs_from_csv(csv_file_path: str) -> list[str]:
    """
    Charge les paragraphes depuis la colonne 'contexte' d'un fichier CSV.

    Args:
        csv_file_path (str): Chemin vers le fichier CSV.

    Returns:
        list[str]: Liste des paragraphes provenant de la colonne 'contexte'.
    """
    df = pd.read_csv(csv_file_path)
    paragraphs = df['contexte'].dropna().tolist()
    return paragraphs


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


# Charger les questions depuis le fichier CSV
def charger_questions(chemin_csv):
    df = pd.read_csv(chemin_csv)
    return df


# Sauvegarder les résultats RAG dans un fichier CSV
def sauvegarder_mentions_csv(mentions, chemin_csv):
    df_mentions = pd.DataFrame(mentions)
    df_mentions.to_csv(chemin_csv, index=False)
    print(f"Mentions sauvegardées dans le fichier {chemin_csv}")

# Charger les paragraphes et les mentions du GIEC


def charger_paragraphes_et_mentions(chemin_paragraphes_csv, chemin_mentions_csv):
    paragraphes_df = pd.read_csv(chemin_paragraphes_csv)
    mentions_df = pd.read_csv(chemin_mentions_csv)
    return paragraphes_df, mentions_df


# Sauvegarder les résultats d'évaluation
def sauvegarder_resultats_evaluation(resultats, chemin_resultats_csv):
    resultats.to_csv(chemin_resultats_csv, index=False)
    print(f"Résultats d'évaluation sauvegardés dans {chemin_resultats_csv}")


# Fonction pour lire un fichier local et créer des paragraphes toutes les 4 phrases
def load_and_group_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = sent_tokenize(text)  # Divise le texte en phrases
    return sentences


# Sauvegarder les résultats dans un fichier CSV
def save_results_to_csv(results, output_path="climate_analysis_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")



# Charger les phrases et les sections extraites du fichier rag_results.csv
def charger_rag_results(chemin_rag_csv):
    rag_df = pd.read_csv(chemin_rag_csv)
    return rag_df


# Sauvegarder les résultats de résumé
def sauvegarder_resultats_resume(resultats, chemin_resultats_csv):
    resultats.to_csv(chemin_resultats_csv, index=False)
    print(f"Résumés sauvegardés dans le fichier {chemin_resultats_csv}")


