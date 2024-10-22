
# Chargement et nettoyage du glossaire GIEC
import csv
import json
import os

import pandas as pd
from llama_index.core import Settings
from pdfminer.high_level import extract_text


# Fonction pour charger le glossaire traduit avec termes et définitions
def charger_glossaire(chemin_glossaire):
    glossaire = pd.read_csv(chemin_glossaire)
    # Utiliser les termes traduits
    termes = glossaire['Translated_Term'].tolist()
    # Utiliser les définitions traduites
    definitions = glossaire['Translated_Definition'].tolist()
    return termes, definitions


# Fonction pour sauvegarder le texte nettoyé
def save_cleaned_text(text, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(text)


# Fonction pour charger le texte brut
def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


# Step 1: Extract and clean text from the PDF (rapport GIEC)
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


# Step 4: Save the indexed sections to a JSON file
def save_database(sections, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
    print(f"Sections saved to {output_path}")



# Fonction pour charger les embeddings du rapport
def charger_embeddings_rapport(chemin_rapport_embeddings):
    with open(chemin_rapport_embeddings, 'r', encoding='utf-8') as file:
        data = json.load(file)
    sections = [section['text'] for section in data]
    embeddings = [Settings.embed_model.get_text_embedding(
        section) for section in sections]
    return embeddings, sections


# Fonction pour sauvegarder les correspondances dans un fichier CSV
def sauvegarder_mentions_csv(mentions, chemin_csv, fieldnames):
    fichier_existe = os.path.exists(chemin_csv)
    with open(chemin_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not fichier_existe:
            writer.writeheader()
        for mention in mentions:
            writer.writerow(mention)
    print(f"Mentions sauvegardées dans le fichier {chemin_csv}")
