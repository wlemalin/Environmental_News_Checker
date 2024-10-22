import os
import re
import nltk
from file_utils import load_text, save_cleaned_text


# Fonction pour appliquer un pré-nettoyage manuel avec regex
def pre_nettoyage_regex(texte):
    print("Application du pré-nettoyage...")
    # Supprimer les sauts de ligne multiples
    texte = re.sub(r'\n+', '\n', texte)
    texte = re.sub(r'\s+', ' ', texte)  # Supprimer les espaces multiples
    texte = re.sub(r'©.*\d{4}', '', texte)  # Supprimer les droits d'auteur
    texte = re.sub(r'(Rédigé par .+|Publié le .+)',
                   '', texte)  # Métadonnées auteurs
    texte = re.sub(r'http\S+', '', texte)  # Supprimer les URLs
    # Espaces début et fin de ligne
    texte = re.sub(r'^\s+|\s+$', '', texte, flags=re.MULTILINE)

    # Supprimer les doublons de paragraphes
    paragraphs = texte.split('\n')
    texte_unique = []
    for paragraph in paragraphs:
        if paragraph not in texte_unique:
            texte_unique.append(paragraph)
    texte = '\n'.join(texte_unique)

    print("Pré-nettoyage terminé.")
    return texte


# Fonction pour prétraiter l'article
def pretraiter_article(chemin_article, chemin_dossier_nettoye):
    texte_article = load_text(chemin_article)
    texte_pre_nettoye = pre_nettoyage_regex(texte_article)

    if not os.path.exists(chemin_dossier_nettoye):
        os.makedirs(chemin_dossier_nettoye)

    chemin_cleaned_article = os.path.join(chemin_dossier_nettoye, os.path.basename(
        chemin_article).replace('.txt', '_cleaned.txt'))
    save_cleaned_text(texte_pre_nettoye, chemin_cleaned_article)
    print(f"Texte pré-nettoyé sauvegardé dans {chemin_cleaned_article}")
    return chemin_cleaned_article


def clean_text(text):
    text = re.sub(r'\bPage\s+\d+\b', '', text)  # Remove page numbers
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'\n+', '\n', text)  # Remove extra newlines
    return text


def split_text_by_sections(text):
    section_pattern = r'([A-Z]\.\d+(\.\d+)*)\s+|\b(Partie|Chapitre)\b\s+\d+[:\.\s]+\b'
    matches = list(re.finditer(section_pattern, text))
    sections = []

    for i, match in enumerate(matches):
        section_title = match.group(0)
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i < len(matches) - 1 else len(text)
        section_text = text[start_idx:end_idx].strip()
        sections.append({
            "title": section_title,
            "text": section_text
        })
    return sections


# Fonction pour découper le texte en phrases
def decouper_en_phrases(texte):
    print("Découpage du texte en phrases...")
    phrases = nltk.sent_tokenize(texte)
    print(f"{len(phrases)} phrases trouvées.")
    return phrases
