"""
Ce script permet de manipuler et de nettoyer des fichiers texte, en appliquant diverses techniques de prétraitement.
Il comprend des fonctions pour effectuer des nettoyages basés sur des expressions régulières, pour découper le texte en sections ou en phrases, et pour sauvegarder les résultats nettoyés.

Fonctionnalités principales :
- Nettoyage manuel avec des expressions régulières
- Découpage du texte en sections et en phrases
- Sauvegarde des textes nettoyés dans des fichiers
- Chargement de texte à partir de fichiers et gestion des dossiers de sortie

"""

import os
import re
import nltk
from file_utils import load_text, save_cleaned_text


# Fonction pour appliquer un pré-nettoyage manuel avec regex
def pre_nettoyage_regex(texte: str) -> str:
    """
    Applique un pré-nettoyage manuel au texte à l'aide d'expressions régulières.

    Args:
        texte (str): Texte à nettoyer.

    Returns:
        str: Texte nettoyé.
    """
    print("Application du pré-nettoyage...")
    # Supprimer les sauts de ligne multiples
    texte = re.sub(r'\n+', '\n', texte)
    texte = re.sub(r'\s+', ' ', texte)  # Supprimer les espaces multiples
    texte = re.sub(r'©.*\d{4}', '', texte)  # Supprimer les droits d'auteur
    texte = re.sub(r'(Rédigé par .+|Publié le .+)', '', texte)  # Supprimer les métadonnées auteurs
    texte = re.sub(r'http\S+', '', texte)  # Supprimer les URLs
    # Supprimer les espaces en début et fin de ligne
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


# Fonction pour prétraiter l'article et sauvegarder le texte nettoyé
def pretraiter_article(chemin_article: str, chemin_dossier_nettoye: str) -> str:
    """
    Prétraite un article en appliquant le pré-nettoyage et en sauvegardant le texte nettoyé dans un fichier.

    Args:
        chemin_article (str): Chemin du fichier texte de l'article à prétraiter.
        chemin_dossier_nettoye (str): Dossier où sauvegarder l'article nettoyé.

    Returns:
        str: Chemin du fichier de l'article nettoyé.
    """
    texte_article = load_text(chemin_article)
    texte_pre_nettoye = pre_nettoyage_regex(texte_article)

    if not os.path.exists(chemin_dossier_nettoye):
        os.makedirs(chemin_dossier_nettoye)

    chemin_cleaned_article = os.path.join(chemin_dossier_nettoye, os.path.basename(chemin_article).replace('.txt', '_cleaned.txt'))
    save_cleaned_text(texte_pre_nettoye, chemin_cleaned_article)
    print(f"Texte pré-nettoyé sauvegardé dans {chemin_cleaned_article}")
    return chemin_cleaned_article


# Fonction pour nettoyer le texte de manière simple
def clean_text(text: str) -> str:
    """
    Nettoie un texte en supprimant les numéros de page, les espaces multiples et les sauts de ligne supplémentaires.

    Args:
        text (str): Texte à nettoyer.

    Returns:
        str: Texte nettoyé.
    """
    text = re.sub(r'\bPage\s+\d+\b', '', text)  # Supprimer les numéros de page
    text = re.sub(r'\s+', ' ', text)  # Supprimer les espaces multiples
    text = re.sub(r'\n+', '\n', text)  # Supprimer les sauts de ligne en excès
    return text


# Fonction pour découper le texte en sections
def split_text_by_sections(text: str) -> list[dict]:
    """
    Découpe un texte en sections à partir d'un modèle de section basé sur des expressions régulières.

    Args:
        text (str): Texte à découper en sections.

    Returns:
        list: Liste de dictionnaires contenant les titres et le contenu des sections.
    """
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
def decouper_en_phrases(texte: str) -> list[str]:
    """
    Découpe un texte en phrases à l'aide de l'outil de tokenisation de NLTK.

    Args:
        texte (str): Texte à découper.

    Returns:
        list: Liste de phrases extraites du texte.
    """
    print("Découpage du texte en phrases...")
    phrases = nltk.sent_tokenize(texte)
    print(f"{len(phrases)} phrases trouvées.")
    return phrases
