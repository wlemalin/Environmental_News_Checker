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
def pretraiter_article(chemin_article: str, chemin_cleaned_article, chemin_dossier_nettoye: str) -> str:
    """
    Prétraite un article en appliquant le pré-nettoyage et en sauvegardant le texte nettoyé dans un fichier.

    Args:
        chemin_article (str): Chemin du fichier texte de l'article à prétraiter.
        chemin_dossier_nettoye (str): Dossier où sauvegarder l'article nettoyé.

    Returns:
        str: Chemin du fichier de l'article nettoyé.
    """
    with open(chemin_article, 'r', encoding='utf-8') as file:
        texte_article = file.read()
    texte_pre_nettoye = pre_nettoyage_regex(texte_article)

    if not os.path.exists(chemin_dossier_nettoye):
        os.makedirs(chemin_dossier_nettoye)

    with open(chemin_cleaned_article, 'w', encoding='utf-8') as file:
        file.write(texte_pre_nettoye)
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


# Function to split text into optimized chunks using NLTK for sentence tokenization
def split_text_into_chunks(text: str, max_tokens: int = 382) -> list[dict]:
    """
    Splits a text into optimized chunks for embedding generation, ensuring each chunk is within the token limit.

    Args:
        text (str): The input text to split into chunks.
        max_tokens (int): The maximum number of tokens for each chunk (default is 382 for the model).

    Returns:
        list: A list of dictionaries containing the index and content of each chunk.
    """
    # Split the text into sentences using NLTK
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    chunk_index = 0

    for sentence in sentences:
        # Check if adding the next sentence exceeds the max token limit
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk += sentence + " "
        else:
            # Add the current chunk to the list
            chunks.append({
                "title": f"Chunk {chunk_index}",
                "text": current_chunk.strip()
            })
            chunk_index += 1
            # Start a new chunk with the current sentence
            current_chunk = sentence + " "

    # Add the last chunk if it has any remaining text
    if current_chunk:
        chunks.append({
            "title": f"Chunk {chunk_index}",
            "text": current_chunk.strip()
        })

    return chunks


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


# Function to parse the LLM's response
def parse_llm_response(response: str) -> tuple[str, list[str]]:
    """
    Parse la réponse d'un modèle LLM pour extraire une réponse binaire et une liste de sujets abordés.

    Args:
        response (str): La réponse du modèle LLM à analyser.

    Returns:
        tuple: Une réponse binaire ('0' ou '1') et une liste des sujets abordés.
    """
    binary_match = re.search(r"Réponse binaire\s?\(0 ou 1\)\s?:?\s?(\d)", response)
    binary_response = binary_match.group(1) if binary_match else None
    
    # Extraire la liste des sujets abordés
    subjects_match = re.search(r"Liste des sujets abordés\s?:?\s*(\[.*?\])", response, re.DOTALL)
    if subjects_match:
        subjects_list = subjects_match.group(1).strip("[]").split(",")
        subjects_list = [subject.strip() for subject in subjects_list]
    else:
        # Cas où la liste des sujets n'est pas dans un format de liste explicite
        subjects_match_alt = re.search(r"Liste des sujets abordés\s?:?\s*(.*)", response, re.DOTALL)
        subjects_list = subjects_match_alt.group(1).strip().split(",") if subjects_match_alt else []
    
    return binary_response, subjects_list
