"""
Ce script détecte les termes du glossaire dans un article, génère des fenêtres contextuelles à partir des phrases de l'article,
et compare ces fenêtres avec des sections d'un rapport en utilisant les embeddings. Il ajuste dynamiquement le seuil de similarité et 
génère des mentions lorsque des sections similaires sont trouvées.

Fonctionnalités principales :
- Détection de termes du glossaire dans une phrase
- Génération de fenêtres contextuelles à partir de phrases
- Comparaison de fenêtres contextuelles avec des sections du rapport en utilisant les embeddings
- Détection de termes du glossaire et ajout de définitions correspondantes dans les résultats

"""

import nltk
import numpy as np
from llama_index.core import Settings
from sentence_transformers import util


# Fonction pour détecter les termes du glossaire dans une phrase
def detect_glossary_terms(phrase: str, termes_glossaire: list[str]) -> list[str]:
    """
    Détecte les termes du glossaire dans une phrase donnée.

    Args:
        phrase (str): La phrase dans laquelle rechercher les termes du glossaire.
        termes_glossaire (list[str]): Liste des termes du glossaire.

    Returns:
        list: Liste des termes du glossaire trouvés dans la phrase.
    """
    phrase_lower = phrase.lower()
    return [term for term in termes_glossaire if term.lower() in phrase_lower]


def generate_context_windows(phrases: list[str], window_size: int = 3) -> list[dict]:
    """
    Génère des fenêtres contextuelles à partir de la liste des phrases.

    Args:
        phrases (list[str]): Liste de phrases à regrouper en fenêtres contextuelles.
        window_size (int): Taille de la fenêtre (nombre de phrases par fenêtre).

    Returns:
        list: Liste des fenêtres contextuelles sous forme de dictionnaires.
    """
    windows = []
    for i in range(0, len(phrases)):
        context_window = " ".join(phrases[max(0, i - window_size):min(i + window_size + 1, len(phrases))])
        windows.append({"context": context_window, "current_phrase": phrases[i]})
    return windows

# Fonction pour comparer les phrases d'un article avec les sections d'un rapport en utilisant des fenêtres contextuelles
def keywords_for_each_chunck(phrases_article: list[str], termes_glossaire: list[str], definitions_glossaire: list[str], window_size: int = 3) -> list[dict]:
    """
    Compare les phrases de l'article avec les sections du rapport en générant des fenêtres contextuelles et en utilisant les embeddings.

    Args:
        phrases_article (list[str]): Liste des phrases de l'article.
        termes_glossaire (list[str]): Liste des termes du glossaire.
        definitions_glossaire (list[str]): Liste des définitions des termes du glossaire.
        window_size (int): Taille de la fenêtre contextuelle (nombre de phrases par fenêtre).

    Returns:
        list[dict]: Liste des mentions trouvées, comprenant les phrases, sections, similarités et termes du glossaire associés.
    """
    print("Comparaison des phrases de l'article avec les sections du rapport...")
    mentions = []

    # Générer les fenêtres contextuelles à partir des phrases de l'article
    context_windows = generate_context_windows(phrases_article, window_size)

    # Générer les embeddings pour les fenêtres contextuelles
    print("Génération des embeddings des fenêtres contextuelles...")
    # Itérer à travers les sections et trouver les sections pertinentes
    for i in range(0,len(context_windows)):
            glossary_terms = detect_glossary_terms(context_windows[i]["current_phrase"], termes_glossaire)
            mentions.append({
                # "article_title":
                "phrase": context_windows[i]["current_phrase"],  # Récupération de la phrase
                "contexte": context_windows[i]["context"],
                "glossary_terms": glossary_terms,
                # Ajouter les définitions correspondantes aux termes du glossaire détectés
                "definitions": [definitions_glossaire[termes_glossaire.index(term)] for term in glossary_terms if term in termes_glossaire]
            })
    print(f"{len(mentions)} mentions trouvées.")
    return mentions
