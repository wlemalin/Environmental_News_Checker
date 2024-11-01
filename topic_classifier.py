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

# Fonction pour détecter les termes du glossaire dans une phrase

import pandas as pd
from file_utils import save_to_csv
from txt_manipulation import decouper_en_phrases


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


def generate_context_windows(phrases, window_size=2, len_phrase_focus=1):
    """
    Génère des fenêtres contextuelles à partir de la liste des phrases.

    Args:
        phrases (list[str]): Liste de phrases à regrouper en fenêtres contextuelles.
        window_size (int): Taille de la fenêtre (nombre de phrases par fenêtre).

    Returns:
        list: Liste des fenêtres contextuelles sous forme de dictionnaires.
    """
    windows = []
    focus_window = []
    for i in range(0, len(phrases), 3):
        # Combine les phrases avant et après la phrase actuelle dans une seule chaîne de caractères
        context_window = " ".join(
            phrases[max(0, i - window_size):min(i + window_size + 1, len(phrases))])
        focus_window = " ".join(
            phrases[max(0, i - len_phrase_focus):min(i + len_phrase_focus + 1, len(phrases))])

        windows.append({
            "id": i,  # Ajout de l'index
            "context": context_window,
            "current_phrase": focus_window
        })
    return windows


# Fonction pour comparer les phrases d'un article avec les sections d'un rapport en utilisant des fenêtres contextuelles


def keywords_for_each_chunk(phrases_article: list[str], termes_glossaire: list[str], definitions_glossaire: list[str], window_size: int = 3) -> list[dict]:
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
    for i in range(0, len(context_windows)):
        glossary_terms = detect_glossary_terms(
            context_windows[i]["current_phrase"], termes_glossaire)
        mentions.append({
            # "article_title":
            # Récupération de la phrase
            "phrase": context_windows[i]["current_phrase"],
            "contexte": context_windows[i]["context"],
            "glossary_terms": glossary_terms,
            # Ajouter les définitions correspondantes aux termes du glossaire détectés
            "definitions": [definitions_glossaire[termes_glossaire.index(term)] for term in glossary_terms if term in termes_glossaire]
        })
    print(f"{len(mentions)} mentions trouvées.")
    return mentions


def glossaire_topics(chemin_glossaire, chemin_cleaned_article, chemin_resultats_csv):
    # Charger le glossaire (termes et définitions)
    glossaire = pd.read_csv(chemin_glossaire)
    termes_glossaire = glossaire['Translated_Term'].tolist()
    definitions_glossaire = glossaire['Translated_Definition'].tolist()

    with open(chemin_cleaned_article, 'r', encoding='utf-8') as file:
        texte_nettoye = file.read()
    # Découper l'article en phrases
    phrases = decouper_en_phrases(texte_nettoye)

    # Comparer l'article avec le rapport
    mentions = keywords_for_each_chunk(
        phrases, termes_glossaire, definitions_glossaire)

    # Sauvegarder les correspondances dans un fichier CSV
    save_to_csv(mentions, chemin_resultats_csv, [
        "phrase", "contexte", "glossary_terms", "definitions"])
