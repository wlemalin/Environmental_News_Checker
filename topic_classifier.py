import nltk
import numpy as np
from llama_index.core import Settings
from sentence_transformers import util


# Fonction pour détecter les termes du glossaire dans une phrase
def detect_glossary_terms(phrase, termes_glossaire):
    words_in_phrase = set(nltk.word_tokenize(phrase.lower()))
    return [term for term in termes_glossaire if term.lower() in words_in_phrase]

# Fonction pour générer des fenêtres contextuelles
def generate_context_windows(phrases, window_size=3):
    windows = []
    for i in range(0, len(phrases), window_size):
        context_window = " ".join(phrases[i:i+window_size])
        windows.append(context_window)
    return windows

# Fonction pour comparer l'article avec le rapport avec contexte et glossaire
def comparer_article_rapport(phrases_article, embeddings_rapport, sections_rapport, termes_glossaire, definitions_glossaire, window_size=3):
    print("Comparaison des phrases de l'article avec les sections du rapport...")
    mentions = []
    # Générer les fenêtres contextuelles
    context_windows = generate_context_windows(phrases_article, window_size)
    # Générer les embeddings pour les fenêtres contextuelles
    print("Génération des embeddings des fenêtres contextuelles...")
    embeddings_windows = [Settings.embed_model.get_text_embedding(
        window) for window in context_windows]
    # Comparer chaque fenêtre contextuelle aux sections du rapport
    for i, window_embedding in enumerate(embeddings_windows):
        similarites = util.cos_sim(window_embedding, embeddings_rapport)
        # Convertir en tableau numpy pour le calcul des statistiques
        similarites_np = similarites.cpu().detach().numpy()  # Conversion explicite en numpy
        # Ajuster dynamiquement le seuil de similarité
        seuil_similarite = max(0.3, np.mean(
            similarites_np) - np.std(similarites_np) / 2)

        for j, similarite in enumerate(similarites_np[0]):
            if similarite > seuil_similarite:
                glossary_terms = detect_glossary_terms(
                    context_windows[i], termes_glossaire)

                mentions.append({
                    "phrase": context_windows[i],
                    "section": sections_rapport[j],
                    "similarite": similarite,
                    "glossary_terms": glossary_terms,
                    # Ajouter les définitions correspondantes
                    "definitions": [definitions_glossaire[termes_glossaire.index(term)] for term in glossary_terms if term in termes_glossaire]
                })

    print(f"{len(mentions)} mentions trouvées.")
    return mentions
