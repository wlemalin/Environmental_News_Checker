"""
Ce script configure des modèles d'embeddings et utilise un modèle LLMChain pour générer des réponses basées sur des sections récupérées d'un rapport. 
Il inclut des fonctions pour configurer les embeddings, générer des réponses à des questions, et comparer des phrases d'articles à des sections d'un rapport via des embeddings.

Fonctionnalités principales :
- Configuration des embeddings Ollama ou HuggingFace
- Génération de réponses avec LLMChain
- Comparaison de phrases d'articles avec des sections de rapport via RAG (Retrieve-then-Generate)
- Utilisation de la similarité cosinus pour le classement des sections pertinentes

"""

from concurrent.futures import ThreadPoolExecutor  # Pour la parallélisation
import numpy as np
import tqdm
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from sentence_transformers import util
from tqdm import tqdm  # Pour la barre de progression


# Fonction pour configurer les modèles d'embeddings
def configure_embeddings(use_ollama: bool = True) -> None:
    """
    Configure le modèle d'embeddings à utiliser (Ollama ou HuggingFace).

    Args:
        use_ollama (bool): Utiliser OllamaEmbedding si True, sinon HuggingFaceEmbedding.
    """
    if use_ollama:
        Settings.embed_model = OllamaEmbedding(model_name="llama3.2:3b-instruct-fp16")
    else:
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# Fonction pour générer des réponses à l'aide de LLMChain
def rag_answer_generation_with_llmchain(question: str, relevant_sections: list[str], llm_chain) -> str:
    """
    Génère une réponse à une question en utilisant LLMChain avec des sections pertinentes du texte.

    Args:
        question (str): La question à laquelle répondre.
        relevant_sections (list[str]): Les sections pertinentes récupérées du texte.
        llm_chain: Le modèle LLMChain pour générer les réponses.

    Returns:
        str: La réponse générée par LLMChain.
    """
    # Combiner les sections pertinentes en un seul contexte
    context = " ".join(relevant_sections)

    # Préparer les entrées pour LLMChain
    inputs = {
        "question": question,
        "consolidated_text": context
    }

    # Générer la réponse en utilisant LLMChain
    response = llm_chain.invoke(inputs)  # Mise à jour de __call__ à invoke

    # Extraire la réponse générée à partir de la réponse
    generated_answer = response['text'] if isinstance(response, dict) and "text" in response else response
    return generated_answer.strip()


# Fonction pour comparer les phrases d'un article avec les sections d'un rapport
def comparer_article_rapport_with_rag(phrases_article: list[str], embeddings_rapport: np.ndarray, sections_rapport: list[str], llm_chain, seuil_similarite: float = 0.5, top_k: int = 3) -> list[dict]:
    """
    Compare les phrases d'un article avec les sections d'un rapport en utilisant les embeddings et RAG.

    Args:
        phrases_article (list[str]): Liste des phrases à comparer.
        embeddings_rapport (np.ndarray): Embeddings des sections du rapport.
        sections_rapport (list[str]): Liste des sections du rapport.
        llm_chain: Le modèle LLMChain pour générer les réponses.
        seuil_similarite (float): Seuil de similarité pour la comparaison (par défaut 0.5).
        top_k (int): Nombre de sections les plus similaires à récupérer (par défaut 3).

    Returns:
        list[dict]: Liste des mentions trouvées, avec les phrases, sections récupérées et réponses générées.
    """
    print("Comparaison des phrases de l'article avec les sections du rapport...")
    mentions = []

    # Encoder les phrases de l'article avec une barre de progression et parallélisation
    embeddings_phrases = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        # Génération parallèle des embeddings
        futures = [executor.submit(Settings.embed_model.get_text_embedding, phrase) for phrase in phrases_article]

        # Utilisation de tqdm pour afficher la progression
        for future in tqdm(futures, desc="Génération des Embeddings", total=len(futures)):
            embedding = future.result()
            # Assurer que l'embedding n'est pas vide
            if embedding is not None and len(embedding) > 0:
                embeddings_phrases.append(embedding)
            else:
                # Ajouter un embedding nul comme solution de secours pour maintenir la forme
                embeddings_phrases.append(np.zeros(3072))

    # Boucler sur chaque phrase de l'article avec une barre de progression
    for i, phrase_embedding in tqdm(enumerate(embeddings_phrases), total=len(embeddings_phrases), desc="Comparaison des Phrases"):
        if len(phrase_embedding) == 0:  # Passer les embeddings vides
            continue

        similarites = util.cos_sim(phrase_embedding, embeddings_rapport)
        # Récupérer les top-k sections les plus proches du rapport
        top_k_indices = np.argsort(-similarites[0])[:top_k]
        top_k_sections = [sections_rapport[j] for j in top_k_indices]

        # Utiliser RAG (Retrieve-then-Generate) avec LLMChain
        question = phrases_article[i]
        generated_answer = rag_answer_generation_with_llmchain(question, top_k_sections, llm_chain)

        # Stocker les résultats
        mentions.append({
            "phrase": question,
            "retrieved_sections": " ".join(top_k_sections),
            "generated_answer": generated_answer
        })

    print(f"{len(mentions)} mentions trouvées.")
    return mentions
