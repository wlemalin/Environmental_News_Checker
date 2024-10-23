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
def configure_embeddings(use_ollama: bool = False) -> None:
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



# Function to analyze a paragraph with Llama 3.2
def analyze_paragraph_with_llm(paragraph, llm_chain):
    """
    Analyse un paragraphe à l'aide du modèle Llama 3.2 et génère une réponse.

    Args:
        paragraph (str): Le paragraphe à analyser.
        llm_chain: Le modèle LLMChain utilisé pour l'analyse.

    Returns:
        str: La réponse générée après l'analyse du paragraphe.
    """
    inputs = {"paragraph": paragraph}
    response = llm_chain.invoke(inputs)
    if isinstance(response, dict) and "text" in response:
        return response["text"].strip()
    return response.strip()



# Function to handle the analysis of paragraphs in parallel
def analyze_paragraphs_parallel(paragraphs, llm_chain):
    """
    Traite l'analyse de plusieurs paragraphes en parallèle en utilisant le modèle Llama 3.2.

    Args:
        paragraphs (list[str]): Liste des paragraphes à analyser.
        llm_chain: Le modèle LLMChain utilisé pour l'analyse.

    Returns:
        list[dict]: Résultats de l'analyse pour chaque paragraphe, avec le texte original et la réponse générée.
    """
    results = []
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a task for each paragraph
        futures = {executor.submit(analyze_paragraph_with_llm, paragraph, llm_chain): paragraph for paragraph in paragraphs if len(paragraph.strip()) > 0}
        # Iterate over results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Analyzing paragraphs"):
            paragraph = futures[future]
            try:
                analysis = future.result()
                results.append({"paragraph": paragraph, "climate_related": analysis})
                # Display the response after each analysis
                print(f"Paragraph:\n{paragraph}\nLLM Response: {analysis}\n")
            except Exception as exc:
                print(f"Error analyzing paragraph: {paragraph} - {exc}")
    return results



def create_prompt_template():
    """
    Create a prompt template for generating a question based on a paragraph and themes.

    Returns:
        PromptTemplate: A template to generate specific questions for verification purposes.
    """
# Template pour générer une question à partir d'un paragraphe et des thèmes
    prompt_template = """
    Vous êtes chargé de formuler une **question précise** pour vérifier les informations mentionnées dans un article de presse en consultant directement les rapports du GIEC (Groupe d'experts intergouvernemental sur l'évolution du climat).

    Cette question sera utilisée dans un système de récupération d'information (RAG) pour extraire les sections pertinentes des rapports du GIEC et comparer les informations des rapports avec celles de l'article de presse.

    **Objectif** : La question doit permettre de vérifier si les informations fournies dans le paragraphe de l'article sont corroborées ou contestées par les preuves scientifiques dans les rapports du GIEC.

    **Instructions** :

    1. Analysez le paragraphe et les thèmes fournis pour identifier les affirmations clés ou les informations à vérifier.
    2. Formulez une **question claire et spécifique** orientée vers la vérification de ces affirmations ou informations à partir des rapports du GIEC.
    3. La question doit être **directement vérifiable** dans les rapports du GIEC via un système RAG.
    4. **IMPORTANT** : Répondez uniquement avec la question, sans ajouter d'explications ou de contexte supplémentaire.

    Paragraphe : {paragraph}

    Thèmes principaux : {themes}

    Générez uniquement la **question** spécifique qui permettrait de vérifier les informations mentionnées dans ce paragraphe en consultant les rapports du GIEC via un système de récupération d'information (RAG).
    """
    return PromptTemplate(template=prompt_template, input_variables=["paragraph", "themes"])



# Fonction pour générer une question avec Llama3.2
def generate_question(paragraph, themes, llm_chain):
    """
    Generate a verification question using Llama3.2 based on a given paragraph and themes.

    Args:
        paragraph (str): The paragraph from which to generate the question.
        themes (list of str): Themes related to the paragraph.
        llm_chain (LLMChain): The language model chain to use for generation.

    Returns:
        str: The generated question.
    """
    inputs = {"paragraph": paragraph, "themes": ', '.join(themes)}
    response = llm_chain.invoke(inputs)  # Utilisation de invoke pour garantir une invocation appropriée
    if isinstance(response, dict) and "text" in response:
        return response["text"].strip()
    return response.strip()

# Fonction pour traiter les questions en parallèle
def generate_questions_parallel(df, llm_chain):
    """
    Generate questions for multiple paragraphs in parallel.

    Args:
        df (DataFrame): A pandas DataFrame containing paragraphs and their corresponding themes.
        llm_chain (LLMChain): The language model chain to use for generation.

    Returns:
        DataFrame: A DataFrame containing the original rows with generated questions.
    """
    results = []
    
    # Utilisation de ThreadPoolExecutor pour le traitement parallèle
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(generate_question, row['paragraph'], row['subjects'].split(', '), llm_chain): row for idx, row in df.iterrows() if row['binary_response'] == 1}
        
        # Parcourir les résultats à mesure qu'ils sont terminés
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating questions"):
            row = futures[future]
            try:
                question = future.result()
                row['question'] = question
                results.append(row)
            except Exception as exc:
                print(f"Error generating question for paragraph: {row['paragraph']} - {exc}")
    
    return pd.DataFrame(results)

