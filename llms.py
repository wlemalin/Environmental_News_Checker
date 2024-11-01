"""
Ce script configure des modèles d'embeddings et utilise un modèle LLMChain pour générer des réponses basées sur des sections récupérées d'un rapport. 
Il inclut des fonctions pour configurer les embeddings, générer des réponses à des questions, et comparer des phrases d'articles à des sections d'un rapport via des embeddings.

Fonctionnalités principales :
- Configuration des embeddings Ollama ou HuggingFace
- Génération de réponses avec LLMChain
- Comparaison de phrases d'articles avec des sections de rapport via RAG (Retrieve-then-Generate)
- Utilisation de la similarité cosinus pour le classement des sections pertinentes

"""

import concurrent.futures  # For parallelization
import re
from concurrent.futures import ThreadPoolExecutor  # Pour la parallélisation

import numpy as np
import pandas as pd
import tqdm
from langchain import LLMChain, PromptTemplate
from langchain_ollama import OllamaLLM 
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from sentence_transformers import util
from tqdm import tqdm


def prompt_selection_phrase_pertinente(model_name="llama3.2:3b-instruct-fp16") :
    llm = OllamaLLM(model=model_name)
    # Define the improved prompt template for LLM climate analysis in French with detailed instructions
    prompt_template = """
    Vous êtes un expert chargé d'identifier tous les sujets abordés dans le texte suivant, qu'ils soient ou non liés à l'environnement, au changement climatique ou au réchauffement climatique.
    
    Phrase : {current_phrase}
    context : {context}
    
    1. Si le texte mentionne de près ou de loin l'environnement, le changement climatique, le réchauffement climatique, ou des organisations, événements ou accords liés à ces sujets (par exemple le GIEC, les conférences COP, les accords de Paris, etc.), répondez '1'. Sinon, répondez '0'.
    2. Listez **tous** les sujets abordés dans le texte, y compris ceux qui ne sont pas liés à l'environnement ou au climat.
    
    Format de réponse attendu :
    - Réponse binaire (0 ou 1) : [Réponse]
    - Liste des sujets abordés : [Sujet 1, Sujet 2, ...]
    
    Exemple de réponse :
    - Réponse binaire (0 ou 1) : 1
    - Liste des sujets abordés : [Incendies, gestion des forêts, réchauffement climatique, économie locale, GIEC]
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["current_phrase", "context"])

    # Directly chain prompt with LLM using the | operator
    llm_chain = prompt | llm  # Using simplified chaining without LLMChain

    # Use pipe to create a chain where prompt output feeds into the LLM
    return llm_chain

# Fonction pour configurer les modèles d'embeddings
def configure_embeddings(use_ollama: bool = False) -> None:
    """
    Configure le modèle d'embeddings à utiliser (Ollama ou HuggingFace).

    Args:
        use_ollama (bool): Utiliser OllamaEmbedding si True, sinon HuggingFaceEmbedding.
    """
    if use_ollama:
        Settings.embed_model = OllamaEmbedding(
            model_name="llama3.2:3b-instruct-fp16")
    else:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")


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

    inputs = {
        "question": question,
        "consolidated_text": context
    }

    response = llm_chain.invoke(inputs)

    generated_answer = response['text'] if isinstance(
        response, dict) and "text" in response else response
    return generated_answer.strip()

# Fonction pour comparer les phrases d'un article avec les sections d'un rapport


def comparer_article_rapport_with_rag(phrases_article: list[str], embeddings_rapport: np.ndarray, sections_rapport: list[str], llm_chain,  top_k: int = 3) -> list[dict]:
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
        futures = [executor.submit(
            Settings.embed_model.get_text_embedding, phrase) for phrase in phrases_article]

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
        generated_answer = rag_answer_generation_with_llmchain(
            question, top_k_sections, llm_chain)

        # Stocker les résultats
        mentions.append({
            "phrase": question,
            "retrieved_sections": " ".join(top_k_sections),
            "generated_answer": generated_answer
        })

    print(f"{len(mentions)} mentions trouvées.")
    return mentions


# Fonction pour analyser un paragraphe avec Llama 3.2
def analyze_paragraph_with_llm(current_phrase, context, llm_chain):
    # Créer un seul dictionnaire avec les deux clés
    inputs = {"current_phrase": current_phrase, "context": context}

    # Appeler le LLM avec les inputs
    response = llm_chain.invoke(inputs)

    if isinstance(response, dict) and "text" in response:
        return response["text"].strip()
    return response.strip()


# Fonction pour gérer l'analyse des paragraphes en parallèle
def analyze_paragraphs_parallel(splitted_text, llm_chain):
    results = []

    # Utilisation de ThreadPoolExecutor pour le traitement parallèle
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
        # Créez une tâche pour chaque entrée de splitted_text (chaque phrase avec son contexte et son index)
        futures = {executor.submit(
            analyze_paragraph_with_llm, entry["current_phrase"], entry["context"], llm_chain): entry for entry in splitted_text}

        # Parcourir les résultats à mesure qu'ils sont terminés
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Analyzing paragraphs"):
            entry = futures[future]
            current_phrase = entry["current_phrase"]
            context = entry["context"]
            index = entry["id"]  # Récupération de l'index

            try:
                # Obtenir le résultat de l'analyse
                analysis = future.result()

                # Enregistrer l'index, la phrase, le contexte et la réponse du LLM dans le résultat
                results.append({
                    "id": index,  # Ajout de l'index dans les résultats
                    "current_phrase": current_phrase,
                    "context": context,
                    "climate_related": analysis
                })

                # Affichage après chaque analyse
                print(
                    f"ID: {index}\nPhrase:\n{current_phrase}\nContext:\n{context}\nLLM Response: {analysis}\n")

            except Exception as exc:
                print(
                    f"Error analyzing phrase ID {index}: {current_phrase} - {exc}")

    return results

# Fonction améliorée pour parser les réponses du LLM


def parse_llm_response(response):
    # Séparer le texte en lignes pour analyser chaque ligne individuellement
    lines = response.split("\n")

    # Parcourir chaque ligne et supprimer les caractères non alphabétiques au début de chaque ligne
    clean_lines = [re.sub(r'^[^a-zA-Z]+', '', line).strip() for line in lines]

    # Identifier la première ligne contenant la réponse binaire
    response_line = clean_lines[0]

    # Compter le nombre de 0 et de 1 dans cette première ligne
    if response_line:
        count_0 = response_line.count("0")
        count_1 = response_line.count("1")

        # Renvoyer la réponse binaire par vote majoritaire
        if count_1 > count_0:
            binary_response = "1"
        else:
            binary_response = "0"
    else:
        binary_response = None  # Si aucune ligne ne contient de 0 ou de 1, renvoyer None

    # Extraire la section de la liste des sujets abordés
    subjects_section_match = re.search(
        r"Liste des sujets abordés\s?:?\s*(.*)", response, re.DOTALL)

    if subjects_section_match:
        # On récupère toute la section des sujets abordés
        subjects_section = subjects_section_match.group(1).strip()

        # Séparer par lignes pour chaque sujet (utiliser les sauts de ligne)
        subjects_lines = subjects_section.split("\n")

        # Nettoyer chaque ligne pour enlever les tirets, espaces en trop, etc.
        subjects_list = []
        for line in subjects_lines:
            # Enlever les tirets au début et nettoyer les espaces
            clean_subject = re.sub(r'^[^a-zA-Z]+', '', line).strip()
            if clean_subject:  # S'assurer que la ligne n'est pas vide
                subjects_list.append(clean_subject)
    else:
        subjects_list = []

    return binary_response, subjects_list

# Fonction pour parser toutes les réponses dans un DataFrame existant


def parsed_responses(df):
    parsed_data = []
    for _, row in df.iterrows():
        ID = row['id']
        current_phrase = row['current_phrase']
        context = row['context']
        response = row['climate_related']
        binary_response, subjects_list = parse_llm_response(response)
        parsed_data.append({
            "id": ID,
            "current_phrase": current_phrase,
            "context": context,
            "binary_response": binary_response,
            "subjects": subjects_list
        })

    return pd.DataFrame(parsed_data)


def create_questions_llm(model_name="llama3.2:3b-instruct-fp16"):
    """
    Initialise le modèle LLM et crée une LLMChain.
    """
    llm = OllamaLLM(model=model_name)
    prompt_template = """
    Vous êtes chargé de formuler une **question précise** pour vérifier les informations mentionnées dans un extrait spécifique d'un article de presse en consultant directement les rapports du GIEC (Groupe d'experts intergouvernemental sur l'évolution du climat).

    Cette question sera utilisée dans un système de récupération d'information (RAG) pour extraire les sections pertinentes des rapports du GIEC et comparer les informations des rapports avec celles de l'article de presse.

    **Objectif** : La question doit permettre de vérifier si les informations fournies dans la phrase de l'article sont corroborées ou contestées par les preuves scientifiques dans les rapports du GIEC. La question doit donc englober tous les sujets abordés par l'extrait de l'article de presse.

    **Instructions** :

    1. Analysez l'extrait et son contexte pour identifier les affirmations clées et/ou les informations à vérifier.
    2. Formulez une **question claire et spécifique** orientée vers la vérification de ces affirmations ou informations à partir des rapports du GIEC. La question doit permettre de vérifier toutes les informations de l'extraits. La question peut être un ensemble de questions comme : "Quel est l'impact des activités humaines sur le taux de CO2 dans l'atomsphère ? Comment la concentration du CO2 dans l'atmosphère impact l'argiculture?"
    3. La question doit être **directement vérifiable** dans les rapports du GIEC via un système RAG.
    4. **IMPORTANT** : Répondez uniquement avec la question, sans ajouter d'explications ou de contexte supplémentaire.

    Extrait de l'article de presse : {current_phrase}

    Contexte : {context}

    Générez uniquement la **question** spécifique qui permettrait de vérifier les informations mentionnées dans cette phrase en consultant les rapports du GIEC via un système de récupération d'information (RAG).
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "current_phrase", "context"])

    # Directly chain prompt with LLM using the | operator
    llm_chain = prompt | llm  # Using simplified chaining without LLMChain

    # Use pipe to create a chain where prompt output feeds into the LLM
    return llm_chain


# Fonction pour générer une question avec Llama3.2
def generate_question(current_phrase, context, llm_chain):
    inputs = {"current_phrase": current_phrase, "context": context}
    # Utilisation de invoke pour garantir une invocation appropriée
    response = llm_chain.invoke(inputs)
    if isinstance(response, dict) and "text" in response:
        return response["text"].strip()
    return response.strip()


# Fonction pour traiter les questions en parallèle
def generate_questions_parallel(df, llm_chain):
    results = []

    # Utilisation de ThreadPoolExecutor pour le traitement parallèle
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(generate_question, row['current_phrase'], row['context'],
                                   llm_chain): row for _, row in df.iterrows() if row['binary_response'] == '1'}

        # Parcourir les résultats à mesure qu'ils sont terminés
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating questions"):
            row = futures[future]
            try:
                question = future.result()
                row['question'] = question
                results.append(row)
            except Exception as exc:
                print(
                    f"Error generating question for phrase: {row['current_phrase']} - {exc}")

    return pd.DataFrame(results)


def creer_llm_resume(model_name="llama3.2:3b-instruct-fp16"):
    """
    Creates and configures the LLM chain for summarization.
    """
    llm = OllamaLLM(model=model_name)
    prompt_template_resume = """
    **Tâche** : Fournir un résumé structuré des faits contenus dans la section du rapport du GIEC, en les organisant par pertinence pour répondre à la question posée. La réponse doit être sous forme de liste numérotée.

    **Instructions** :
    - **Objectif** : Lister tous les faits pertinents, y compris les éléments indirects ou contextuels pouvant enrichir la réponse.
    - **Éléments à inclure** : 
        1. Faits scientifiques directement liés à la question.
        2. Faits indirects apportant un contexte utile.
        3. Tendances, implications, ou statistiques pertinentes.
        4. Autres informations utiles pour comprendre le sujet.
    - **Restrictions** : Ne pas inclure d'opinions ou interprétations, uniquement les faits.
    - **Format** : Utiliser une liste numérotée, chaque point limité à une ou deux phrases. Commencer par les faits les plus directement liés et finir par les éléments contextuels.

    ### Question :
    "{question}"

    ### Section du rapport :
    {retrieved_sections}

    **Exemple** :
        1. Le niveau global de la mer a augmenté de 0,19 m entre 1901 et 2010.
        2. Les températures mondiales ont augmenté de 1,09°C entre 1850-1900 et 2011-2020.
        3. Les concentrations de CO2 ont atteint 410 ppm en 2019.


    **Remarque** : Respecter strictement ces consignes et ne présenter que les faits sous forme de liste numérotée.
    """
    prompt = PromptTemplate(template=prompt_template_resume, input_variables=[
                            "question", "retrieved_sections"])

    # Directly chain prompt with LLM using the | operator
    llm_chain = prompt | llm  # Using simplified chaining without LLMChain

    # Use pipe to create a chain where prompt output feeds into the LLM
    return llm_chain
