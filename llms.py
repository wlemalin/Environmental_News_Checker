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
from langchain import PromptTemplate
from langchain_ollama import OllamaLLM
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from sentence_transformers import util
from tqdm import tqdm




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


