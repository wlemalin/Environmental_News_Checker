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
import pandas as pd
from tqdm import tqdm
import re
import replicate


"""

Functions for extract_relevant_ipcc_references.py et pour extract_relevant_ipcc_references_api.py

"""

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





"""

Functions for extract_relevant_ipcc_references.py

"""




"""

Functions for filtrer_extraits_api.py

"""


# Function to create the prompt and call Replicate's API
def generate_response_with_replicate(current_phrase, context):
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
    prompt_text = prompt_template.format(
        current_phrase=current_phrase,
        context=context
    )

    # Call Replicate API
    try:
        output = replicate.run("meta/meta-llama-3-70b-instruct",
                               input={"prompt": prompt_text, "max_tokens": 500})
        return "".join(output).strip()
    except Exception as e:
        print(f"Error during API call to Replicate: {e}")
        return "Erreur de l'API Replicate"

# Function to analyze paragraphs in parallel using Replicate API
def analyze_paragraphs_parallel_api(splitted_text):
    results = []

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
        # Create a task for each entry in splitted_text (each phrase with its context and index)
        futures = {
            executor.submit(generate_response_with_replicate, entry["current_phrase"], entry["context"]): entry
            for entry in splitted_text
        }

        # Iterate through the results as they are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Analyzing paragraphs"):
            entry = futures[future]
            current_phrase = entry["current_phrase"]
            context = entry["context"]
            index = entry["id"]

            try:
                # Get the result from the API
                analysis = future.result()

                # Store the index, phrase, context, and API response in the result
                results.append({
                    "id": index,
                    "current_phrase": current_phrase,
                    "context": context,
                    "climate_related": analysis
                })

                # Print after each analysis
                print(f"ID: {index}\nPhrase:\n{current_phrase}\nContext:\n{context}\nReplicate API Response: {analysis}\n")

            except Exception as exc:
                print(f"Error analyzing phrase ID {index}: {current_phrase} - {exc}")

    return results







"""

Functions for questions.py

"""


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


"""

Functions for questions_api.py

"""

# Function to generate a question using Replicate API
def generate_question_with_replicate(current_phrase, context):
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
    prompt_text = prompt_template.format(current_phrase=current_phrase, context=context)

    try:
        output = replicate.run("meta/meta-llama-3-70b-instruct", input={"prompt": prompt_text, "max_tokens": 200})
        return "".join(output).strip()
    except Exception as e:
        print(f"Error generating question with Replicate API: {e}")
        return "Erreur de l'API Replicate"

# Function to process questions in parallel using Replicate API
def generate_questions_parallel_api(df):
    results = []

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(generate_question_with_replicate, row['current_phrase'], row['context']): row
            for _, row in df.iterrows()
        }

        # Retrieve results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating questions"):
            row = futures[future]
            try:
                question = future.result()
                row['question'] = question
                results.append(row)
            except Exception as exc:
                print(f"Error generating question for phrase: {row['current_phrase']} - {exc}")

    return pd.DataFrame(results)



"""

Functions for .py

"""



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









