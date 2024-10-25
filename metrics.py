from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tqdm
from langchain import LLMChain, PromptTemplate
from langchain_ollama import OllamaLLM
from tqdm import tqdm

from file_utils import sauvegarder_resultats_evaluation
from llms import creer_prompts
from file_utils import charger_rag_results

# Fonction pour évaluer une phrase spécifique (exactitude, biais et ton)
def evaluer_trois_taches_sur_phrase(phrase_id, question, current_phrase, context, retrieved_sections_concat,
                                    llm_chain_exactitude, llm_chain_biais, llm_chain_ton):
    # Évaluation de l'exactitude
    response_exactitude = llm_chain_exactitude.invoke({
        "current_phrase": current_phrase,
        "context": context,
        "retrieved_sections": retrieved_sections_concat
    })
    exactitude = response_exactitude['text'].strip() if isinstance(
        response_exactitude, dict) and "text" in response_exactitude else response_exactitude.strip()

    # Évaluation du biais
    response_biais = llm_chain_biais.invoke({
        "current_phrase": current_phrase,
        "context": context,
        "retrieved_sections": retrieved_sections_concat
    })
    biais = response_biais['text'].strip() if isinstance(
        response_biais, dict) and "text" in response_biais else response_biais.strip()

    # Évaluation du ton
    response_ton = llm_chain_ton.invoke({
        "current_phrase": current_phrase,
        "context": context,
        "retrieved_sections": retrieved_sections_concat
    })
    ton = response_ton['text'].strip() if isinstance(
        response_ton, dict) and "text" in response_ton else response_ton.strip()

    # Retourner les résultats pour cette phrase avec id et question
    return {
        "id": phrase_id,  # Ajout de l'ID dans les résultats
        "question": question,  # Ajout de la question dans les résultats
        "current_phrase": current_phrase,
        "context": context,
        "retrieved_sections": retrieved_sections_concat,
        "exactitude": exactitude,
        "biais": biais,
        "ton": ton
    }


# Fonction pour évaluer l'exactitude, le biais et le ton des phrases avec un seul modèle LLM et différents prompts
def evaluer_phrase_trois_taches(rag_df, llm_chain_exactitude, llm_chain_biais, llm_chain_ton):
    results = []

    # Utilisation de ThreadPoolExecutor pour exécuter plusieurs évaluations en parallèle
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        for _, row in rag_df.iterrows():
            phrase_id = row['id']
            current_phrase = row['current_phrase']
            # Utilisation du contexte généré comme contexte
            context = row['generated_answer']
            question = row['question']  # Récupérer la question associée
            # Récupérer les sections associées
            retrieved_sections_concat = row['retrieved_sections']

            # Soumettre les trois évaluations (exactitude, biais, ton) à exécuter en parallèle
            futures.append(executor.submit(
                evaluer_trois_taches_sur_phrase,
                phrase_id, question, current_phrase, context, retrieved_sections_concat,
                llm_chain_exactitude, llm_chain_biais, llm_chain_ton
            ))

        # Récupérer les résultats au fur et à mesure que les tâches sont terminées
        for future in tqdm(as_completed(futures), total=len(futures), desc="Évaluation des phrases"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Erreur lors de l'évaluation d'une phrase : {exc}")

    return pd.DataFrame(results)


# Main function to execute the accuracy, bias, and tone evaluation process
def process_evaluation(rag_csv, resultats_csv):
    # Charger les phrases et les mentions du fichier rag_results.csv
    rag_df = charger_rag_results(rag_csv)

    # Configurer un seul modèle LLM (Llama3.2 via Ollama)
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")

    # Créer des templates de prompts pour chaque tâche
    prompt_exactitude, prompt_biais, prompt_ton = creer_prompts()

    # Créer des chaînes LLM pour chaque tâche en utilisant le même modèle
    llm_chain_exactitude = LLMChain(prompt=PromptTemplate(template=prompt_exactitude, input_variables=[
                                    "current_phrase", "context", "retrieved_sections"]), llm=llm)
    llm_chain_biais = LLMChain(prompt=PromptTemplate(template=prompt_biais, input_variables=[
                               "current_phrase", "context", "retrieved_sections"]), llm=llm)
    llm_chain_ton = LLMChain(prompt=PromptTemplate(template=prompt_ton, input_variables=[
                             "current_phrase", "context", "retrieved_sections"]), llm=llm)

    # Évaluer les phrases pour les trois tâches
    resultats = evaluer_phrase_trois_taches(
        rag_df, llm_chain_exactitude, llm_chain_biais, llm_chain_ton)

    # Sauvegarder les résultats
    sauvegarder_resultats_evaluation(resultats, resultats_csv)
