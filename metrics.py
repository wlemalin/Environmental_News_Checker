import pandas as pd
import tqdm
from langchain import LLMChain, PromptTemplate
from langchain.llms import Ollama
from tqdm import tqdm
from llms import creer_prompts_metrics
from file_utils import charger_paragraphes_et_mentions, sauvegarder_resultats_evaluation



# Fonction pour évaluer l'exactitude, le biais et le ton des paragraphes avec trois LLM différents
def evaluer_paragraphe_trois_taches(paragraphes, mentions, llm_exactitude_chain, llm_biais_chain, llm_ton_chain):
    results = []

    for _, row in tqdm(paragraphes.iterrows(), total=len(paragraphes), desc="Évaluation des paragraphes"):
        paragraphe = row['paragraph']

        # Filtrer les mentions associées au paragraphe
        mentions_associees = mentions[mentions['paragraph']
                                      == paragraphe]['retrieved_sections'].tolist()
        mentions_concat = " ".join(mentions_associees)

        # Évaluation de l'exactitude
        response_exactitude = llm_exactitude_chain.invoke(
            {"paragraphe": paragraphe, "mentions": mentions_concat})
        exactitude = response_exactitude['text'].strip() if isinstance(
            response_exactitude, dict) and "text" in response_exactitude else response_exactitude.strip()

        # Évaluation du biais
        response_biais = llm_biais_chain.invoke(
            {"paragraphe": paragraphe, "mentions": mentions_concat})
        biais = response_biais['text'].strip() if isinstance(
            response_biais, dict) and "text" in response_biais else response_biais.strip()

        # Évaluation du ton
        response_ton = llm_ton_chain.invoke(
            {"paragraphe": paragraphe, "mentions": mentions_concat})
        ton = response_ton['text'].strip() if isinstance(
            response_ton, dict) and "text" in response_ton else response_ton.strip()

        # Sauvegarder le résultat
        results.append({
            "paragraph": paragraphe,
            "exactitude": exactitude,
            "biais": biais,
            "ton": ton
        })

    return pd.DataFrame(results)


# Main function to execute the accuracy, bias, and tone evaluation process
def process_evaluation(paragraphes_csv, mentions_csv, resultats_csv):
    # Charger les paragraphes et les mentions du rapport du GIEC
    paragraphes_df, mentions_df = charger_paragraphes_et_mentions(
        paragraphes_csv, mentions_csv)
    # Configurer les modèles LLM (Llama3.2 via Ollama)
    llm_exactitude = Ollama(model="llama3.2:3b-instruct-fp16")
    llm_biais = Ollama(model="llama3.2:3b-instruct-fp16")
    llm_ton = Ollama(model="llama3.2:3b-instruct-fp16")
    # Créer des templates de prompts pour chaque tâche
    prompt_exactitude, prompt_biais, prompt_ton = creer_prompts_metrics()
    # Créer des chaînes LLM pour chaque tâche
    llm_exactitude_chain = LLMChain(prompt=PromptTemplate(
        template=prompt_exactitude, input_variables=["paragraphe", "mentions"]), llm=llm_exactitude)
    llm_biais_chain = LLMChain(prompt=PromptTemplate(
        template=prompt_biais, input_variables=["paragraphe", "mentions"]), llm=llm_biais)
    llm_ton_chain = LLMChain(prompt=PromptTemplate(
        template=prompt_ton, input_variables=["paragraphe", "mentions"]), llm=llm_ton)
    # Évaluer les paragraphes pour les trois tâches
    resultats = evaluer_paragraphe_trois_taches(
        paragraphes_df, mentions_df, llm_exactitude_chain, llm_biais_chain, llm_ton_chain)
    # Sauvegarder les résultats
    sauvegarder_resultats_evaluation(resultats, resultats_csv)
