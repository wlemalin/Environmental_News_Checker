from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from langchain import LLMChain, PromptTemplate
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from embeddings_creation import charger_embeddings_rapport, embed_texts
from file_utils import sauvegarder_resultats_resume
from llms import creer_prompt_resume

def resumer_sections_sur_question(phrase_id, question, retrieved_sections, llm_chain_resume):
    resumes = []  # Liste pour stocker les résumés de chaque section
    sections = []  # Liste pour stocker les sections originales

    for section in retrieved_sections:
        # Générer le résumé de chaque section pertinente
        response_resume = llm_chain_resume.invoke({
            "question": question,
            "retrieved_section": section
        })
        resume = response_resume['text'].strip() if isinstance(
            response_resume, dict) and "text" in response_resume else response_resume.strip()
        resumes.append(resume)  # Ajouter chaque résumé à la liste
        sections.append(section)

    # Retourner les résultats avec ID, question, sections, et la liste des résumés des sections pertinentes
    return {
        "id": phrase_id,               # Ajout de l'ID dans les résultats
        "question": question,           # Ajout de la question dans les résultats
        "sections": sections,           # Liste des sections originales
        "resume_sections": resumes      # Liste des résumés des sections pertinentes
    }
# Nouvelle fonction pour filtrer les sections pertinentes


def filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, titles_rapport, sections_rapport, top_k=5):


    for resultat in df_questions:
        question_embedding = embed_texts(
            [resultat['question']], embed_model)[0]
        # Calculer les similarités
        similarites = util.cos_sim(question_embedding, torch.tensor(
            embeddings_rapport, device='cpu'))
        # Obtenir les indices des sections les plus pertinentes
        top_k_indices = np.argsort(-similarites[0])[:top_k]
        # Filtrer et reconstruire retrieved_sections_concat avec les top_k sections
        top_k_sections = [
            f"{titles_rapport[j]}: {sections_rapport[j]}" for j in top_k_indices
        ]
        resultat['retrieved_sections'] = top_k_sections  # Mise à jour des sections concaténées
    return df_questions

# Fonction principale pour exécuter le processus de résumé des sections


def process_resume(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, top_k=5):

    # Charger les phrases et les mentions du fichier rag_results.csv
    df_questions = pd.read_csv(chemin_csv_questions)

    embed_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2', device='cpu')  # Set the model to CPU
    
    embeddings_rapport, sections_rapport, titles_rapport = charger_embeddings_rapport(chemin_rapport_embeddings)

    # Filtrer les sections pertinentes avant la sauvegarde
    filtered_sections = filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, titles_rapport, sections_rapport, top_k)

    # Configurer un seul modèle LLM (Llama3.2 via Ollama)
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    # Créer un template de prompt pour le résumé des sections du GIEC
    prompt_resume = creer_prompt_resume()
    # Créer une chaîne LLM pour la tâche de résumé
    llm_chain_resume = LLMChain(prompt=prompt_resume, llm=llm)
    # Résumer les sections du GIEC pour chaque question

    resultats = resumer_sections_pertinentes(filtered_sections, llm_chain_resume)
    # Sauvegarder les résultats
    sauvegarder_resultats_resume(resultats, chemin_resultats_csv)


# Fonction pour extraire et résumer les informations pertinentes des sections du GIEC associées à chaque question
def resumer_sections_pertinentes(rag_df, llm_chain_resume):
    results = []
    # Utilisation de ThreadPoolExecutor pour exécuter plusieurs résumés en parallèle
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _, row in rag_df.iterrows():
            phrase_id = row['id']
            question = row['question']
            # Sections associées à la question
            retrieved_sections_concat = row['retrieved_sections']
            # Soumettre la tâche de résumé au LLM
            futures.append(executor.submit(
                resumer_sections_sur_question,
                phrase_id, question, retrieved_sections_concat,
                llm_chain_resume
            ))
        # Récupérer les résultats au fur et à mesure que les tâches sont terminées
        for future in tqdm(as_completed(futures), total=len(futures), desc="Résumé des sections du GIEC"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Erreur lors du résumé d'une phrase : {exc}")
    return pd.DataFrame(results)


