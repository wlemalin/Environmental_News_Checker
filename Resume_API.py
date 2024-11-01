#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 05:12:30 2024

@author: mateodib
"""

import os
import replicate
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
import tqdm
from embeddings_creation import generer_embeddings_rapport, embed_texts

# Set up Replicate API key
os.environ["REPLICATE_API_TOKEN"] = "r8_KVdlDIHTh9T6xEuEJhDkNxvfCXleqe814zH72"
replicate.api_token = os.getenv("REPLICATE_API_TOKEN")

# Save summary results
def sauvegarder_resultats_resume(resultats, chemin_resultats_csv):
    resultats.to_csv(chemin_resultats_csv, index=False, quotechar='"')
    print(f"Résumés sauvegardés dans le fichier {chemin_resultats_csv}")

# Load data and model for questions and report sections
def charger_donnees_et_modele(chemin_csv_questions, chemin_rapport_embeddings, top_k=5):
    df_questions = pd.read_csv(chemin_csv_questions)
    df_questions = pd.read_csv(chemin_csv_questions).iloc[[0]]
    print(f"Questions loaded. Total: {len(df_questions)}")
    
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    embeddings_rapport, sections_rapport, titles_rapport = generer_embeddings_rapport(chemin_rapport_embeddings, embed_model)
    print("Report embeddings and sections loaded.")
    
    if not (len(embeddings_rapport) == len(sections_rapport) == len(titles_rapport)):
        print("Erreur : Le nombre d'embeddings, de sections, et de titres ne correspond pas.")
    else:
        print(f"Données chargées avec succès : {len(embeddings_rapport)} embeddings, {len(sections_rapport)} sections, et {len(titles_rapport)} titres.")
    
    return df_questions, embeddings_rapport, sections_rapport, titles_rapport, embed_model

# Filter relevant sections based on similarity
def filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k=5):
    retrieved_sections_list = []
    
    for idx, row in df_questions.iterrows():
        question_embedding = embed_texts([row['question']], embed_model)[0]
        similarites = util.cos_sim(question_embedding, torch.tensor(embeddings_rapport, device='cpu'))
        top_k_indices = np.argsort(-similarites[0].cpu()).tolist()[:top_k]
        sections = [sections_rapport[i] for i in top_k_indices if sections_rapport[i].strip()]
        retrieved_sections_list.append(sections)
    
    df_questions['retrieved_sections'] = retrieved_sections_list
    return df_questions

# Create the summary prompt template
def creer_prompt_resume(question, retrieved_sections):
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
    return prompt_template_resume.format(question=question, retrieved_sections=retrieved_sections)

# Function to call Replicate API for generating summaries
def appeler_replicate_summarization(prompt_text):
    input_payload = {
        "prompt": prompt_text,
        "max_tokens": 1000
    }
    try:
        output = replicate.run("meta/meta-llama-3.1-405b-instruct", input=input_payload)
        return "".join(output)  # Join the response segments into a single text
    except Exception as e:
        print(f"Erreur lors de l'appel à Replicate : {e}")
        return "Erreur de l'API Replicate"

# Generate summary for a specific section and question using Replicate API
def generer_resume(phrase_id, question, section):
    prompt_text = creer_prompt_resume(question, section)
    response_resume = appeler_replicate_summarization(prompt_text)
    return {
        "id": phrase_id,
        "question": question,
        "section": section,
        "resume_section": response_resume
    }

# Parallel summary generation for questions and their sections
def generer_resume_parallel(df_questions):
    resultats = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for _, row in df_questions.iterrows():
            question_id = row['id']
            question_text = row['question']
            retrieved_sections = row['retrieved_sections']
            
            for section in retrieved_sections:
                futures.append(executor.submit(
                    generer_resume, 
                    question_id, 
                    question_text, 
                    section
                ))
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Résumés des sections"):
            try:
                result = future.result()
                resultats.append(result)
            except Exception as exc:
                print(f"Erreur lors du résumé d'une question : {exc}")
    
    return pd.DataFrame(resultats)

# Main function to process and save summaries
def process_resume_api(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, top_k=3):
    df_questions, embeddings_rapport, sections_rapport, _, embed_model = charger_donnees_et_modele(
        chemin_csv_questions, chemin_rapport_embeddings, top_k)
    
    df_questions = filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k)
    
    resultats = generer_resume_parallel(df_questions)
    
    resultats_grouped = resultats.groupby('id').agg({
        'question': 'first',
        'section': lambda x: ' '.join(x),
        'resume_section': lambda x: ' '.join(x)
    }).reset_index()
    
    resultats_grouped.rename(columns={'section': 'sections', 'resume_section': 'resume_sections'}, inplace=True)
    
    sauvegarder_resultats_resume(resultats_grouped, chemin_resultats_csv)
    print(f"Résumés sauvegardés dans le fichier {chemin_resultats_csv}")

# File paths for processing
chemin_csv_questions = "/Users/mateodib/Desktop/Environmental_News_Checker-main/final_climate_analysis_with_questions.csv"
chemin_resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-main/resume_sections_results.csv"
chemin_rapport_embeddings = "/Users/mateodib/Desktop/Environmental_News_Checker-main/rapport_indexed.json"

# Run the process
process_resume_api(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, 5) # Top-K = 5
