#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 05:03:44 2024

@author: mateodib
"""

import os
import replicate
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain import PromptTemplate

# Load the phrases and extracted sections from the file rag_results.csv
def charger_rag_results(chemin_rag_csv):
    return pd.read_csv(chemin_rag_csv)

# Save evaluation results to a CSV file
def sauvegarder_resultats_evaluation(resultats, chemin_resultats_csv):
    resultats.to_csv(chemin_resultats_csv, index=False)
    print(f"Résultats d'évaluation sauvegardés dans {chemin_resultats_csv}")

# Prompts for each task (accuracy, bias, tone)
def creer_prompts():
    prompt_template_exactitude = PromptTemplate(
        template="""
        Vous êtes chargé de comparer un extrait d'un article de presse aux informations officielles du rapport du GIEC. 
        Évaluez l'exactitude de cet extrait en fonction des sections du rapport fournies.

        **Extrait de l'article** :
        {current_phrase}

        **Sections du rapport du GIEC** :
        {sections_resumees}
        
        **Format de la réponse** :
        - Score entre 0 et 5
        - Justifications
        """,
        input_variables=["current_phrase", "sections_resumees"]
    )
    
    prompt_template_biais = PromptTemplate(
        template="""
        Vous êtes chargé d'analyser un extrait d'un article de presse pour détecter tout biais potentiel.
        
        **Extrait de l'article** :
        {current_phrase}

        **Sections du rapport du GIEC** :
        {sections_resumees}
        
        **Format de la réponse** :
        - Type de biais (Exagéré, Minimisé, Neutre)
        - Justifications
        """,
        input_variables=["current_phrase", "sections_resumees"]
    )
    
    prompt_template_ton = PromptTemplate(
        template="""
        Vous êtes chargé d'analyser le ton d'un extrait d'un article de presse en le comparant aux informations du rapport du GIEC.
        
        **Extrait de l'article** :
        {current_phrase}

        **Sections du rapport du GIEC** :
        {sections_resumees}
        
        **Format de la réponse** :
        - Ton (Alarmiste, Minimisant, Neutre, Factuel)
        - Justifications
        """,
        input_variables=["current_phrase", "sections_resumees"]
    )
    
    return prompt_template_exactitude, prompt_template_biais, prompt_template_ton

# Helper function to call Replicate API with specific prompt
def appeler_replicate(prompt_text):
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

# Evaluate a specific phrase (accuracy, bias, and tone) with three different models
def evaluer_trois_taches_sur_phrase(phrase_id, question, current_phrase, sections_resumees,
                                    prompt_exactitude, prompt_biais, prompt_ton):
    # Generate prompts for each task
    prompt_text_exactitude = prompt_exactitude.format(current_phrase=current_phrase, sections_resumees=sections_resumees)
    prompt_text_biais = prompt_biais.format(current_phrase=current_phrase, sections_resumees=sections_resumees)
    prompt_text_ton = prompt_ton.format(current_phrase=current_phrase, sections_resumees=sections_resumees)

    try:
        # Call Replicate API for each task
        exactitude = appeler_replicate(prompt_text_exactitude)
        biais = appeler_replicate(prompt_text_biais)
        ton = appeler_replicate(prompt_text_ton)
    
    except Exception as e:
        print(f"Erreur lors de l'appel à Replicate : {e}")
        return {
            "id": phrase_id,
            "question": question,
            "current_phrase": current_phrase,
            "sections_resumees": sections_resumees,
            "exactitude": "Erreur",
            "biais": "Erreur",
            "ton": "Erreur"
        }

    # Return results for this phrase with id and question
    return {
        "id": phrase_id,
        "question": question,
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees,
        "exactitude": exactitude,
        "biais": biais,
        "ton": ton
    }

# Function to evaluate accuracy, bias, and tone for each phrase with a single LLM model and different prompts
def evaluer_phrase_parallele(rag_df, prompt_exactitude, prompt_biais, prompt_ton):
    results = []
    
    # Use ThreadPoolExecutor to execute multiple evaluations in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        
        for _, row in rag_df.iterrows():
            phrase_id = row['id']
            question = row['question']
            current_phrase = row['current_phrase']
            sections_resumees = row['sections_resumees']
            
            # Submit three evaluations (accuracy, bias, tone) to execute in parallel
            futures.append(executor.submit(
                evaluer_trois_taches_sur_phrase,
                phrase_id, question, current_phrase, sections_resumees,
                prompt_exactitude, prompt_biais, prompt_ton
            ))
        
        # Retrieve results as tasks complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Évaluation des phrases"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Erreur lors de l'évaluation d'une phrase : {exc}")
    
    return pd.DataFrame(results)

# Main function to process evaluation
def process_evaluation_api(chemin_questions_csv, rag_csv, resultats_csv):
    # Load rag_results.csv
    rag_df = charger_rag_results(rag_csv)
    
    # Load final_climate_analysis_with_questions.csv with only 'id' and 'current_phrase' columns
    questions_df = pd.read_csv(chemin_questions_csv, usecols=['id', 'current_phrase'])
    
    # Merge rag_df with questions_df on 'id' to add the 'current_phrase' column
    rag_df = rag_df.merge(questions_df, on='id', how='left')
    
    # Create prompt templates for each task
    prompt_exactitude, prompt_biais, prompt_ton = creer_prompts()
    
# Set up the Replicate API key
    os.environ["REPLICATE_API_TOKEN"] = "r8_KVdlDIHTh9T6xEuEJhDkNxvfCXleqe814zH72"
    replicate.api_token = os.getenv("REPLICATE_API_TOKEN")

    # Evaluate phrases for accuracy, bias, and tone
    resultats = evaluer_phrase_parallele(rag_df, prompt_exactitude, prompt_biais, prompt_ton)
    
    # Save results
    sauvegarder_resultats_evaluation(resultats, resultats_csv)

