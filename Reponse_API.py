#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 05:36:03 2024

@author: mateodib
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import replicate
from tqdm import tqdm


# Load questions from CSV file


def charger_questions(chemin_csv):
    df = pd.read_csv(chemin_csv)
    return df

# Save RAG results to a CSV file


def sauvegarder_mentions_csv(mentions, chemin_csv):
    df_mentions = pd.DataFrame(mentions)
    df_mentions.to_csv(chemin_csv, index=False, quotechar='"')
    print(f"Mentions saved to file {chemin_csv}")

# Generate answers with Replicate API using relevant sections


def rag_answer_generation_with_replicate(question, relevant_summaries):
    # Create the prompt with the template
    prompt_template = """
    Vous êtes un expert en climatologie. Répondez à la question ci-dessous en vous basant uniquement sur les sections pertinentes du rapport du GIEC.

    **Instructions** :
    1. Utilisez les informations des sections pour formuler une réponse précise et fondée.
    2. Justifiez votre réponse en citant les sections, si nécessaire.
    3. Limitez votre réponse aux informations fournies dans les sections.

    **Question** : {question}
    
    **Sections du rapport** : {consolidated_text}
    
    **Réponse** :
    - **Résumé de la réponse** : (Réponse concise)
    - **Justification basée sur le rapport** : (Citez et expliquez les éléments pertinents)
    """
    prompt_text = prompt_template.format(
        question=question, consolidated_text=relevant_summaries)

    # Call Replicate API to get the answer
    try:
        output = replicate.run("meta/meta-llama-3-70b-instruct",
                               input={"prompt": prompt_text, "max_tokens": 1000})
        # Join the output list to form a single string
        return "".join(output).strip()
    except Exception as e:
        print(f"Error during API call to Replicate: {e}")
        return "Erreur de l'API Replicate"

# Compare questions to the summarized report sections


def comparer_questions_rapport(questions):
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Comparing questions"):
            ID = row['id']
            question = row['question']
            resume_sections = row['resume_sections']
            sections = row['sections']

            futures.append(executor.submit(
                trouver_sections_et_generer_reponse,
                question,
                resume_sections,
                ID
            ))

        # Gather results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Retrieving answers"):
            try:
                question, resume_sections, generated_answer, ID = future.result()
                results.append({
                    "id": ID,
                    "question": question,
                    "sections_resumees": resume_sections,
                    "retrieved_sections": sections,
                    "reponse": generated_answer
                })
            except Exception as exc:
                print(f"Error during RAG: {exc}")

    return results

# Function to find relevant sections and generate a response


def trouver_sections_et_generer_reponse(question, resume_sections, ID):
    generated_answer = rag_answer_generation_with_replicate(
        question, resume_sections)
    return question, resume_sections, generated_answer, ID

# Main function to execute the RAG process


def rag_process_api(chemin_questions_csv, chemin_resultats_csv):
    # Configure the Replicate API key
    os.environ["REPLICATE_API_TOKEN"] = "r8_KVdlDIHTh9T6xEuEJhDkNxvfCXleqe814zH72"
    replicate.api_token = os.getenv("REPLICATE_API_TOKEN")
    questions_df = charger_questions(chemin_questions_csv)

    # Generate answers and save them to CSV
    mentions = comparer_questions_rapport(questions_df)
    sauvegarder_mentions_csv(mentions, chemin_resultats_csv)


