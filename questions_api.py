#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:49:28 2024

@author: mateodib
"""

import os
import pandas as pd
import replicate
import concurrent.futures
from tqdm import tqdm

# Configure the Replicate API key
os.environ["REPLICATE_API_TOKEN"] = "r8_KVdlDIHTh9T6xEuEJhDkNxvfCXleqe814zH72"

def question_generation_process_api(file_path, output_path_questions):
    """
    Generates questions from sentences identified as environmentally related and saves them to a CSV file.
    """
    df = pd.read_csv(file_path)

    # Convert the 'binary_response' column to a string format if needed
    df['binary_response'] = df['binary_response'].astype(str)

    # Filter only sentences identified as environment-related (binary response '1')
    df_environment = df[df['binary_response'] == '1']

    # Generate questions for environment-related sentences using Replicate API
    questions_df = generate_questions_parallel(df_environment)

    # Save the results to a new CSV file
    questions_df.to_csv(output_path_questions, index=False)
    print(f"Questions generated and saved to {output_path_questions}")


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
def generate_questions_parallel(df):
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

