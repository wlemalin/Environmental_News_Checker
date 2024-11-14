#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 04:12:42 2024

@author: mateodib
"""


import os
import replicate
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain import PromptTemplate


def creer_prompts():
    prompts = {
        "accuracy": PromptTemplate(
            template="""
            Vous êtes chargé de comparer un extrait d'un article de presse aux informations officielles du rapport du GIEC.
            Évaluez l'exactitude de cet extrait en fonction des sections du rapport fournies. Utilisez l'échelle suivante :

            - **Score 0** : Totalement inexact, contredit directement les informations du rapport.
            - **Score 1** : Très inexact, contient plusieurs erreurs ou omissions importantes par rapport au rapport.
            - **Score 2** : Partiellement inexact, quelques erreurs importantes, mais certaines informations sont correctes.
            - **Score 3** : Modérément exact, contient des informations correctes mais avec quelques imprécisions.
            - **Score 4** : Largement exact, avec seulement de légères imprécisions.
            - **Score 5** : Parfaitement exact, correspond pleinement aux informations du rapport.

            **Extrait de l'article** :
            "{current_phrase}"

            **Sections du rapport du GIEC** :
            "{sections_resumees}"
            
            Répondez au format suivant en JSON:
            ```json
            {{
                "score": <score_entre_0_et_5>,
                "justifications": "<votre_justification_en_une_ou_plusieurs_phrases>"
            }}
            ```
            """,
            input_variables=["current_phrase", "sections_resumees"]
        ),
        "bias": PromptTemplate(
            template="""
            Vous êtes chargé d'analyser un extrait d'un article de presse pour détecter tout biais potentiel en fonction des sections du rapport du GIEC. Utilisez l'échelle suivante :

            - **Score 0** : Fortement biaisé, présente une vision totalement exagérée ou minimisée.
            - **Score 1** : Biaisé, avec une inclinaison évidente, soit en exagérant soit en minimisant les faits.
            - **Score 2** : Modérément biaisé, certains aspects exagérés ou minimisés mais dans l'ensemble équilibré.
            - **Score 3** : Légèrement biaisé, de petites nuances de biais mais globalement équilibré.
            - **Score 4** : Largement neutre, avec très peu de biais.
            - **Score 5** : Totalement neutre, sans aucun biais détectable.

            **Extrait de l'article** :
            "{current_phrase}"

            **Sections du rapport du GIEC** :
            "{sections_resumees}"

            Répondez au format suivant en JSON:
            ```json
            {{
                "score": <score_entre_0_et_5>,
                "justifications": "<votre_justification_en_une_ou_plusieurs_phrases>"
            }}
            ```
            """,
            input_variables=["current_phrase", "sections_resumees"]
        ),
        "tone": PromptTemplate(
            template="""
            Vous êtes chargé d'analyser le ton d'un extrait d'un article de presse en le comparant aux informations du rapport du GIEC. Utilisez l'échelle suivante :

            - **Score 0** : Ton fortement alarmiste ou minimisant, très éloigné du ton neutre.
            - **Score 1** : Ton exagérément alarmiste ou minimisant.
            - **Score 2** : Ton quelque peu alarmiste ou minimisant.
            - **Score 3** : Ton modérément factuel avec une légère tendance à l'alarmisme ou à la minimisation.
            - **Score 4** : Ton largement factuel, presque totalement neutre.
            - **Score 5** : Ton complètement neutre et factuel, sans tendance perceptible.

            **Extrait de l'article** :
            "{current_phrase}"

            **Sections du rapport du GIEC** :
            "{sections_resumees}"

            Répondez au format suivant en JSON:
            ```json
            {{
                "score": <score_entre_0_et_5>,
                "justifications": "<votre_justification_en_une_ou_plusieurs_phrases>"
            }}
            ```
            """,
            input_variables=["current_phrase", "sections_resumees"]
        ),
        "clarity": PromptTemplate(
            template="""
            Vous êtes chargé d'évaluer la clarté et la lisibilité d'un extrait d'un article de presse en fonction de sa simplicité et de son accessibilité. Utilisez l'échelle suivante :

            - **Score 0** : Très confus, difficile à lire et à comprendre.
            - **Score 1** : Peu clair, nécessite beaucoup d'efforts pour comprendre.
            - **Score 2** : Assez clair, mais certaines phrases ou idées sont difficiles à suivre.
            - **Score 3** : Modérément clair, quelques passages pourraient être simplifiés.
            - **Score 4** : Largement clair, facile à lire avec une structure compréhensible.
            - **Score 5** : Parfaitement clair, très facile à lire et accessible à tous les lecteurs.

            **Extrait de l'article** :
            "{current_phrase}"

            Répondez au format suivant en JSON:
            ```json
            {{
                "score": <score_entre_0_et_5>,
                "justifications": "<votre_justification_en_une_ou_plusieurs_phrases>"
            }}
            ```
            """,
            input_variables=["current_phrase"]
        ),
        "completeness": PromptTemplate(
            template="""
            Vous êtes chargé d'évaluer la complétude de l'information contenue dans un extrait d'un article de presse par rapport aux sections du rapport du GIEC. Utilisez l'échelle suivante :

            - **Score 0** : Très incomplet, de nombreuses informations importantes sont manquantes.
            - **Score 1** : Incomplet, plusieurs points essentiels ne sont pas couverts.
            - **Score 2** : Partiellement complet, des informations importantes sont manquantes mais certains éléments sont présents.
            - **Score 3** : Modérément complet, couvre l'essentiel mais manque de détails.
            - **Score 4** : Largement complet, contient presque toutes les informations nécessaires.
            - **Score 5** : Complètement complet, toutes les informations importantes sont présentes.

            **Extrait de l'article** :
            "{current_phrase}"

            **Sections du rapport du GIEC** :
            "{sections_resumees}"

            Répondez au format suivant en JSON:
            ```json
            {{
                "score": <score_entre_0_et_5>,
                "justifications": "<votre_justification_en_une_ou_plusieurs_phrases>"
            }}
            ```
            """,
            input_variables=["current_phrase", "sections_resumees"]
        ),
        "objectivity": PromptTemplate(
            template="""
            Vous êtes chargé d'évaluer l'objectivité d'un extrait d'un article de presse en vérifiant s'il est libre de langage subjectif ou d'opinions. Utilisez l'échelle suivante :

            - **Score 0** : Très subjectif, plein d'opinions ou de langages émotifs.
            - **Score 1** : Subjectif, contient des opinions ou un langage non neutre.
            - **Score 2** : Modérément subjectif, quelques opinions ou expressions biaisées.
            - **Score 3** : Légèrement subjectif, quelques nuances de subjectivité mais largement objectif.
            - **Score 4** : Largement objectif, avec très peu de subjectivité.
            - **Score 5** : Totalement objectif, sans aucune opinion ou langage subjectif.

            **Extrait de l'article** :
            "{current_phrase}"

            **Sections du rapport du GIEC** :
            "{sections_resumees}"

            Répondez au format suivant en JSON:
            ```json
            {{
                "score": <score_entre_0_et_5>,
                "justifications": "<votre_justification_en_une_ou_plusieurs_phrases>"
            }}
            ```
            """,
            input_variables=["current_phrase", "sections_resumees"]
        ),
        "alignment": PromptTemplate(
            template="""
            Vous êtes chargé d'évaluer si cet extrait d'un article de presse reflète bien les priorités et l'importance des points soulignés dans les sections du rapport du GIEC. Utilisez l'échelle suivante :

            - **Score 0** : Complètement désaligné avec les priorités du rapport.
            - **Score 1** : Largement désaligné, manque les points principaux.
            - **Score 2** : Partiellement aligné, couvre quelques points mais ignore des éléments essentiels.
            - **Score 3** : Modérément aligné, couvre l'essentiel mais manque de priorisation.
            - **Score 4** : Largement aligné, avec une bonne couverture des priorités.
            - **Score 5** : Parfaitement aligné avec les priorités et l'importance du rapport.

            **Extrait de l'article** :
            "{current_phrase}"

            **Sections du rapport du GIEC** :
            "{sections_resumees}"

            Répondez au format suivant en JSON:
            ```json
            {{
                "score": <score_entre_0_et_5>,
                "justifications": "<votre_justification_en_une_ou_plusieurs_phrases>"
            }}
            ```
            """,
            input_variables=["current_phrase", "sections_resumees"]
        )
    }
    return prompts

# Helper function to call Replicate API for a specific metric
def appeler_replicate(prompt_text):
    input_payload = {
        "prompt": prompt_text,
        "max_tokens": 1000
    }
    try:
        output = replicate.run("meta/meta-llama-3-70b-instruct", input=input_payload)
        return "".join(output)  # Join the response segments into a single text
    except Exception as e:
        print(f"Erreur lors de l'appel à Replicate : {e}")
        return "Erreur de l'API Replicate"

# Evaluate a specific phrase for all seven metrics using dedicated models
def evaluer_phrase_toutes_metrices(phrase_id, question, current_phrase, sections_resumees, prompts):
    evaluations = {}
    
    # Iterate over each metric and apply its specific LLM prompt
    for metric, prompt_template in prompts.items():
        prompt_text = prompt_template.format(current_phrase=current_phrase, sections_resumees=sections_resumees)
        evaluations[metric] = appeler_replicate(prompt_text)

    # Return results for this phrase with id and question
    return {
        "id": phrase_id,
        "question": question,
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees,
        **evaluations
    }

# Function to evaluate phrases in parallel for all metrics
def evaluer_phrase_parallele(rag_df, prompts):
    results = []
    
    # Use ThreadPoolExecutor to execute multiple evaluations in parallel
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = []
        
        for _, row in rag_df.iterrows():
            phrase_id = row['id']
            question = row['question']
            current_phrase = row['current_phrase']
            sections_resumees = row['sections_resumees']
            
            # Submit the evaluation for all metrics
            futures.append(executor.submit(
                evaluer_phrase_toutes_metrices,
                phrase_id, question, current_phrase, sections_resumees,
                prompts
            ))
        
        # Retrieve results as tasks complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Évaluation des phrases pour toutes les métriques"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Erreur lors de l'évaluation d'une phrase : {exc}")
    
    return pd.DataFrame(results)

# Main function to process evaluation
def process_evaluation_api(chemin_questions_csv, rag_csv, resultats_csv):
    # Load rag_results.csv
    rag_df = pd.read_csv(rag_csv)
    
    # Load final_climate_analysis_with_questions.csv with only 'id' and 'current_phrase' columns
    questions_df = pd.read_csv(chemin_questions_csv, usecols=['id', 'current_phrase'])
    
    # Merge rag_df with questions_df on 'id' to add the 'current_phrase' column
    rag_df = rag_df.merge(questions_df, on='id', how='left')
    
    # Create prompt templates for each metric
    prompts = creer_prompts()
    
    # Set up the Replicate API key
    os.environ["REPLICATE_API_TOKEN"] = "r8_KVdlDIHTh9T6xEuEJhDkNxvfCXleqe814zH72"
    replicate.api_token = os.getenv("REPLICATE_API_TOKEN")

    # Evaluate phrases for all metrics
    resultats = evaluer_phrase_parallele(rag_df, prompts)
    
    # Save results
    resultats.to_csv(resultats_csv, index=False)
    print(f"Résultats d'évaluation sauvegardés dans {resultats_csv}")
