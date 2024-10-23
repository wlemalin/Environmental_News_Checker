#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour le traitement d'un article de presse et d'un rapport du GIEC.
Ce script effectue plusieurs tâches liées au traitement de texte et à l'intelligence artificielle, réparties en quatre étapes :

1. Nettoyage et prétraitement de l'article de presse : Suppression des éléments non pertinents et préparation du texte pour les étapes suivantes.
2. Extraction, nettoyage et indexation des sections du rapport PDF : Transformation du rapport en texte exploitable, puis découpage en sections indexées.
3. Identification des mentions directes et indirectes au GIEC : Utilisation d'un modèle d'embeddings pour comparer les phrases de l'article avec les sections du rapport, et détection des termes du glossaire.
4. Vérification des faits avec un modèle LLM (Llama) : Génération de réponses basées sur les sections extraites du rapport en utilisant un modèle de type RAG (Retrieve-and-Generate).
"""

import pandas as pd
import nltk
from langchain import LLMChain, PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain_ollama import OllamaLLM
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from file_utils import (charger_embeddings_rapport, charger_glossaire,
                        load_text, save_to_csv, load_paragraphs_from_csv, create_final_dataframe)
from llms import comparer_article_rapport_with_rag, configure_embeddings, analyze_paragraphs_parallel, create_prompt_template, generate_questions_parallel
from pdf_processing import process_pdf_to_index
from topic_classifier import keywords_for_each_chunck
from txt_manipulation import decouper_en_phrases, pretraiter_article


def run_script_1():
    """
    Première Partie : Nettoyage de l'article de Presse.
    Charge et prétraite l'article en supprimant les éléments non pertinents, puis le sauvegarde dans un dossier.
    """
    chemin_article = './IPCC_Answer_Based/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned.txt'
    chemin_dossier_nettoye = './IPCC_Answer_Based/Nettoye_Articles/'
    # Prétraiter l'article
    pretraiter_article(chemin_article, chemin_dossier_nettoye)


def run_script_2():
    """
    Seconde Partie : Nettoyage du rapport de synthèse et indexation.
    Extrait le texte d'un rapport PDF, le nettoie et l'indexe en sections, puis sauvegarde le tout dans un fichier JSON.
    """
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5")
    chemin_rapport_pdf = './IPCC_Answer_Based/IPCC_AR6_SYR_SPM.pdf'
    chemin_output_json = './IPCC_Answer_Based/rapport_indexed.json'
    # Traiter le PDF et sauvegarder les sections indexées
    process_pdf_to_index(chemin_rapport_pdf, chemin_output_json)


def run_script_3():
    """
    Troisième Partie : Identification des mentions directes/indirectes au GIEC.
    Compare les phrases d'un article avec les sections d'un rapport et identifie les termes du glossaire.
    Sauvegarde les résultats dans un fichier CSV.
    """
    chemin_cleaned_article = './IPCC_Answer_Based/Nettoye_Articles/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_resultats_csv = './IPCC_Answer_Based/mentions_extraites.csv'
    chemin_glossaire = './IPCC_Answer_Based/translated_glossary_with_definitions.csv'
    # chemin_rapport_embeddings = './IPCC_Answer_Based/rapport_indexed.json'

    # Charger le glossaire (termes et définitions)
    termes_glossaire, definitions_glossaire = charger_glossaire(
        chemin_glossaire)

    # Charger l'article nettoyé
    texte_nettoye = load_text(chemin_cleaned_article)

    # Découper l'article en phrases
    phrases = decouper_en_phrases(texte_nettoye)

    # Comparer l'article avec le rapport
    mentions = keywords_for_each_chunck(
        phrases, termes_glossaire, definitions_glossaire)

    # Sauvegarder les correspondances dans un fichier CSV
    save_to_csv(mentions, chemin_resultats_csv, [
        "phrase", "contexte", "glossary_terms", "definitions"])


def run_script_4():
    """
    Quatrième Partie : Vérification des mentions au GIEC dans un article.
    Utilise un LLM pour analyser chaque paragraphe d'un article de presse afin d'identifier s'il mentionne le climat,
    et pour lister tous les sujets abordés. Les résultats sont sauvegardés dans un fichier CSV.
    """
    # Path to the CSV file
    file_path = './IPCC_Answer_Based/mentions_extraites.csv'

    nltk.download('punkt')  # Download sentence tokenization model


    # Initialize the LLM (Ollama)
    llm = Ollama(model="llama3.2:3b-instruct-fp16")

    # Define the improved prompt template for LLM climate analysis in French with detailed instructions
    prompt_template = """
    Vous êtes un expert chargé d'identifier tous les sujets abordés dans le texte suivant, qu'ils soient ou non liés à l'environnement, au changement climatique ou au réchauffement climatique.

    Texte : {paragraph}

    1. Si le texte mentionne de près ou de loin l'environnement, le changement climatique, le réchauffement climatique, ou des organisations, événements ou accords liés à ces sujets (par exemple le GIEC, les conférences COP, les accords de Paris, etc.), répondez '1'. Sinon, répondez '0'.
    2. Listez **tous** les sujets abordés dans le texte, y compris ceux qui ne sont pas liés à l'environnement ou au climat.

    Format de réponse attendu :
    - Réponse binaire (0 ou 1) : [Réponse]
    - Liste des sujets abordés : [Sujet 1, Sujet 2, ...]

    Exemple de réponse :
    - Réponse binaire (0 ou 1) : 1
    - Liste des sujets abordés : [Incendies, gestion des forêts, réchauffement climatique, économie locale, GIEC]
    """

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["paragraph"])

    # Create the LLM chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)


    # Load paragraphs from the "contexte" column of the CSV
    paragraphs = load_paragraphs_from_csv(file_path)

    # Analyze the paragraphs with Llama 3.2 in parallel
    analysis_results = analyze_paragraphs_parallel(paragraphs, llm_chain)
    analysis_results_df = pd.DataFrame(analysis_results)


    # Save the initial analysis results to a CSV file using save_to_csv
    chemin_resultats_csv = './IPCC_Answer_Based/climate_analysis_results.csv'
    fieldnames = ["paragraph", "climate_related"]
    save_to_csv(analysis_results, chemin_resultats_csv, fieldnames)

    # Parse the analysis results to create a final DataFrame
    parsed_df = create_final_dataframe(analysis_results_df)

    # Process the 'subjects' column to convert lists to strings
    parsed_df['subjects'] = parsed_df['subjects'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Convert the DataFrame to a list of dictionaries for saving
    mentions = parsed_df.to_dict('records')

    # Save the parsed results to a CSV file using save_to_csv
    chemin_final_csv = './IPCC_Answer_Based/final_climate_analysis_results.csv'
    fieldnames_final = ["paragraph", "binary_response", "subjects"]
    save_to_csv(mentions, chemin_final_csv, fieldnames_final)


def run_script_5():
    """
    Cinquième Partie : Génération de questions pour les paragraphes liés à l'environnement.
    Utilise un modèle LLM pour générer des questions spécifiques pour vérifier les informations contenues
    dans des paragraphes classés comme étant liés à l'environnement.
    Les questions sont ensuite sauvegardées dans un fichier CSV.
    """
    output_path_questions = './IPCC_Answer_Based/final_climate_analysis_with_questions.csv'

    # Charger la base de données CSV contenant les paragraphes, la réponse binaire, et les thèmes
    df = pd.read_csv('./IPCC_Answer_Based/final_climate_analysis_results_improved.csv')

    # Initialize the LLM (Ollama)
    llm = Ollama(model="llama3.2:3b-instruct-fp16")

    # Créer le template de prompt
    prompt = create_prompt_template()


    # Convertir la colonne 'binary_response' en entier (si elle est en format texte)
    df['binary_response'] = pd.to_numeric(df['binary_response'], errors='coerce')

    # Filtrer uniquement les paragraphes identifiés comme liés à l'environnement (réponse binaire '1')
    df_environment = df[df['binary_response'] == 1]

    # Vérifier le DataFrame filtré
    print(df_environment)

    # Créer la LLMChain pour la génération des questions
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Générer les questions pour les paragraphes liés à l'environnement
    questions_df = generate_questions_parallel(df_environment, llm_chain)

    # Sauvegarder les résultats dans un nouveau fichier CSV
    questions_df.to_csv(output_path_questions, index=False)
    print(f"Questions generated and saved to {output_path_questions}")


def run_script_6():
    """
    Sixième Partie : RAG (Retrieve-and-Generate) avec Llama3.2.
    Utilise un modèle LLM pour générer des réponses à partir de sections pertinentes d'un rapport.
    Sauvegarde les résultats dans un fichier CSV.
    """
    chemin_cleaned_article = './IPCC_Answer_Based/Nettoye_Articles/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_resultats_csv = './IPCC_Answer_Based/mentions_rag_extraites.csv'
    chemin_rapport_embeddings = './IPCC_Answer_Based/rapport_indexed.json'

    # Configurer les embeddings (Ollama ou HuggingFace)
    configure_embeddings()

    # Charger l'article nettoyé
    texte_nettoye = load_text(chemin_cleaned_article)

    # Découper l'article en phrases
    phrases = decouper_en_phrases(texte_nettoye)

    # Charger les embeddings du rapport
    embeddings_rapport, sections_rapport = charger_embeddings_rapport(
        chemin_rapport_embeddings)

    # Définir le template de prompt pour la génération de réponses LLM
    prompt_template = """
    You are tasked with answering the following question based on the provided sections from the IPCC report.
    Ensure that your response is factual and based on the retrieved sections.
    
    Question: {question}
    Relevant Sections: {consolidated_text}
    
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "question", "consolidated_text"])

    # Créer la chaîne LLM avec Ollama
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Comparer l'article au rapport avec RAG et parallélisation
    mentions = comparer_article_rapport_with_rag(
        phrases, embeddings_rapport, sections_rapport, llm_chain)

    # Sauvegarder les résultats
    save_to_csv(mentions, chemin_resultats_csv, [
        "phrase", "retrieved_sections", "generated_answer"])


def run_all_scripts():
    """
    Exécute toutes les parties du script, dans l'ordre.
    """
    run_script_1()
    run_script_2()
    run_script_3()
    run_script_4()
    run_script_5()
    run_script_6()

if __name__ == "__main__":
    """
    Interface principale pour sélectionner et exécuter l'une des parties du script ou l'ensemble.
    """
    print("Choose an option:")
    print("1. Clean press articles")
    print("2. Embed IPCC report")
    print("3. Topic Recognition")
    print("4. Check for IPCC references")
    print("5. Create question for each chunk")
    print("6. Run RAG")
    print("7. Run all scripts")
    choice = input("Enter your choice: ")

    match choice:
        case "1":
            print("You chose Option 1")
            run_script_1()
        case "2":
            print("You chose Option 2")
            run_script_2()
        case "3":
            print("You chose Option 3")
            run_script_3()
        case "4":
            print("You chose Option 4")
            run_script_6()
        case "5":
            print("You chose Option 5")
            run_script_4()
        case "6":
            print("You chose Option 6")
            run_script_5()
        case "7":
            print("You chose Option 7")
            run_all_scripts()
        case _:
            print("Invalid choice. Please choose a valid option.")
