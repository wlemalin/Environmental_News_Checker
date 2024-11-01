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

from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
from langchain import LLMChain, PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
from file_utils import save_to_csv
from llms import (analyze_paragraphs_parallel, create_questions_llm,
                  generate_questions_parallel, parsed_responses)
from pdf_processing import process_pdf_to_index

from topic_classifier import generate_context_windows, keywords_for_each_chunk
from txt_manipulation import decouper_en_phrases, pretraiter_article
from resume_sources import process_resume
from reponse import process_reponses
from metrics import process_evaluation

def run_script_1():
    """
    Première Partie : Nettoyage de l'article de Presse.
    Charge et prétraite l'article en supprimant les éléments non pertinents, puis le sauvegarde dans un dossier.
    """
    chemin_article = '/Users/mateodib/Desktop/Environmental_News_Checker-main/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_dossier_nettoye = '/Users/mateodib/Desktop/Environmental_News_Checker-main/Nettoye_Articles/'
    # Prétraiter l'article
    pretraiter_article(chemin_article, chemin_dossier_nettoye)


def run_script_2():
    """
    Seconde Partie : Nettoyage du rapport de synthèse et indexation.
    Extrait le texte d'un rapport PDF, le nettoie et l'indexe en sections, puis sauvegarde le tout dans un fichier JSON.
    """
    chemin_rapport_pdf = '/Users/mateodib/Desktop/Environmental_News_Checker-main/IPCC_AR6_SYR_SPM.pdf'
    chemin_output_json = '/Users/mateodib/Desktop/Environmental_News_Checker-main/rapport_indexed.json'
    # Traiter le PDF et sauvegarder les sections indexées
    process_pdf_to_index(chemin_rapport_pdf, chemin_output_json)


def run_script_3():
    """
    Troisième Partie : Identification des mentions directes/indirectes au GIEC.
    Compare les phrases d'un article avec les sections d'un rapport et identifie les termes du glossaire.
    Sauvegarde les résultats dans un fichier CSV.
    """
    chemin_cleaned_article = '/Users/mateodib/Desktop/Environmental_News_Checker-main/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_resultats_csv = '/Users/mateodib/Desktop/Environmental_News_Checker-main/mentions_extraites.csv'
    chemin_glossaire = '/Users/mateodib/Desktop/Environmental_News_Checker-main/translated_glossary_with_definitions.csv'
    # chemin_rapport_embeddings = './IPCC_Answer_Based/rapport_indexed.json'

    # Charger le glossaire (termes et définitions)
    glossaire = pd.read_csv(chemin_glossaire)
    termes_glossaire = glossaire['Translated_Term'].tolist()
    definitions_glossaire = glossaire['Translated_Definition'].tolist()

    with open(chemin_cleaned_article, 'r', encoding='utf-8') as file:
        texte_nettoye = file.read()
    # Découper l'article en phrases
    phrases = decouper_en_phrases(texte_nettoye)

    # Comparer l'article avec le rapport
    mentions = keywords_for_each_chunk(
        phrases, termes_glossaire, definitions_glossaire)

    # Sauvegarder les correspondances dans un fichier CSV
    save_to_csv(mentions, chemin_resultats_csv, [
        "phrase", "contexte", "glossary_terms", "definitions"])


def run_script_4():
    nltk.download('punkt')  # Téléchargez le modèle de tokenisation des phrases

    # Initialize the LLM (Ollama)
    llm = Ollama(model="llama3.2:3b-instruct-fp16")

    # Define the improved prompt template for LLM climate analysis in French with detailed instructions
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

    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "current_phrase", "context"])

    # Create the LLM chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Chemin vers le fichier texte
    file_path = "/Users/mateodib/Desktop/Environmental_News_Checker-main/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt"

    # Charger et regrouper le texte en phrases
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    sentences = sent_tokenize(text)  # Divise le texte en phrases

    splitted_text = generate_context_windows(sentences)

    # Analyser les paragraphes avec Llama 3.2 en parallèle
    analysis_results = analyze_paragraphs_parallel(splitted_text, llm_chain)

    # Sauvegarder les résultats dans un fichier CSV
    df = pd.DataFrame(analysis_results)
    output_path = "/Users/mateodib/Desktop/Environmental_News_Checker-main/climate_analysis_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")



    # Conversion de la liste en DataFrame
    analysis_results_df = pd.DataFrame(analysis_results)

    # Appliquer la méthode de parsing au DataFrame
    parsed_df_improved = parsed_responses(analysis_results_df)

    # Sauvegarder le DataFrame avec les résultats parsés
    output_path_improved = "/Users/mateodib/Desktop/Environmental_News_Checker-main/final_climate_analysis_results_improved.csv"
    parsed_df_improved['subjects'] = parsed_df_improved['subjects'].apply(
        lambda x: ', '.join(x))
    parsed_df_improved.to_csv(output_path_improved, index=False)

    # Affichage de quelques lignes du DataFrame final
    print(parsed_df_improved.head())


def run_script_5():
    # Charger la base de données CSV contenant les phrases, la réponse binaire, et le contexte
    df = pd.read_csv(
        "/Users/mateodib/Desktop/Environmental_News_Checker-main/final_climate_analysis_results_improved.csv")

    # Convertir la colonne 'binary_response' en texte (si elle est en format texte)
    df['binary_response'] = df['binary_response'].astype(str)

    # Filtrer uniquement les phrases identifiées comme liées à l'environnement (réponse binaire '1')
    df_environment = df[df['binary_response'] == '1']

    # Créer la LLMChain pour la génération des questions
    llm_chain = create_questions_llm()

    # Générer les questions pour les phrases liées à l'environnement
    questions_df = generate_questions_parallel(df_environment, llm_chain)

    # Sauvegarder les résultats dans un nouveau fichier CSV
    output_path_questions = "/Users/mateodib/Desktop/Environmental_News_Checker-main/final_climate_analysis_with_questions.csv"
    questions_df.to_csv(output_path_questions, index=False)
    print(f"Questions generated and saved to {output_path_questions}")


def resume_sources():
    chemin_csv_questions = "/Users/mateodib/Desktop/Environmental_News_Checker-main/final_climate_analysis_with_questions.csv" # final_climate_analysis_with_questions.csv TODO
    chemin_resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-main/resume_sections_results.csv"
    chemin_rapport_embeddings = "/Users/mateodib/Desktop/Environmental_News_Checker-main/rapport_indexed.json"
    process_resume(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, 5) # Top-K = 5

def run_script_6():
    chemin_questions_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-main/resume_sections_results.csv"
    chemin_resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-main/rag_results.csv"
    process_reponses(chemin_questions_csv, chemin_resultats_csv)


def run_script_7():
    rag_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-main/rag_results.csv"
    resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-main/evaluation_results.csv"
    process_evaluation(rag_csv, resultats_csv)


def run_all_scripts():
    """
    Exécute toutes les parties du script, dans l'ordre.
    """
    run_script_1()
    run_script_2()
    run_script_3()
    run_script_4()
    run_script_5()  # Create questions
    resume_sources()
    run_script_6()  # Answer questions
    run_script_7()


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
    print("t. Resume sections source")
    print("6. Run RAG on questions")
    print("7. Get metrics")
    print("8. Run all scripts")
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
            run_script_4()
        case "5":
            print("You chose Option 5")
            run_script_5()
        case "t":
            print("You chose Option t")
            resume_sources()
        case "6":
            print("You chose Option 6")
            run_script_6()
        case "7":
            print("You chose Option 7")
            run_script_7()
        case "8":
            print("You chose option 8.")
            run_all_scripts()
        case _:
            print("Invalid choice. Please choose a valid option.")



