#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour le traitement d'un article de presse et d'un rapport du GIEC.
"""

import os

from Evaluation_API import process_evaluation_api
from filtrer_extraits import identifier_extraits_sur_giec
from metrics import process_evaluation
from pdf_processing import process_pdf_to_index
from questions import question_generation_process
from reponse import process_reponses
from Reponse_API import rag_process_api
from Resume_API import process_resume_api
from resume_sources import process_resume
from txt_manipulation import pretraiter_article

# from topic_classifier import glossaire_topics


def clean_press_articles():
    """
    Première Partie : Nettoyage de plusieurs articles de presse.
    """
    chemin_articles = 'Data/presse/articles/'
    chemin_dossier_nettoye = 'Data/presse/articles_cleaned/'

    # Lister tous les fichiers .txt dans le dossier des articles
    fichiers_articles = [f for f in os.listdir( chemin_articles) if f.endswith('.txt')]

    if not os.path.exists(os.path.dirname(chemin_dossier_nettoye)):
        os.makedirs(os.path.dirname(chemin_dossier_nettoye))

    # Itérer sur chaque fichier d'article
    for fichier in fichiers_articles:
        chemin_article = os.path.join(chemin_articles, fichier)
        chemin_article_nettoye = os.path.join(chemin_dossier_nettoye, fichier.replace('.txt', '_cleaned.txt'))
        pretraiter_article(chemin_article, chemin_article_nettoye, chemin_dossier_nettoye)


def process_ipcc_reports():
    """
    Seconde Partie : Nettoyage et indexation de plusieurs rapports IPCC.
    """
    chemin_rapports_pdf = 'Data/IPCC/rapports/'
    chemin_output_indexed = 'Data/IPCC/rapports_indexed/'

    # Lister tous les fichiers .pdf dans le dossier des rapports
    fichiers_rapports = [f for f in os.listdir(chemin_rapports_pdf) if f.endswith('.pdf')]

    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(os.path.dirname(chemin_output_indexed)):
        os.makedirs(os.path.dirname(chemin_output_indexed))

    # Itérer sur chaque fichier de rapport
    for fichier in fichiers_rapports:
        chemin_rapport_pdf = os.path.join(chemin_rapports_pdf, fichier)
        chemin_rapport_indexed = os.path.join(chemin_output_indexed, fichier.replace('.pdf', '_indexed.json'))
        process_pdf_to_index(chemin_rapport_pdf, chemin_rapport_indexed)

# def identify_ipcc_mentions():
#     """
#     Troisième Partie : Identification des mentions directes/indirectes au GIEC.
#     """
#     chemin_cleaned_article = '_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
#     chemin_resultats_csv = '/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/mentions_extraites.csv'
#     chemin_glossaire = 'translated_glossary_with_definitions.csv'
#     glossaire_topics(chemin_glossaire, chemin_cleaned_article,
#                      chemin_resultats_csv)


def extract_relevant_ipcc_references():
    """
    Quatrième Partie : Identification des extraits relatifs au GIEC.
    """
    chemin_articles_nettoyes = 'Data/presse/articles_cleaned/'
    chemin_output_chunked = 'Data/presse/articles_chunked/'

    # Lister tous les fichiers .txt nettoyés dans le dossier des articles
    fichiers_articles_nettoyes = [f for f in os.listdir(chemin_articles_nettoyes) if f.endswith('_cleaned.txt')]

    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(os.path.dirname(chemin_output_chunked)):
        os.makedirs(os.path.dirname(chemin_output_chunked))

    # Itérer sur chaque fichier nettoyé
    for fichier in fichiers_articles_nettoyes:
        file_path = os.path.join(chemin_articles_nettoyes, fichier)
        output_path = os.path.join(chemin_output_chunked, fichier.replace('_cleaned.txt', '_analysis_results.csv'))
        output_path_improved = os.path.join(chemin_output_chunked, fichier.replace('_cleaned.txt', '_final_analysis_results_improved.csv'))
        identifier_extraits_sur_giec(file_path, output_path, output_path_improved)


def generate_questions():
    """
    Cinquième Partie : Génération de questions pour plusieurs fichiers.
    """
    chemin_articles_chunked = 'Data/presse/articles_chunked/'
    chemin_output_questions = 'Data/resultats/resultats_intermediaires/questions/'

    # Lister tous les fichiers .csv dans le dossier des articles analysés
    fichiers_analysis_results = [f for f in os.listdir(chemin_articles_chunked) if f.endswith('_final_analysis_results_improved.csv')]

    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(os.path.dirname(chemin_output_questions)):
        os.makedirs(os.path.dirname(chemin_output_questions))

    # Itérer sur chaque fichier d'analyse
    for fichier in fichiers_analysis_results:
        file_path = os.path.join(chemin_articles_chunked, fichier)
        output_path_questions = os.path.join(chemin_output_questions, fichier.replace('_final_analysis_results_improved.csv', '_with_questions.csv'))
        question_generation_process(file_path, output_path_questions)

def summarize_source_sections(LocalLLM):
    """
    Résumé des sources pour chaque question pour plusieurs fichiers.
    """
    chemin_csv_questions = 'Data/resultats/resultats_intermediaires/questions/'
    chemin_resultats_sources = 'Data/resultats/resultats_intermediaires/sources_resumees/'
    chemin_rapport_embeddings = 'Data/IPCC/rapports_indexed/rapport_indexed.json'

    # Lister tous les fichiers .csv dans le dossier des questions générées
    fichiers_questions = [f for f in os.listdir(chemin_csv_questions) if f.endswith('_with_questions.csv')]

    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(os.path.dirname(chemin_resultats_sources)):
        os.makedirs(os.path.dirname(chemin_resultats_sources))

    # Itérer sur chaque fichier de questions
    for fichier in fichiers_questions:
        chemin_csv_question = os.path.join(chemin_csv_questions, fichier)
        chemin_resultats_csv = os.path.join(chemin_resultats_sources, fichier.replace('_with_questions.csv', '_resume_sections_results.csv'))

        if LocalLLM:
            process_resume(chemin_csv_question, chemin_rapport_embeddings, chemin_resultats_csv, 5)  # Top-K = 5
        else:
            process_resume_api(chemin_csv_question, chemin_rapport_embeddings, chemin_resultats_csv, 5)


def generate_rag_responses(LocalLLM):
    """
    Sixième Partie : Génération de réponses (RAG) pour plusieurs fichiers.
    """
    chemin_sources_resumees = 'Data/resultats/resultats_intermediaires/sources_resumees/'
    chemin_output_reponses = 'Data/resultats/resultats_intermediaires/reponses/'

    # Lister tous les fichiers .csv dans le dossier des résumés de sources
    fichiers_sources_resumees = [f for f in os.listdir(chemin_sources_resumees) if f.endswith('_resume_sections_results.csv')]

    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(os.path.dirname(chemin_output_reponses)):
        os.makedirs(os.path.dirname(chemin_output_reponses))

    # Itérer sur chaque fichier de résumés de sources
    for fichier in fichiers_sources_resumees:
        chemin_questions_csv = os.path.join(chemin_sources_resumees, fichier)
        chemin_resultats_csv = os.path.join(chemin_output_reponses, fichier.replace('_resume_sections_results.csv', '_rag_results.csv'))

        if LocalLLM:
            process_reponses(chemin_questions_csv, chemin_resultats_csv)
        else:
            rag_process_api(chemin_questions_csv, chemin_resultats_csv)


def evaluate_generated_responses(LocalLLM):
    """
    Septième Partie : Évaluation des réponses pour plusieurs fichiers.
    """
    chemin_reponses = 'Data/resultats/resultats_intermediaires/reponses/'
    chemin_output_evaluation = 'Data/resultats/resultats_intermediaires/evaluation/'
    chemin_questions_csv = 'Data/resultats/resultats_intermediaires/questions/'

    # Lister tous les fichiers .csv dans le dossier des réponses générées
    fichiers_reponses = [f for f in os.listdir(chemin_reponses) if f.endswith('_rag_results.csv')]

    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(os.path.dirname(chemin_output_evaluation)):
        os.makedirs(os.path.dirname(chemin_output_evaluation))

    # Itérer sur chaque fichier de réponses
    for fichier in fichiers_reponses:
        rag_csv = os.path.join(chemin_reponses, fichier)
        resultats_csv = os.path.join(chemin_output_evaluation, fichier.replace('_rag_results.csv', '_evaluation_results.csv'))
        chemin_question_csv = os.path.join(chemin_questions_csv, fichier.replace('_rag_results.csv', '_with_questions.csv'))

        if LocalLLM:
            process_evaluation(chemin_question_csv, rag_csv, resultats_csv)
        else:
            process_evaluation_api(chemin_question_csv, rag_csv, resultats_csv)


def run_full_processing_pipeline(LocalLLM):
    """
    Exécute toutes les parties du script, dans l'ordre.
    """
    clean_press_articles()
    process_ipcc_reports()
    # extract_relevant_ipcc_references()
    generate_questions()
    summarize_source_sections(LocalLLM)
    generate_rag_responses(LocalLLM)
    evaluate_generated_responses(LocalLLM)


if __name__ == "__main__":
    # Demander à l'utilisateur s'il veut utiliser un LLM local
    use_local_llm = input("Souhaitez-vous utiliser un LLM local ? (y/n) : ").strip().lower()
    LocalLLM = use_local_llm == 'y'

    print("Choose an option:")
    print("1. Clean press article")
    print("2. Process IPCC report")
    print("3. Identify IPCC mentions")
    print("4. Extract relevant IPCC references")
    print("5. Generate questions for each chunk")
    print("6. Summarize source sections")
    print("7. Generate RAG responses for questions")
    print("8. Evaluate generated responses")
    print("9. Run full processing pipeline")
    choice = input("Enter your choice: ")

    match choice:
        case "1":
            clean_press_articles()
        case "2":
            process_ipcc_reports()
        # case "3":
            # identify_ipcc_mentions()
        case "4":
            extract_relevant_ipcc_references()
        case "5":
            generate_questions()
        case "6":
            summarize_source_sections(LocalLLM)
        case "7":
            generate_rag_responses(LocalLLM)
        case "8":
            evaluate_generated_responses(LocalLLM)
        case "9":
            run_full_processing_pipeline(LocalLLM)
        case _:
            print("Invalid choice. Please choose a valid option.")
