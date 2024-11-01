#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour le traitement d'un article de presse et d'un rapport du GIEC.
"""

from Evaluation_API import process_evaluation_api
from filtrer_extraits import identifier_extraits_sur_giec
from metrics import process_evaluation
from pdf_processing import process_pdf_to_index
from questions import question_generation_process
from reponse import process_reponses
from Reponse_API import rag_process_api
from Resume_API import process_resume_api
from resume_sources import process_resume
from topic_classifier import glossaire_topics
from txt_manipulation import pretraiter_article


def clean_press_article():
    """
    Première Partie : Nettoyage de l'article de Presse.
    """
    chemin_article = '_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_dossier_nettoye = '/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/Nettoye_Articles/'
    pretraiter_article(chemin_article, chemin_dossier_nettoye)


def process_ipcc_report():
    """
    Seconde Partie : Nettoyage du rapport de synthèse et indexation.
    """
    chemin_rapport_pdf = 'IPCC_AR6_SYR_SPM.pdf'
    chemin_output_json = '/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/rapport_indexed.json'
    process_pdf_to_index(chemin_rapport_pdf, chemin_output_json)


def identify_ipcc_mentions():
    """
    Troisième Partie : Identification des mentions directes/indirectes au GIEC.
    """
    chemin_cleaned_article = '_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_resultats_csv = '/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/mentions_extraites.csv'
    chemin_glossaire = 'translated_glossary_with_definitions.csv'
    glossaire_topics(chemin_glossaire, chemin_cleaned_article,
                     chemin_resultats_csv)


def extract_relevant_ipcc_references():
    """
    Quatrième Partie : Identification des extraits relatifs au GIEC.
    """
    file_path = "_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt"
    output_path = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/climate_analysis_results.csv"
    output_path_improved = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_results_improved.csv"
    identifier_extraits_sur_giec(file_path, output_path, output_path_improved)


def generate_questions():
    """
    Cinquième Partie : Génération de questions.
    """
    file_path = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_results_improved.csv"
    output_path_questions = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_with_questions.csv"
    question_generation_process(file_path, output_path_questions)


def summarize_source_sections(LocalLLM):
    """
    Résumé des sources pour chaque question.
    """
    chemin_csv_questions = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_with_questions.csv"
    chemin_resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/resume_sections_results.csv"
    chemin_rapport_embeddings = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/rapport_indexed.json"

    if LocalLLM:
        process_resume(chemin_csv_questions, chemin_rapport_embeddings,
                       chemin_resultats_csv, 5)  # Top-K = 5
    else:
        process_resume_api(chemin_csv_questions,
                           chemin_rapport_embeddings, chemin_resultats_csv, 5)


def generate_rag_responses(LocalLLM):
    """
    Sixième Partie : Génération de réponses (RAG).
    """
    chemin_questions_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/resume_sections_results.csv"
    chemin_resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/rag_results.csv"

    if LocalLLM:
        process_reponses(chemin_questions_csv, chemin_resultats_csv)
    else:
        rag_process_api(chemin_questions_csv, chemin_resultats_csv)


def evaluate_generated_responses(LocalLLM):
    """
    Septième Partie : Évaluation des réponses.
    """
    rag_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/rag_results.csv"
    resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/evaluation_results.csv"
    chemin_questions_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_with_questions.csv"

    if LocalLLM:
        process_evaluation(chemin_questions_csv, rag_csv, resultats_csv)
    else:
        process_evaluation_api(chemin_questions_csv, rag_csv, resultats_csv)


def run_full_processing_pipeline(LocalLLM):
    """
    Exécute toutes les parties du script, dans l'ordre.
    """
    clean_press_article()
    process_ipcc_report()
    extract_relevant_ipcc_references()
    generate_questions()
    summarize_source_sections(LocalLLM)
    generate_rag_responses(LocalLLM)
    evaluate_generated_responses(LocalLLM)


if __name__ == "__main__":
    # Demander à l'utilisateur s'il veut utiliser un LLM local
    use_local_llm = input(
        "Souhaitez-vous utiliser un LLM local ? (y/n) : ").strip().lower()
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
            clean_press_article()
        case "2":
            process_ipcc_report()
        case "3":
            identify_ipcc_mentions()
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
