#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour le traitement d'un article de presse et d'un rapport du GIEC.
Ce script effectue plusieurs tâches liées au traitement de texte et à l'intelligence artificielle, réparties en plusieurs étapes :

1. Nettoyage et prétraitement de l'article de presse : Suppression des éléments non pertinents et préparation du texte pour les étapes suivantes.
2. Extraction, nettoyage et indexation des sections du rapport PDF : Transformation du rapport en texte exploitable, puis découpage en sections indexées.
3. Identification des mentions directes et indirectes au GIEC : Utilisation d'un modèle d'embeddings pour comparer les phrases de l'article avec les sections du rapport, et détection des termes du glossaire.
4. Génération de questions, résumés et vérification des faits avec un modèle LLM : Génération de réponses basées sur les sections extraites du rapport en utilisant un modèle de type RAG (Retrieve-and-Generate).
5. Évaluation des réponses générées.
"""

from filtrer_extraits import identifier_extraits_sur_giec
from metrics import process_evaluation
from pdf_processing import process_pdf_to_index
from reponse import process_reponses
from resume_sources import process_resume
from topic_classifier import glossaire_topics
from txt_manipulation import pretraiter_article
from questions import question_generation_process


def clean_press_article():
    """
    Première Partie : Nettoyage de l'article de Presse.
    Charge et prétraite l'article en supprimant les éléments non pertinents, puis le sauvegarde dans un dossier spécifié.
    """
    chemin_article = '_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_dossier_nettoye = '/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/Nettoye_Articles/'
    # Prétraiter l'article
    pretraiter_article(chemin_article, chemin_dossier_nettoye)


def process_ipcc_report():
    """
    Seconde Partie : Nettoyage du rapport de synthèse et indexation.
    Extrait le texte d'un rapport PDF, le nettoie et l'indexe en sections, puis sauvegarde le tout dans un fichier JSON.
    """
    chemin_rapport_pdf = 'IPCC_AR6_SYR_SPM.pdf'
    chemin_output_json = '/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/rapport_indexed.json'
    # Traiter le PDF et sauvegarder les sections indexées
    process_pdf_to_index(chemin_rapport_pdf, chemin_output_json)


def identify_ipcc_mentions():
    """
    Troisième Partie : Identification des mentions directes/indirectes au GIEC.
    Compare les phrases d'un article avec les sections d'un rapport et identifie les termes du glossaire.
    Sauvegarde les résultats dans un fichier CSV.
    """
    chemin_cleaned_article = '_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_resultats_csv = '/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/mentions_extraites.csv'
    chemin_glossaire = 'translated_glossary_with_definitions.csv'
    glossaire_topics(chemin_glossaire, chemin_cleaned_article, chemin_resultats_csv)


def extract_relevant_ipcc_references():
    """
    Quatrième Partie : Identification des extraits relatifs au GIEC.
    Identifie les extraits pertinents de l'article qui mentionnent directement ou indirectement le GIEC et améliore les résultats.
    """
    file_path = "_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt"
    output_path = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/climate_analysis_results.csv"
    output_path_improved = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_results_improved.csv"
    identifier_extraits_sur_giec(file_path, output_path, output_path_improved)


def generate_questions():
    """
    Cinquième Partie : Génération de questions.
    Génère des questions basées sur les extraits améliorés de l'article et sauvegarde les questions dans un fichier CSV.
    """
    file_path = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_results_improved.csv"
    output_path_questions = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_with_questions.csv"
    question_generation_process(file_path, output_path_questions)


def summarize_source_sections():
    """
    Résumé des sources pour chaque question.
    Prend les questions générées, trouve les sections du rapport les plus pertinentes et les résume pour chaque question.
    """
    chemin_csv_questions = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_with_questions.csv"
    chemin_resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/resume_sections_results.csv"
    chemin_rapport_embeddings = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/rapport_indexed.json"
    process_resume(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, 5)  # Top-K = 5


def generate_rag_responses():
    """
    Sixième Partie : Génération de réponses (RAG).
    Utilise un modèle de type RAG (Retrieve-and-Generate) pour répondre aux questions générées.
    """
    chemin_questions_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/resume_sections_results.csv"
    chemin_resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/rag_results.csv"
    process_reponses(chemin_questions_csv, chemin_resultats_csv)


def evaluate_generated_responses():
    """
    Septième Partie : Évaluation des réponses.
    Évalue les résultats des réponses générées par le modèle et sauvegarde les évaluations dans un fichier CSV.
    """
    rag_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/rag_results.csv"
    resultats_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/evaluation_results.csv"
    chemin_questions_csv = "/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_with_questions.csv"
    process_evaluation(chemin_questions_csv, rag_csv, resultats_csv)


def run_full_processing_pipeline():
    """
    Exécute toutes les parties du script, dans l'ordre, afin de traiter complètement un article de presse et un rapport du GIEC.
    """
    clean_press_article()
    process_ipcc_report()
    # identify_ipcc_mentions()
    extract_relevant_ipcc_references()
    generate_questions()  # Create questions
    summarize_source_sections()
    generate_rag_responses()  # Answer questions
    evaluate_generated_responses()


if __name__ == "__main__":
    """
    Interface principale pour sélectionner et exécuter l'une des parties du script ou l'ensemble.
    """
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
            print("You chose Option 1")
            clean_press_article()
        case "2":
            print("You chose Option 2")
            process_ipcc_report()
        case "3":
            print("You chose Option 3")
            identify_ipcc_mentions()
        case "4":
            print("You chose Option 4")
            extract_relevant_ipcc_references()
        case "5":
            print("You chose Option 5")
            generate_questions()
        case "6":
            print("You chose Option 6")
            summarize_source_sections()
        case "7":
            print("You chose Option 7")
            generate_rag_responses()
        case "8":
            print("You chose Option 8")
            evaluate_generated_responses()
        case "9":
            print("You chose option 9.")
            run_full_processing_pipeline()
        case _:
            print("Invalid choice. Please choose a valid option.")
            
            
            
