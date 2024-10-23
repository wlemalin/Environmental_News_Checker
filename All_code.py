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

from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain_ollama import OllamaLLM
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from file_utils import (charger_embeddings_rapport, charger_glossaire,
                        load_text, sauvegarder_mentions_csv)
from txt_manipulation import decouper_en_phrases, pretraiter_article
from llms import comparer_article_rapport_with_rag, configure_embeddings
from topic_classifier import comparer_article_rapport
from pdf_processing import process_pdf_to_index


def run_script_1():
    """
    Première Partie : Nettoyage de l'article de Presse.
    Charge et prétraite l'article en supprimant les éléments non pertinents, puis le sauvegarde dans un dossier.
    """
    chemin_article = '/Users/mateodib/Desktop/IPCC_Answer_Based/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned.txt'
    chemin_dossier_nettoye = '/Users/mateodib/Desktop/IPCC_Answer_Based/Nettoye_Articles/'
    # Prétraiter l'article
    pretraiter_article(chemin_article, chemin_dossier_nettoye)


def run_script_2():
    """
    Seconde Partie : Nettoyage du rapport de synthèse et indexation.
    Extrait le texte d'un rapport PDF, le nettoie et l'indexe en sections, puis sauvegarde le tout dans un fichier JSON.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    chemin_rapport_pdf = '/Users/mateodib/Desktop/IPCC_Answer_Based/IPCC_AR6_SYR_SPM.pdf'
    chemin_output_json = '/Users/mateodib/Desktop/IPCC_Answer_Based/rapport_indexed.json'
    # Traiter le PDF et sauvegarder les sections indexées
    process_pdf_to_index(chemin_rapport_pdf, chemin_output_json)


def run_script_3():
    """
    Troisième Partie : Identification des mentions directes/indirectes au GIEC.
    Compare les phrases d'un article avec les sections d'un rapport et identifie les termes du glossaire.
    Sauvegarde les résultats dans un fichier CSV.
    """
    chemin_cleaned_article = '/Users/mateodib/Desktop/IPCC_Answer_Based/Nettoye_Articles/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_resultats_csv = '/Users/mateodib/Desktop/IPCC_Answer_Based/mentions_extraites.csv'
    chemin_rapport_embeddings = '/Users/mateodib/Desktop/IPCC_Answer_Based/rapport_indexed.json'
    chemin_glossaire = '/Users/mateodib/Desktop/IPCC_Answer_Based/translated_glossary_with_definitions.csv'
    
    # Charger le glossaire (termes et définitions)
    termes_glossaire, definitions_glossaire = charger_glossaire(chemin_glossaire)
    
    # Charger l'article nettoyé
    texte_nettoye = load_text(chemin_cleaned_article)
    
    # Découper l'article en phrases
    phrases = decouper_en_phrases(texte_nettoye)
    
    # Charger les embeddings du rapport
    embeddings_rapport, sections_rapport = charger_embeddings_rapport(chemin_rapport_embeddings)
    
    # Comparer l'article avec le rapport
    mentions = comparer_article_rapport(phrases, embeddings_rapport, sections_rapport, termes_glossaire, definitions_glossaire)
    
    # Sauvegarder les correspondances dans un fichier CSV
    sauvegarder_mentions_csv(mentions, chemin_resultats_csv, [
                             "phrase", "section", "similarite", "glossary_terms", "definitions"])


def run_script_4():
    """
    Quatrième Partie : RAG (Retrieve-and-Generate) avec Llama3.2.
    Utilise un modèle LLM pour générer des réponses à partir de sections pertinentes d'un rapport.
    Sauvegarde les résultats dans un fichier CSV.
    """
    chemin_cleaned_article = '/Users/mateodib/Desktop/IPCC_Answer_Based/Nettoye_Articles/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_resultats_csv = '/Users/mateodib/Desktop/IPCC_Answer_Based/mentions_rag_extraites.csv'
    chemin_rapport_embeddings = '/Users/mateodib/Desktop/IPCC_Answer_Based/rapport_indexed.json'
    
    # Configurer les embeddings (Ollama ou HuggingFace)
    configure_embeddings(use_ollama=False)
    
    # Charger l'article nettoyé
    texte_nettoye = load_text(chemin_cleaned_article)
    
    # Découper l'article en phrases
    phrases = decouper_en_phrases(texte_nettoye)
    
    # Charger les embeddings du rapport
    embeddings_rapport, sections_rapport = charger_embeddings_rapport(chemin_rapport_embeddings)
    
    # Définir le template de prompt pour la génération de réponses LLM
    prompt_template = """
    You are tasked with answering the following question based on the provided sections from the IPCC report.
    Ensure that your response is factual and based on the retrieved sections.
    
    Question: {question}
    Relevant Sections: {consolidated_text}
    
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "consolidated_text"])

    # Créer la chaîne LLM avec Ollama
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # Comparer l'article au rapport avec RAG et parallélisation
    mentions = comparer_article_rapport_with_rag(phrases, embeddings_rapport, sections_rapport, llm_chain)
    
    # Sauvegarder les résultats
    sauvegarder_mentions_csv(mentions, chemin_resultats_csv, [
                             "phrase", "retrieved_sections", "generated_answer"])


def run_all_scripts():
    """
    Exécute toutes les parties du script, dans l'ordre.
    """
    run_script_1()
    run_script_2()
    run_script_3()
    run_script_4()


if __name__ == "__main__":
    """
    Interface principale pour sélectionner et exécuter l'une des parties du script ou l'ensemble.
    """
    print("Choose an option:")
    print("1. Clean press articles")
    print("2. Embed IPCC report")
    print("3. Topic Recognition")
    print("4. Fact Check with LLM")
    print("5. Run all scripts")
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
            run_all_scripts()
        case _:
            print("Invalid choice. Please choose a valid option.")
