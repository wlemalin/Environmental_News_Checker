#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:58:14 2024

@author: mateodib
"""


"""
Première Partie : Nettoyage de l'article de Presse
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
    chemin_article = '/Users/mateodib/Desktop/IPCC_Answer_Based/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned.txt'
    chemin_dossier_nettoye = '/Users/mateodib/Desktop/IPCC_Answer_Based/Nettoye_Articles/'
    # Prétraiter l'article
    pretraiter_article(chemin_article, chemin_dossier_nettoye)


def run_script_2():
    """
    Seconde Partie : Nettoyage du rapport de synthèse et indexation
    """
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5")
    chemin_rapport_pdf = '/Users/mateodib/Desktop/IPCC_Answer_Based/IPCC_AR6_SYR_SPM.pdf'
    chemin_output_json = '/Users/mateodib/Desktop/IPCC_Answer_Based/rapport_indexed.json'
    # Process the PDF and save the indexed sections
    process_pdf_to_index(chemin_rapport_pdf, chemin_output_json)


def run_script_3():
    """
    Troisième Partie : Identification des mentions directs/indirectes au GIEC
    """
    chemin_cleaned_article = '/Users/mateodib/Desktop/IPCC_Answer_Based/Nettoye_Articles/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_resultats_csv = '/Users/mateodib/Desktop/IPCC_Answer_Based/mentions_extraites.csv'
    chemin_rapport_embeddings = '/Users/mateodib/Desktop/IPCC_Answer_Based/rapport_indexed.json'
    chemin_glossaire = '/Users/mateodib/Desktop/IPCC_Answer_Based/translated_glossary_with_definitions.csv'
    # Charger le glossaire (termes et définitions)
    termes_glossaire, definitions_glossaire = charger_glossaire(
        chemin_glossaire)
    # Charger l'article nettoyé
    texte_nettoye = load_text(chemin_cleaned_article)
    # Découper l'article en phrases
    phrases = decouper_en_phrases(texte_nettoye)
    # Charger les embeddings du rapport
    embeddings_rapport, sections_rapport = charger_embeddings_rapport(
        chemin_rapport_embeddings)
    # Comparer l'article avec le rapport
    mentions = comparer_article_rapport(
        phrases, embeddings_rapport, sections_rapport, termes_glossaire, definitions_glossaire)
    # Sauvegarder les correspondances dans un fichier CSV
    sauvegarder_mentions_csv(mentions, chemin_resultats_csv, [
                             "phrase", "section", "similarite", "glossary_terms", "definitions"])


def run_script_4():
    """
    Quatrième Partie : RAG avec Llama3.2
    """
    chemin_cleaned_article = '/Users/mateodib/Desktop/IPCC_Answer_Based/Nettoye_Articles/_ _ C_est plus confortable de se dire que ce n_est pas si grave __cleaned_cleaned.txt'
    chemin_resultats_csv = '/Users/mateodib/Desktop/IPCC_Answer_Based/mentions_rag_extraites.csv'
    chemin_rapport_embeddings = '/Users/mateodib/Desktop/IPCC_Answer_Based/rapport_indexed.json'
    # Configure embeddings for Ollama or HuggingFace
    configure_embeddings(use_ollama=False)
    # Load the cleaned article
    texte_nettoye = load_text(chemin_cleaned_article)
    # Split article into sentences
    phrases = decouper_en_phrases(texte_nettoye)
    # Load report embeddings
    embeddings_rapport, sections_rapport = charger_embeddings_rapport(
        chemin_rapport_embeddings)
    # Define the prompt template for LLM response generation
    prompt_template = """
    You are tasked with answering the following question based on the provided sections from the IPCC report.
    Ensure that your response is factual and based on the retrieved sections.
    
    Question: {question}
    Relevant Sections: {consolidated_text}
    
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "question", "consolidated_text"])

    # Create the LLM chain with Ollama
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    # Compare the article to the report using RAG with parallelization and progress bar
    mentions = comparer_article_rapport_with_rag(
        phrases, embeddings_rapport, sections_rapport, llm_chain)
    # Save the results
    sauvegarder_mentions_csv(mentions, chemin_resultats_csv, [
                             "phrase", "retrieved_sections", "generated_answer"])


def run_all_scripts():
    run_script_1()
    run_script_2()
    run_script_3()
    run_script_4()


if __name__ == "__main__":
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
