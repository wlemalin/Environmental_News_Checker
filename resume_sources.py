from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from embeddings_creation import embed_texts, generer_embeddings_rapport
from prompt import creer_llm_resume


def charger_donnees_et_modele(chemin_csv_questions, chemin_rapport_embeddings):
    """
    Loads the questions data from a CSV file and report embeddings from the specified file.

    Args:
        chemin_csv_questions (str): Path to the CSV file containing the questions.
        chemin_rapport_embeddings (str): Path to the file containing the report embeddings.

    Returns:
        tuple: A tuple containing the DataFrame of questions, report embeddings, report sections, report titles, and the embedding model.
    """
    # Charger les questions
    df_questions = pd.read_csv(chemin_csv_questions)
    print(f"Questions loaded. Total: {len(df_questions)}")

    # Configurer le modèle d'embedding et charger les embeddings et sections du rapport
    embed_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    embeddings_rapport, sections_rapport, titles_rapport = generer_embeddings_rapport(
        chemin_rapport_embeddings, embed_model)
    print("Report embeddings and sections loaded.")

    # Vérifier la cohérence des données
    if not (len(embeddings_rapport) == len(sections_rapport) == len(titles_rapport)):
        print(
            "Erreur : Le nombre d'embeddings, de sections, et de titres ne correspond pas.")
    else:
        print(
            f"Données chargées avec succès : {len(embeddings_rapport)} embeddings, {len(sections_rapport)} sections, et {len(titles_rapport)} titres.")

    # Afficher quelques exemples pour vérification
    print("Exemples de sections et leurs embeddings :")
    for i in range(min(3, len(sections_rapport))):
        print(f"Titre : {titles_rapport[i]}")
        # Limiter l'affichage à 100 caractères
        print(f"Texte : {sections_rapport[i][:100]}...")
        print(f"Embedding (taille) : {len(embeddings_rapport[i])}\n")

    return df_questions, embeddings_rapport, sections_rapport, titles_rapport, embed_model


def filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k=5):
    """
    Filters the most relevant sections of the report for each question based on cosine similarity.

    Args:
        df_questions (DataFrame): DataFrame containing the questions.
        embed_model: Embedding model used to generate embeddings for the questions.
        embeddings_rapport (list): List of embeddings for the report sections.
        sections_rapport (list): List of report sections.
        top_k (int, optional): The number of most similar sections to retrieve for each question. Default is 5.

    Returns:
        DataFrame: DataFrame with an additional column containing the retrieved relevant sections for each question.
    """
    retrieved_sections_list = []

    for _, row in df_questions.iterrows():
        question_embedding = embed_texts([row['question']], embed_model)[0]

        # Calculer les similarités et sélectionner les top-k indices
        similarites = util.cos_sim(question_embedding, torch.tensor(
            embeddings_rapport, device='cpu'))
        top_k_indices = np.argsort(-similarites[0].cpu()).tolist()[:top_k]

        # Récupérer les sections correspondantes
        sections = [sections_rapport[i]
                    for i in top_k_indices if sections_rapport[i].strip()]
        retrieved_sections_list.append(sections)

    # Ajouter les sections retrouvées comme nouvelle colonne
    df_questions['retrieved_sections'] = retrieved_sections_list
    return df_questions


def generer_resume_parallel(df_questions, llm_chain_resume):
    """
    Summarizes each associated section of each question in parallel using the LLM.

    Args:
        df_questions (DataFrame): DataFrame containing questions and their retrieved sections.
        llm_chain_resume: LLM chain for generating section summaries.

    Returns:
        DataFrame: DataFrame containing question IDs, questions, original sections, and their corresponding summaries.
    """
    resultats = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        # Iterate over each question and its associated sections
        for _, row in df_questions.iterrows():
            question_id = row['id']
            question_text = row['question']
            # List of sections for this question
            retrieved_sections = row['retrieved_sections']
            # For each section associated with the question, create a future for summarization
            for section in retrieved_sections:
                futures.append(executor.submit(
                    generer_resume,
                    question_id,
                    question_text,
                    section,  # Pass each section individually
                    llm_chain_resume
                ))
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Résumés des sections"):
            try:
                result = future.result()
                resultats.append(result)
            except Exception as exc:
                print(f"Erreur lors du résumé d'une question : {exc}")
    # Combine all results into a DataFrame for saving or further processing
    return pd.DataFrame(resultats)


def generer_resume(phrase_id, question, section, llm_chain_resume):
    """
    Generates a summary for a single section of a question using the LLM.

    Args:
        phrase_id (int): The unique identifier for the question.
        question (str): The question context for generating the summary.
        section (str): The specific section of the report to summarize.
        llm_chain_resume: LLM chain used for summarization.

    Returns:
        dict: A dictionary containing the question ID, question text, original section, and the section summary.
    """
    # Generate the summary for the individual section with the given question context
    response_resume = llm_chain_resume.invoke({
        "question": question,
        "retrieved_sections": section
    })
    # Extract and clean up the generated summary text
    resume = response_resume['text'].strip() if isinstance(
        response_resume, dict) and "text" in response_resume else response_resume.strip()
    # Return the result as a dictionary with ID, question, original section, and the section summary
    return {
        "id": phrase_id,               # Add ID to the results
        "question": question,           # Add question to the results
        "section": section,             # The original section being summarized
        # Summary of the specific section in context of the question
        "resume_section": resume
    }


def process_resume(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, top_k=3):
    """
    Executes the full process of loading questions, filtering relevant sections, summarizing sections, and saving results to a CSV file.

    Args:
        chemin_csv_questions (str): Path to the CSV file containing the questions.
        chemin_rapport_embeddings (str): Path to the file containing report embeddings.
        chemin_resultats_csv (str): Path to the output CSV file where results will be saved.
        top_k (int, optional): The number of most similar sections to retrieve for each question. Default is 3.

    Workflow:
        1. Load questions and report embeddings.
        2. Filter the most relevant sections for each question.
        3. Summarize the relevant sections for each question in parallel.
        4. Group results by question ID to create a consolidated summary.
        5. Save the consolidated summaries to an output CSV file.
    """
    # Charger les données et le modèle
    df_questions, embeddings_rapport, sections_rapport, _, embed_model = charger_donnees_et_modele(
        chemin_csv_questions, chemin_rapport_embeddings)

    # Filtrer les sections pertinentes et obtenir un DataFrame avec retrieved_sections
    df_questions = filtrer_sections_pertinentes(
        df_questions, embed_model, embeddings_rapport, sections_rapport, top_k)

    # Configure the LLM chain for summarization
    llm_chain_resume = creer_llm_resume()

    # Summarize sections per question
    resultats = generer_resume_parallel(df_questions, llm_chain_resume)

    # Group by question ID to concatenate sections and summaries
    resultats_grouped = resultats.groupby('id').agg({
        # Take the first instance of the question (as it's the same for each ID)
        'question': 'first',
        'section': lambda x: ' '.join(x),  # Concatenate all sections
        'resume_section': lambda x: ' '.join(x)  # Concatenate all summaries
    }).reset_index()

    # Rename columns for clarity
    resultats_grouped.rename(
        columns={'section': 'sections', 'resume_section': 'resume_sections'}, inplace=True)

    # Save results
    resultats_grouped.to_csv(chemin_resultats_csv)
    print(f"Résumés sauvegardés dans le fichier {chemin_resultats_csv}")
