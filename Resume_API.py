import os
import replicate
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
import tqdm
from embeddings_creation import generer_embeddings_rapport, embed_texts

# Save summary results
def sauvegarder_resultats_resume(resultats, chemin_resultats_csv):
    resultats.to_csv(chemin_resultats_csv, index=False, quotechar='"')
    print(f"Résumés sauvegardés dans le fichier {chemin_resultats_csv}")

# Load data and model for questions and report sections
def charger_donnees_et_modele(chemin_csv_questions, chemin_rapport_embeddings, top_k=5):
    df_questions = pd.read_csv(chemin_csv_questions)
    print(f"Questions loaded. Total: {len(df_questions)}")
    
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    embeddings_rapport, sections_rapport, titles_rapport = generer_embeddings_rapport(chemin_rapport_embeddings, embed_model)
    print("Report embeddings and sections loaded.")
    
    if not (len(embeddings_rapport) == len(sections_rapport) == len(titles_rapport)):
        print("Erreur : Le nombre d'embeddings, de sections, et de titres ne correspond pas.")
    else:
        print(f"Données chargées avec succès : {len(embeddings_rapport)} embeddings, {len(sections_rapport)} sections, et {len(titles_rapport)} titres.")
    
    return df_questions, embeddings_rapport, sections_rapport, titles_rapport, embed_model

# Filter relevant sections based on similarity
def filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k=5):
    retrieved_sections_list = []
    
    for idx, row in df_questions.iterrows():
        question_embedding = embed_texts([row['question']], embed_model)[0]
        similarites = util.cos_sim(question_embedding, torch.tensor(embeddings_rapport, device='cpu'))
        top_k_indices = np.argsort(-similarites[0].cpu()).tolist()[:top_k]
        sections = [sections_rapport[i] for i in top_k_indices if sections_rapport[i].strip()]
        
        # Concatenate all top_k sections into one text
        concatenated_sections = " ".join(sections)
        retrieved_sections_list.append(concatenated_sections)
    
    df_questions['retrieved_sections'] = retrieved_sections_list
    return df_questions

# Create the summary prompt template
def creer_prompt_resume(question, retrieved_sections):
    prompt_template_resume = """
    **Tâche** : Fournir un résumé détaillé et structuré des faits scientifiques contenus dans la section du rapport du GIEC, en les reliant directement à la question posée. La réponse doit être sous forme de liste numérotée, avec chaque point citant précisément les données chiffrées ou informations textuelles pertinentes.

    **Instructions** :
    - **Objectif** : Présenter une liste complète des faits pertinents, incluant les éléments directement en lien avec la question, ainsi que des informations contextuelles importantes qui peuvent enrichir la compréhension du sujet.
    - **Directives spécifiques** :
        1. Inclure des faits scientifiques directement en rapport avec la question, en les citant de manière précise.
        2. Intégrer des données chiffrées, tendances, et statistiques spécifiques lorsque disponibles, en veillant à la clarté et la précision de la citation.
        3. Fournir des éléments de contexte pertinents qui peuvent éclairer la réponse, sans extrapoler ou interpréter.
        4. Utiliser un langage concis mais précis, en mentionnant chaque fait pertinent dans une ou deux phrases.
        5. Être exhaustif dans les faits listés pouvant être intéressant, directement ou indirectement pou répondre à la question.
    - **Restrictions** : 
        - Ne pas inclure d'opinions ou de généralisations.
        - Ne reformuler les informations que si cela permet de les rendre plus compréhensibles sans en altérer le sens.
        - Ne présenter que des faits, sans ajout de suppositions ou interprétations.
        - Ne pas ajouter de phrase introductrice comme 'Voici les fais retranscris' ou autre.
    - **Format de réponse** : Utiliser une liste numérotée, en commençant par les faits les plus directement liés à la question, suivis par les éléments de contexte. Chaque point doit être limité à une ou deux phrases.

    ### Question :
    "{question}"

    ### Sections du rapport :
    {retrieved_sections}

    **Exemple de réponse attendue** :
        1. Le niveau global de la mer a augmenté de 0,19 m entre 1901 et 2010, en lien direct avec la hausse des températures mondiales.
        2. Les températures moyennes ont augmenté de 1,09°C entre 1850-1900 et 2011-2020, influençant la fréquence des événements climatiques extrêmes.
        3. Les concentrations de CO2 dans l'atmosphère ont atteint 410 ppm en 2019, une donnée clé pour comprendre l'accélération du réchauffement climatique.

    **Remarque** : Respecter strictement les consignes et ne présenter que des faits sous forme de liste numérotée. Citer toutes les données chiffrées ou textuelles de manière exacte pour assurer la rigueur de la réponse.
    """
    return prompt_template_resume.format(question=question, retrieved_sections=retrieved_sections)

# Function to call Replicate API for generating summaries
def appeler_replicate_summarization(prompt_text):
    input_payload = {
        "prompt": prompt_text,
        "max_tokens": 1500
    }
    try:
        output = replicate.run("meta/meta-llama-3-70b-instruct", input=input_payload)
        return "".join(output)  # Join the response segments into a single text
    except Exception as e:
        print(f"Erreur lors de l'appel à Replicate : {e}")
        return "Erreur de l'API Replicate"

# Generate summary for a specific question and concatenated sections
def generer_resume(phrase_id, question, concatenated_sections):
    prompt_text = creer_prompt_resume(question, concatenated_sections)
    response_resume = appeler_replicate_summarization(prompt_text)
    return {
        "id": phrase_id,
        "question": question,
        "sections": concatenated_sections,
        "resume_sections": response_resume
    }

# Parallel summary generation for questions with concatenated sections
def generer_resume_parallel(df_questions):
    resultats = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for _, row in df_questions.iterrows():
            question_id = row['id']
            question_text = row['question']
            concatenated_sections = row['retrieved_sections']
            
            futures.append(executor.submit(
                generer_resume, 
                question_id, 
                question_text, 
                concatenated_sections
            ))
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Résumés des sections"):
            try:
                result = future.result()
                resultats.append(result)
            except Exception as exc:
                print(f"Erreur lors du résumé d'une question : {exc}")
    
    return pd.DataFrame(resultats)

# Main function to process and save summaries
def process_resume_api(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, top_k=3):
    # Set up Replicate API key
    os.environ["REPLICATE_API_TOKEN"] = "r8_KVdlDIHTh9T6xEuEJhDkNxvfCXleqe814zH72"
    replicate.api_token = os.getenv("REPLICATE_API_TOKEN")
    df_questions, embeddings_rapport, sections_rapport, _, embed_model = charger_donnees_et_modele(
        chemin_csv_questions, chemin_rapport_embeddings, top_k)
    
    df_questions = filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k)
    
    resultats = generer_resume_parallel(df_questions)
    
    resultats_grouped = resultats.groupby('id').agg({
        'question': 'first',
        'sections': 'first',  # Already concatenated, no need to join
        'resume_sections': 'first'  # Take the single summarized output
    }).reset_index()
    
    sauvegarder_resultats_resume(resultats_grouped, chemin_resultats_csv)
    print(f"Résumés sauvegardés dans le fichier {chemin_resultats_csv}")