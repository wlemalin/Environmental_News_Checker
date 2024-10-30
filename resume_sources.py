from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from langchain import LLMChain, PromptTemplate
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from embeddings_creation import generer_embeddings_rapport, embed_texts
from file_utils import sauvegarder_resultats_resume
from llms import creer_prompt_resume
from langchain.llms import Ollama


def charger_donnees_et_modele(chemin_csv_questions, chemin_rapport_embeddings, top_k=5):
    # Charger les questions
    df_questions = pd.read_csv(chemin_csv_questions)
    print(f"Questions loaded. Total: {len(df_questions)}")

    # Configurer le modèle d'embedding et charger les embeddings et sections du rapport
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    embeddings_rapport, sections_rapport, titles_rapport = generer_embeddings_rapport(chemin_rapport_embeddings, embed_model)
    print("Report embeddings and sections loaded.")

    # Vérifier la cohérence des données
    if not (len(embeddings_rapport) == len(sections_rapport) == len(titles_rapport)):
        print("Erreur : Le nombre d'embeddings, de sections, et de titres ne correspond pas.")
    else:
        print(f"Données chargées avec succès : {len(embeddings_rapport)} embeddings, {len(sections_rapport)} sections, et {len(titles_rapport)} titres.")

    # Afficher quelques exemples pour vérification
    print("Exemples de sections et leurs embeddings :")
    for i in range(min(3, len(sections_rapport))):
        print(f"Titre : {titles_rapport[i]}")
        print(f"Texte : {sections_rapport[i][:100]}...")  # Limiter l'affichage à 100 caractères
        print(f"Embedding (taille) : {len(embeddings_rapport[i])}\n")

    return df_questions, embeddings_rapport, sections_rapport, titles_rapport, embed_model



def filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k=5):
    retrieved_sections_list = []
    
    for idx, row in df_questions.iterrows():
        question_embedding = embed_texts([row['question']], embed_model)[0]
        
        # Calculer les similarités et sélectionner les top-k indices
        similarites = util.cos_sim(question_embedding, torch.tensor(embeddings_rapport, device='cpu'))
        top_k_indices = np.argsort(-similarites[0].cpu()).tolist()[:top_k]
        
        # Récupérer les sections correspondantes
        sections = [sections_rapport[i] for i in top_k_indices if sections_rapport[i].strip()]
        retrieved_sections_list.append(sections)

    # Ajouter les sections retrouvées comme nouvelle colonne
    df_questions['retrieved_sections'] = retrieved_sections_list
    return df_questions


def creer_llm_resume():
    """
    Crée et configure la chaîne LLM pour le résumé.
    """
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    prompt_template_resume = """
    Votre tâche est d'extraire et de **lister uniquement** les faits les plus pertinents contenus dans les sections du rapport du GIEC en rapport avec la question posée. 
    **Question posée** : {question}
    **Sections associées** : {retrieved_sections}

    Réponse sous forme de liste numérotée :
    1. ...
    2. ...
    """
    return LLMChain(prompt=PromptTemplate(template=prompt_template_resume, input_variables=["question", "retrieved_sections"]), llm=llm)



def generer_resume_parallel(df_questions, llm_chain_resume):
    """
    Pour chaque question, résume les sections associées en utilisant le modèle LLM.
    """
    resultats = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for _, row in df_questions.iterrows():
            futures.append(executor.submit(
                generer_resume,
                row['id'], row['question'], row['retrieved_sections'], llm_chain_resume
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Résumés des sections"):
            try:
                result = future.result()
                resultats.append(result)
            except Exception as exc:
                print(f"Erreur lors du résumé d'une question : {exc}")

    return pd.DataFrame(resultats)




def generer_resume(phrase_id, question, retrieved_sections, llm_chain_resume):
    """
    Generates the summary for each relevant section of a question using the LLM.
    """
    # Lists to store the summaries and original sections
    resumes = []
    sections = []

    for section in retrieved_sections:
        # Generate the summary for each individual section
        response_resume = llm_chain_resume.invoke({
            "question": question,
            "retrieved_sections": section
        })

        resume = response_resume['text'].strip() if isinstance(
            response_resume, dict) and "text" in response_resume else response_resume.strip()

            

        # Append each summary and original section to respective lists
        resumes.append(resume)
        sections.append(section)

    # Return the results as a dictionary with ID, question, original sections, and list of section summaries
    return {
        "id": phrase_id,               # Add ID to the results
        "question": question,           # Add question to the results
        "sections": sections,           # List of original sections
        "resume_sections": resumes      # List of summaries for each relevant section
    }




def process_resume(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, top_k=3):
    # Charger les données et le modèle
    df_questions, embeddings_rapport, sections_rapport, titles_rapport, embed_model = charger_donnees_et_modele(
        chemin_csv_questions, chemin_rapport_embeddings, top_k)
    
    # Filtrer les sections pertinentes et obtenir un DataFrame avec retrieved_sections
    df_questions = filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k)
    
    # Configurer la LLM pour le résumé
    prompt_llm_chain_resume = creer_prompt_resume()
    
    # Initialize the LLM (Ollama)
    llm = Ollama(model="llama3.2:3b-instruct-fp16")

    # Create the LLM chain
    llm_chain_resume = LLMChain(prompt=prompt_llm_chain_resume, llm=llm)

    # Résumer les sections par question
    resultats = generer_resume_parallel(df_questions, llm_chain_resume)

    # Sauvegarder les résultats
    sauvegarder_resultats_resume(resultats, chemin_resultats_csv)
    print(f"Résumés sauvegardés dans le fichier {chemin_resultats_csv}")
    
    
    
    