from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from langchain import LLMChain, PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from embeddings_creation import (charger_embeddings_rapport, embed_texts,
                                 generer_embeddings_rapport)
from file_utils import charger_questions, sauvegarder_mentions_csv


# Fonction pour générer des réponses avec Llama3.2 et les sections du rapport
def rag_answer_generation_with_llmchain(question, relevant_sections, llm_chain):
    context = " ".join(relevant_sections)
    inputs = {
        "question": question,
        "consolidated_text": context
    }
    response = llm_chain.invoke(inputs)
    generated_answer = response['text'] if isinstance(
        response, dict) and "text" in response else response
    return generated_answer.strip()


# Comparer les questions aux sections du rapport via les embeddings
def comparer_questions_rapport(questions, embeddings_rapport, sections_rapport, titles_rapport, llm_chain, embed_model, top_k=3):
    results = []

    # Utilisation d'un pool de threads pour paralléliser le traitement
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Comparing questions"):
            ID = row['id']
            question = row['question']
            # Utilisation de 'current_phrase' au lieu de 'paragraph'
            current_phrase = row['current_phrase']
            futures.append(executor.submit(trouver_sections_et_generer_reponse, question, current_phrase,
                           embeddings_rapport, sections_rapport, titles_rapport, llm_chain, embed_model, top_k, ID))

        # Récupérer les résultats
        for future in tqdm(futures, desc="Retrieving answers"):
            try:
                question, current_phrase, retrieved_sections, generated_answer, ID = future.result()
                results.append({
                    "id": ID,
                    "current_phrase": current_phrase,  # Inclure la phrase dans les résultats
                    "question": question,
                    "retrieved_sections": retrieved_sections,
                    "generated_answer": generated_answer
                })
            except Exception as exc:
                print(f"Error during RAG: {exc}")

    return results

# Fonction pour trouver les sections pertinentes et générer une réponse


def trouver_sections_et_generer_reponse(question, current_phrase, embeddings_rapport, sections_rapport, titles_rapport, llm_chain, embed_model, top_k, ID):
    question_embedding = embed_texts([question], embed_model)[0]

    # Convert embeddings to torch tensor on CPU
    similarites = util.cos_sim(question_embedding, torch.tensor(
        embeddings_rapport, device='cpu'))  # Ensure on CPU

    top_k_indices = np.argsort(-similarites[0])[:top_k]
    top_k_sections = [
        f"{titles_rapport[j]}: {sections_rapport[j]}" for j in top_k_indices]

    generated_answer = rag_answer_generation_with_llmchain(
        question, top_k_sections, llm_chain)

    return question, current_phrase, " ".join(top_k_sections), generated_answer, ID

# Main function to execute the RAG process


def rag_process(chemin_questions_csv, chemin_rapport_embeddings, chemin_resultats_csv):
    questions_df = charger_questions(chemin_questions_csv)

    embed_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2', device='cpu')  # Set the model to CPU

    # Générer les embeddings si nécessaire (si 'embedding' n'existe pas dans le fichier JSON)
    generer_embeddings_rapport(chemin_rapport_embeddings, embed_model)

    embeddings_rapport, sections_rapport, titles_rapport = charger_embeddings_rapport(
        chemin_rapport_embeddings)

    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")

    prompt_template = """
    Vous êtes un expert en climatologie et votre rôle est d'analyser les informations provenant d'un article de presse sur le changement climatique. Votre tâche consiste à répondre à la question ci-dessous en utilisant uniquement les sections pertinentes du rapport du GIEC fournies.

    **Instructions** :
    1. Lisez attentivement la question posée par l'article.
    2. Utilisez les informations pertinentes des sections du rapport du GIEC pour formuler une réponse précise et fondée.
    3. Justifiez votre réponse en citant les sections du rapport du GIEC, si nécessaire.
    4. Limitez votre réponse aux informations présentes dans les sections fournies.
    
    **Question de l'article** : {question}
    
    **Sections pertinentes du rapport du GIEC** : {consolidated_text}
    
    **Réponse** :
    - **Résumé de la réponse** : (Donnez une réponse concise à la question)
    - **Justification basée sur le rapport du GIEC** : (Citez et expliquez les éléments pertinents du rapport)
    
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "question", "consolidated_text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    mentions = comparer_questions_rapport(
        questions_df, embeddings_rapport, sections_rapport, titles_rapport, llm_chain, embed_model)

    sauvegarder_mentions_csv(mentions, chemin_resultats_csv)
