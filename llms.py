from concurrent.futures import ThreadPoolExecutor  # For parallelization
import numpy as np
import tqdm
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from sentence_transformers import util
from tqdm import tqdm  # For the progress bar


def configure_embeddings(use_ollama=True):
    if use_ollama:
        Settings.embed_model = OllamaEmbedding(
            model_name="llama3.2:3b-instruct-fp16")
    else:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")


# Function to generate answers using LLMChain
def rag_answer_generation_with_llmchain(question, relevant_sections, llm_chain):
    # Combine the relevant sections into a single context
    context = " ".join(relevant_sections)

    # Prepare inputs for the LLMChain
    inputs = {
        "question": question,
        "consolidated_text": context
    }

    # Generate the answer using LLMChain
    response = llm_chain.invoke(inputs)  # Updated from __call__ to invoke

    # Extract the generated answer from the response
    generated_answer = response['text'] if isinstance(
        response, dict) and "text" in response else response
    return generated_answer.strip()


def comparer_article_rapport_with_rag(phrases_article, embeddings_rapport, sections_rapport, llm_chain, seuil_similarite=0.5, top_k=3):
    print("Comparaison des phrases de l'article avec les sections du rapport...")
    mentions = []

    # Encode article sentences with progress bar and parallelization
    embeddings_phrases = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        # Parallelizing embedding generation
        futures = [executor.submit(
            Settings.embed_model.get_text_embedding, phrase) for phrase in phrases_article]

        # Using tqdm to display the progress
        for future in tqdm(futures, desc="Generating Embeddings", total=len(futures)):
            embedding = future.result()
            # Ensure the embedding is not empty
            if embedding is not None and len(embedding) > 0:
                embeddings_phrases.append(embedding)
            else:
                # Add a zero embedding as fallback to maintain shape
                embeddings_phrases.append(np.zeros(3072))

    # Loop through each sentence in the article with progress bar
    for i, phrase_embedding in tqdm(enumerate(embeddings_phrases), total=len(embeddings_phrases), desc="Comparing Phrases"):
        if len(phrase_embedding) == 0:  # Skip empty embeddings
            continue

        similarites = util.cos_sim(phrase_embedding, embeddings_rapport)
        # Get top-k closest sections from the report
        top_k_indices = np.argsort(-similarites[0])[:top_k]
        top_k_sections = [sections_rapport[j] for j in top_k_indices]

        # Use RAG (Retrieve-then-Generate) with LLMChain
        question = phrases_article[i]
        generated_answer = rag_answer_generation_with_llmchain(
            question, top_k_sections, llm_chain)

        # Store results
        mentions.append({
            "phrase": question,
            "retrieved_sections": " ".join(top_k_sections),
            "generated_answer": generated_answer
        })

    print(f"{len(mentions)} mentions trouvées.")
    return mentions
