from embeddings_creation import embed_texts, generer_embeddings_rapport
import torch
from transformers import pipeline, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize model and tokenizer with pipeline for text generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to avoid warning
)

# Define the prompt template for summarization
resume_prompt_template = """
**Tâche** : Fournir un résumé détaillé et structuré des faits scientifiques contenus dans la section du rapport du GIEC, en les reliant directement à la question posée. La réponse doit être sous forme de liste numérotée, avec chaque point citant précisément les données chiffrées ou informations textuelles pertinentes.

**Instructions** :
1. Inclure des faits scientifiques directement en rapport avec la question.
2. Intégrer des données chiffrées et précises.
3. Ne pas inclure d'opinions ou d'interprétations.

### Question :
"{question}"

### Sections du rapport :
{retrieved_sections}

**Format de réponse attendu** :
1. Le niveau global de la mer a augmenté de 0,19 m entre 1901 et 2010.
2. Les températures moyennes ont augmenté de 1,09°C entre 1850-1900 et 2011-2020.
"""

# Function to generate the formatted summarization prompt
def generate_resume_prompt(question, retrieved_sections):
    return resume_prompt_template.format(question=question, retrieved_sections=retrieved_sections)

# Function to generate a summary using the LLM
def generate_resume_with_llm(question, section):
    prompt = generate_resume_prompt(question, section)
    output = pipe(prompt, max_new_tokens=100)  # Adjust max_new_tokens to optimize memory
    response_content = output[0]["generated_text"].strip()
    
    # Use prompt length to reliably extract only the generated response
    # Remove the prompt part from the start of the generated text
    response_only = response_content[len(prompt):].strip()
        
    print(f'Voici la réponse du LLM : {response_only}')
    return response_only

# Function to process summaries in parallel
def generer_resume_parallel(df_questions):
    results = []

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=2) as executor:  # Adjust max_workers based on system resources
        futures = {
            executor.submit(generate_resume_with_llm, row['question'], section): (row['id'], row['question'], section)
            for _, row in df_questions.iterrows()
            for section in row['retrieved_sections']
        }

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating summaries"):
            (question_id, question, section) = futures[future]

            try:
                # Get the generated summary
                resume_section = future.result()
                results.append({
                    "id": question_id,
                    "question": question,
                    "sections": section,
                    "resume_sections": resume_section
                })

            except Exception as exc:
                print(f"Error generating summary for question ID {question_id}: {exc}")

    return pd.DataFrame(results)

# Function to load data and embeddings
def charger_donnees_et_modele(chemin_csv_questions, chemin_rapport_embeddings):
    df_questions = pd.read_csv(chemin_csv_questions)
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    embeddings_rapport, sections_rapport, titles_rapport = generer_embeddings_rapport(chemin_rapport_embeddings, embed_model)
    return df_questions, embeddings_rapport, sections_rapport, titles_rapport, embed_model

# Function to filter relevant sections for each question
def filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k=5):
    retrieved_sections_list = []
    for _, row in df_questions.iterrows():
        question_embedding = embed_texts([row['question']], embed_model)[0]
        similarites = util.cos_sim(question_embedding, torch.tensor(embeddings_rapport, device='cpu'))
        top_k_indices = np.argsort(-similarites[0].cpu()).tolist()[:top_k]
        sections = [sections_rapport[i] for i in top_k_indices if sections_rapport[i].strip()]
        retrieved_sections_list.append(sections)
    df_questions['retrieved_sections'] = retrieved_sections_list
    return df_questions

# Main process function
def process_resume(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, top_k=3):
    df_questions, embeddings_rapport, sections_rapport, _, embed_model = charger_donnees_et_modele(chemin_csv_questions, chemin_rapport_embeddings)
    df_questions = filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k)
    resultats = generer_resume_parallel(df_questions)

    # Group by question ID to concatenate sections and summaries
    resultats_grouped = resultats.groupby('id').agg({
        'question': 'first',
        'sections': lambda x: ' '.join(x),
        'resume_sections': lambda x: ' '.join(x)
    }).reset_index()

    # Save results to CSV
    resultats_grouped.to_csv(chemin_resultats_csv, index=False)
    print(f"Résumés sauvegardés dans le fichier {chemin_resultats_csv}")