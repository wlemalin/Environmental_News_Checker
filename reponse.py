from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer

# Initialize model and tokenizer with Hugging Face's pipeline for text generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to avoid warning
)

# Define the prompt template for generating answers
answer_prompt_template = """
Vous êtes un expert en climatologie. Répondez à la question ci-dessous en vous basant uniquement sur les sections pertinentes du rapport du GIEC.

**Instructions** :
1. Utilisez les informations des sections pour formuler une réponse précise et fondée.
2. Justifiez votre réponse en citant les sections, si nécessaire.
3. Limitez votre réponse aux informations fournies dans les sections.

**Question** : {question}

**Sections du rapport** : {consolidated_text}

**Réponse** :
- **Résumé de la réponse** : (Réponse concise)
- **Justification basée sur le rapport** : (Citez et expliquez les éléments pertinents)
"""

# Function to generate the formatted answer prompt
def generate_answer_prompt(question, consolidated_text):
    return answer_prompt_template.format(question=question, consolidated_text=consolidated_text)

# Function to generate an answer using the LLM
def generate_answer_with_llm(question, consolidated_text):
    prompt = generate_answer_prompt(question, consolidated_text)
    output = pipe(prompt, max_new_tokens=500)  # Adjust max_new_tokens as needed
    response_content = output[0]["generated_text"].strip()
    
    # Use prompt length to reliably extract only the generated response
    # Remove the prompt part from the start of the generated text
    response_only = response_content[len(prompt):].strip()
        
    print(f'Voici la réponse du LLM : {response_only}')
    return response_only



# Function to process answers in parallel
def answer_questions_parallel(df_questions):
    results = []

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=1) as executor:  # Adjust max_workers based on system resources
        futures = {
            executor.submit(generate_answer_with_llm, row['question'], row['resume_sections']): row
            for _, row in df_questions.iterrows()
        }

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating answers"):
            row = futures[future]
            question_id = row['id']
            question = row['question']
            resume_sections = row['resume_sections']
            sections_brutes = row['sections']

            try:
                # Get the generated answer
                generated_answer = future.result()
                results.append({
                    "id": question_id,
                    "question": question,
                    "sections_resumees": resume_sections,
                    "retrieved_sections": sections_brutes,
                    "reponse": generated_answer
                })

            except Exception as exc:
                print(f"Error generating answer for question ID {question_id}: {exc}")

    return pd.DataFrame(results)

# Main function to execute the RAG process
def process_reponses(chemin_questions_csv, chemin_resultats_csv):
    """
    Executes the retrieval-augmented generation (RAG) process to generate answers and save them to a CSV file.

    Args:
        chemin_questions_csv (str): Path to the CSV file containing questions.
        chemin_resultats_csv (str): Path to the output CSV file where results will be saved.

    Workflow:
        1. Load questions from the input CSV file into a pandas DataFrame.
        2. Process the questions in parallel to generate answers using the LLM.
        3. Save the generated answers to the output CSV file.
        4. Print a confirmation message indicating where the results have been saved.
    """
    questions_df = pd.read_csv(chemin_questions_csv)

    # Generate answers and save them to CSV
    answers_df = answer_questions_parallel(questions_df)
    answers_df.to_csv(chemin_resultats_csv, index=False, quotechar='"')
    print(f"Answers saved to file {chemin_resultats_csv}")