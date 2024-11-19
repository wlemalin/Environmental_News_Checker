import torch
from transformers import pipeline, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import nltk
from nltk import sent_tokenize
from llms import parsed_responses
from generate_context_windows import generate_context_windows

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

# Define the prompt for relevant phrase selection
prompt_template = """
Vous êtes un expert chargé d'identifier tous les sujets abordés dans le texte suivant, qu'ils soient ou non liés à l'environnement, au changement climatique ou au réchauffement climatique.

Phrase : {current_phrase}
context : {context}

1. Si le texte mentionne de près ou de loin l'environnement, le changement climatique, le réchauffement climatique, ou des organisations, événements ou accords liés à ces sujets (par exemple le GIEC, les conférences COP, les accords de Paris, etc.), répondez '1'. Sinon, répondez '0'.
2. Listez **tous** les sujets abordés dans le texte, y compris ceux qui ne sont pas liés à l'environnement ou au climat.
3. N'incluez aucune phrase introduction ou autre, seulement la réponse dans le format attendu.

Format de réponse attendu :
- Réponse binaire (0 ou 1) : [Réponse]
- Liste des sujets abordés : [Sujet 1, Sujet 2, ...]

Exemple de réponse :
- Réponse binaire (0 ou 1) : 1
- Liste des sujets abordés : [Incendies, gestion des forêts, réchauffement climatique, économie locale, GIEC]
"""

# Function to generate the formatted prompt
def generate_prompt(current_phrase, context):
    return prompt_template.format(current_phrase=current_phrase, context=context)

# Function to analyze a paragraph with the LLM
def analyze_paragraph_with_llm(current_phrase, context):
    prompt = generate_prompt(current_phrase, context)
    print(prompt)
    output = pipe(prompt, max_new_tokens=300)  # Adjust max_new_tokens to optimize memory
    response_content = output[0]["generated_text"]
    
    # Use prompt length to reliably extract only the generated response
    # Remove the prompt part from the start of the generated text
    response_only = response_content[len(prompt):].strip()
        
    print(f'Voici la réponse du LLM : {response_only}')
    return response_only


# Function to manage parallel analysis of paragraphs
def analyze_paragraphs_parallel(splitted_text):
    results = []

    # Using ThreadPoolExecutor with reduced max_workers to optimize memory usage
    with ThreadPoolExecutor(max_workers=1) as executor:  # Adjust max_workers based on available memory
        futures = {
            executor.submit(analyze_paragraph_with_llm, entry["current_phrase"], entry["context"]): entry
            for entry in splitted_text
        }

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing paragraphs"):
            entry = futures[future]
            current_phrase = entry["current_phrase"]
            context = entry["context"]
            index = entry["id"]

            try:
                # Get the result of the analysis
                analysis = future.result()

                # Append only the necessary information to results to save memory
                results.append({
                    "id": index,
                    "current_phrase": current_phrase,
                    "context": context,
                    "climate_related": analysis
                })

                # Print results to avoid storing too much data in memory
                print(
                    f"ID: {index}\nPhrase:\n{current_phrase}\nContext:\n{context}\nLLM Response: {analysis}\n")

            except Exception as exc:
                print(f"Error analyzing phrase ID {index}: {current_phrase} - {exc}")

    return results



# Main function to identify GIEC extracts
def identifier_extraits_sur_giec(file_path, output_path, output_path_improved):
    nltk.download('punkt')  # Download sentence tokenizer model

    # Load and split text into sentences
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    sentences = sent_tokenize(text)  # Divise le texte en phrases
    splitted_text = generate_context_windows(sentences)
    
    # Analyze paragraphs with Llama 3.2 in parallel
    analysis_results = analyze_paragraphs_parallel(splitted_text)

    # Save results to a CSV file immediately
    df = pd.DataFrame(analysis_results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Apply parsing to DataFrame to extract subjects
    parsed_df_improved = parsed_responses(df)

    # Stream results to CSV to avoid keeping large dataframes in memory
    parsed_df_improved['subjects'] = parsed_df_improved['subjects'].apply(lambda x: ', '.join(x))
    parsed_df_improved.to_csv(output_path_improved, index=False)
    print(f"Improved results saved to {output_path_improved}")

    # Display a few lines of the final DataFrame
    print(parsed_df_improved.head())