import os
import replicate
import concurrent.futures
import pandas as pd
from tqdm import tqdm
import nltk
from nltk import sent_tokenize
from llms import parsed_responses
from topic_classifier import generate_context_windows

# Explicitly set the Replicate API key
replicate.api_token = "r8_KVdlDIHTh9T6xEuEJhDkNxvfCXleqe814zH72"

# Function to create the prompt and call Replicate's API
def generate_response_with_replicate(current_phrase, context):
    prompt_template = """
    Vous êtes un expert chargé d'identifier tous les sujets abordés dans le texte suivant, qu'ils soient ou non liés à l'environnement, au changement climatique ou au réchauffement climatique.
    
    Phrase : {current_phrase}
    context : {context}
    
    1. Si le texte mentionne de près ou de loin l'environnement, le changement climatique, le réchauffement climatique, ou des organisations, événements ou accords liés à ces sujets (par exemple le GIEC, les conférences COP, les accords de Paris, etc.), répondez '1'. Sinon, répondez '0'.
    2. Listez **tous** les sujets abordés dans le texte, y compris ceux qui ne sont pas liés à l'environnement ou au climat.
    
    Format de réponse attendu :
    - Réponse binaire (0 ou 1) : [Réponse]
    - Liste des sujets abordés : [Sujet 1, Sujet 2, ...]
    
    Exemple de réponse :
    - Réponse binaire (0 ou 1) : 1
    - Liste des sujets abordés : [Incendies, gestion des forêts, réchauffement climatique, économie locale, GIEC]
    """
    prompt_text = prompt_template.format(
        current_phrase=current_phrase,
        context=context
    )

    # Call Replicate API
    try:
        output = replicate.run("meta/meta-llama-3.1-405b-instruct",
                               input={"prompt": prompt_text, "max_tokens": 500})
        return "".join(output).strip()
    except Exception as e:
        print(f"Error during API call to Replicate: {e}")
        return "Erreur de l'API Replicate"

# Function to analyze paragraphs in parallel using Replicate API
def analyze_paragraphs_parallel(splitted_text):
    results = []

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
        # Create a task for each entry in splitted_text (each phrase with its context and index)
        futures = {
            executor.submit(generate_response_with_replicate, entry["current_phrase"], entry["context"]): entry
            for entry in splitted_text
        }

        # Iterate through the results as they are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Analyzing paragraphs"):
            entry = futures[future]
            current_phrase = entry["current_phrase"]
            context = entry["context"]
            index = entry["id"]

            try:
                # Get the result from the API
                analysis = future.result()

                # Store the index, phrase, context, and API response in the result
                results.append({
                    "id": index,
                    "current_phrase": current_phrase,
                    "context": context,
                    "climate_related": analysis
                })

                # Print after each analysis
                print(f"ID: {index}\nPhrase:\n{current_phrase}\nContext:\n{context}\nReplicate API Response: {analysis}\n")

            except Exception as exc:
                print(f"Error analyzing phrase ID {index}: {current_phrase} - {exc}")

    return results

# Main function to identify excerpts related to IPCC
def identifier_extraits_sur_giec_api(file_path, output_path, output_path_improved):
    nltk.download('punkt')  # Download the sentence tokenizer model
    
    # Load and split text into sentences
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = sent_tokenize(text)  # Split text into sentences
    splitted_text = generate_context_windows(sentences)

    # Analyze paragraphs using Replicate API in parallel
    analysis_results = analyze_paragraphs_parallel(splitted_text)

    # Save the results to a CSV file
    df = pd.DataFrame(analysis_results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Apply the parsing method to the DataFrame
    parsed_df_improved = parsed_responses(df)

    # Save the parsed results DataFrame
    parsed_df_improved['subjects'] = parsed_df_improved['subjects'].apply(
        lambda x: ', '.join(x))
    parsed_df_improved.to_csv(output_path_improved, index=False)

    # Display a few rows of the final DataFrame
    print(parsed_df_improved.head())