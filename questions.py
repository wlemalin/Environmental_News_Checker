import torch
from transformers import pipeline, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

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

# Define the prompt template for generating questions
question_prompt_template = """"
Vous êtes chargé de formuler une **question précise** pour vérifier les informations mentionnées dans un extrait spécifique d'un article de presse en consultant directement les rapports du GIEC (Groupe d'experts intergouvernemental sur l'évolution du climat).

Cette question sera utilisée dans un système de récupération d'information (RAG) pour extraire les sections pertinentes des rapports du GIEC et comparer les informations des rapports avec celles de l'article de presse.

**Objectif** : La question doit permettre de vérifier si les informations fournies dans la phrase de l'article sont corroborées ou contestées par les preuves scientifiques dans les rapports du GIEC. La question doit donc englober tous les sujets abordés par l'extrait de l'article de presse.

**Instructions** :

1. Analysez l'extrait et son contexte pour identifier les affirmations clées et/ou les informations à vérifier.
2. Formulez une **question claire et spécifique** orientée vers la vérification de ces affirmations ou informations à partir des rapports du GIEC. La question doit permettre de vérifier toutes les informations de l'extraits. La question peut être un ensemble de questions comme : "Quel est l'impact des activités humaines sur le taux de CO2 dans l'atomsphère ? Comment la concentration du CO2 dans l'atmosphère impact l'argiculture?"
3. La question doit être **directement vérifiable** dans les rapports du GIEC via un système RAG.
4. **IMPORTANT** : Répondez uniquement avec la question, sans ajouter d'explications ou de contexte supplémentaire.

Extrait de l'article de presse : {current_phrase}

Contexte : {context}

Générez uniquement la **question** spécifique qui permettrait de vérifier les informations mentionnées dans cette phrase en consultant les rapports du GIEC via un système de récupération d'information (RAG).
"""

# Function to generate the formatted question prompt
def generate_question_prompt(current_phrase, context):
    return question_prompt_template.format(current_phrase=current_phrase, context=context)

# Function to generate a question using the LLM
def generate_question_with_llm(current_phrase, context):
    prompt = generate_question_prompt(current_phrase, context)
    output = pipe(prompt, max_new_tokens=100)  # Adjust max_new_tokens to optimize memory
    response_content = output[0]["generated_text"]
    
    response_content = output[0]["generated_text"].strip()

    # Use prompt length to reliably extract only the generated response
    # Remove the prompt part from the start of the generated text
    response_only = response_content[len(prompt):].strip()
        
    print(f'Voici la réponse du LLM : {response_only}')
    return response_only


# Function to process questions in parallel
def generate_questions_parallel(df):
    results = []

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=1) as executor:  # Adjust max_workers based on system resources
        futures = {
            executor.submit(generate_question_with_llm, row['current_phrase'], row['context']): row
            for _, row in df.iterrows()
        }

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating questions"):
            row = futures[future]
            current_phrase = row['current_phrase']
            context = row['context']

            try:
                # Get the generated question
                question = future.result()
                row['question'] = question
                results.append(row)

                # Print the generated question for debugging purposes
                print(f"Phrase:\n{current_phrase}\nContext:\n{context}\nGenerated Question:\n{question}\n")

            except Exception as exc:
                print(f"Error generating question for phrase: {current_phrase} - {exc}")

    return pd.DataFrame(results)

# Example main function to load data, generate questions, and save results
def question_generation_process(file_path, output_path):
    df = pd.read_csv(file_path)

    # Convertir la colonne 'binary_response' en texte (si elle est en format texte)
    df['binary_response'] = df['binary_response'].astype(str)

    # Filtrer uniquement les phrases identifiées comme liées à l'environnement (réponse binaire '1')
    #df_environment = df[df['binary_response'] == '1']

    # Generate questions in parallel and save results
    questions_df = generate_questions_parallel(df)
    questions_df.to_csv(output_path, index=False)
    print(f"Questions saved to {output_path}")
