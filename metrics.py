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
    torch_dtype=torch.bfloat16,  # Use mixed-precision for efficiency
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id  # Avoid warnings
)

# Définir les prompts pour chaque métrique
prompts = {
    "accuracy": """
        Vous êtes chargé de comparer un extrait d'un article de presse aux informations officielles du rapport du GIEC.
        Évaluez l'exactitude de cet extrait en fonction des sections du rapport fournies. Utilisez l'échelle suivante :

        - **Score 0** : Totalement inexact, contredit directement les informations du rapport.
        - **Score 1** : Très inexact, contient plusieurs erreurs ou omissions importantes par rapport au rapport.
        - **Score 2** : Partiellement inexact, quelques erreurs importantes, mais certaines informations sont correctes.
        - **Score 3** : Modérément exact, contient des informations correctes mais avec quelques imprécisions.
        - **Score 4** : Largement exact, avec seulement de légères imprécisions.
        - **Score 5** : Parfaitement exact, correspond pleinement aux informations du rapport.

        **Extrait de l'article** :
        "{current_phrase}"

        **Sections du rapport du GIEC** :
        "{sections_resumees}"

        Répondez en JSON:
        {{
            "score": <score_entre_0_et_5>,
            "justifications": "<votre_justification>"
        }}
    """,
    
    "bias": """
        Vous êtes chargé d'analyser un extrait d'un article de presse pour détecter tout biais potentiel en fonction des sections du rapport du GIEC. Utilisez l'échelle suivante :

        - **Score 0** : Fortement biaisé, présente une vision totalement exagérée ou minimisée.
        - **Score 1** : Biaisé, avec une inclinaison évidente, soit en exagérant soit en minimisant les faits.
        - **Score 2** : Modérément biaisé, certains aspects exagérés ou minimisés mais dans l'ensemble équilibré.
        - **Score 3** : Légèrement biaisé, de petites nuances de biais mais globalement équilibré.
        - **Score 4** : Largement neutre, avec très peu de biais.
        - **Score 5** : Totalement neutre, sans aucun biais détectable.

        **Extrait de l'article** :
        "{current_phrase}"

        **Sections du rapport du GIEC** :
        "{sections_resumees}"

        Répondez en JSON:
        {{
            "score": <score_entre_0_et_5>,
            "justifications": "<votre_justification>"
        }}
    """,
    
    "tone": """
        Vous êtes chargé d'analyser le ton d'un extrait d'un article de presse en le comparant aux informations du rapport du GIEC. Utilisez l'échelle suivante :

        - **Score 0** : Ton fortement alarmiste ou minimisant, très éloigné du ton neutre.
        - **Score 1** : Ton exagérément alarmiste ou minimisant.
        - **Score 2** : Ton quelque peu alarmiste ou minimisant.
        - **Score 3** : Ton modérément factuel avec une légère tendance à l'alarmisme ou à la minimisation.
        - **Score 4** : Ton largement factuel, presque totalement neutre.
        - **Score 5** : Ton complètement neutre et factuel, sans tendance perceptible.

        **Extrait de l'article** :
        "{current_phrase}"

        **Sections du rapport du GIEC** :
        "{sections_resumees}"

        Répondez en JSON:
        {{
            "score": <score_entre_0_et_5>,
            "justifications": "<votre_justification>"
        }}
    """
}

# Generate prompt for a specific metric
def generate_metric_prompt(metric, current_phrase, sections_resumees):
    return prompts[metric].format(current_phrase=current_phrase, sections_resumees=sections_resumees)

# Function to evaluate a single metric using the LLM
def evaluate_metric_with_llm(metric, current_phrase, sections_resumees):
    prompt = generate_metric_prompt(metric, current_phrase, sections_resumees)
    response_content = pipe(prompt, max_new_tokens=500)[0]["generated_text"].strip()
    
    # Use prompt length to reliably extract only the generated response
    # Remove the prompt part from the start of the generated text
    response_only = response_content[len(prompt):].strip()
        
    print(f'Voici la réponse du LLM : {response_only}')
    return response_only

# Evaluate all metrics for a phrase
def evaluer_phrase_sur_toutes_metrices(phrase_id, question, current_phrase, sections_resumees):
    evaluations = {}
    for metric in prompts.keys():
        try:
            evaluations[metric] = evaluate_metric_with_llm(metric, current_phrase, sections_resumees)
        except Exception as exc:
            print(f"Error evaluating metric '{metric}' for phrase ID {phrase_id}: {exc}")
            evaluations[metric] = None
    
    return {
        "id": phrase_id,
        "question": question,
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees,
        **evaluations
    }

# Parallel evaluation for all phrases and metrics
def evaluer_phrase_parallele(rag_df):
    results = []

    # Set max_workers to utilize hardware capacity (adjust based on resources)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(
                evaluer_phrase_sur_toutes_metrices,
                row['id'], row['question'], row['current_phrase'], row['sections_resumees']
            )
            for _, row in rag_df.iterrows()
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating phrases for all metrics"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Error processing phrase: {exc}")
    
    return pd.DataFrame(results)

# Main evaluation process function
def process_evaluation(chemin_questions_csv, rag_csv, resultats_csv):
    # Load data
    rag_df = pd.read_csv(rag_csv)
    questions_df = pd.read_csv(chemin_questions_csv, usecols=['id', 'current_phrase'])
    rag_df = rag_df.merge(questions_df, on='id', how='left')

    # Evaluate each phrase for all metrics
    resultats = evaluer_phrase_parallele(rag_df)
    
    # Save the results to CSV
    resultats.to_csv(resultats_csv, index=False, quotechar='"')
    print(f"Evaluation results saved in {resultats_csv}")