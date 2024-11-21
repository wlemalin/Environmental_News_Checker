from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import torch

# Configuration du modèle LLM
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Charger le modèle quantifié avec déchargement
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    offload_folder="/tmp"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Modèle LLM chargé avec succès.")

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

# Générer un prompt formaté pour une métrique donnée
def generate_metric_prompt(metric, current_phrase, sections_resumees):
    return prompts[metric].format(current_phrase=current_phrase, sections_resumees=sections_resumees)

# Évaluer une métrique avec le LLM
def evaluate_metric_with_llm(metric, current_phrase, sections_resumees):
    prompt = generate_metric_prompt(metric, current_phrase, sections_resumees)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Générer la réponse
    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=500,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7
    )
    
    # Décoder la réponse générée
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"LLM Response for metric {metric}: {response}")
    return response

# Évaluer toutes les métriques pour une phrase
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

# Évaluation parallèle pour toutes les phrases et métriques
def evaluer_phrase_parallele(rag_df):
    results = []

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

# Fonction principale pour exécuter l'évaluation
def process_evaluation(chemin_questions_csv, rag_csv, resultats_csv):
    # Charger les données
    rag_df = pd.read_csv(rag_csv)
    questions_df = pd.read_csv(chemin_questions_csv, usecols=['id', 'current_phrase'])
    rag_df = rag_df.merge(questions_df, on='id', how='left')

    # Évaluer chaque phrase pour toutes les métriques
    resultats = evaluer_phrase_parallele(rag_df)
    
    # Sauvegarder les résultats dans un fichier CSV
    resultats.to_csv(resultats_csv, index=False, quotechar='"')
    print(f"Evaluation results saved in {resultats_csv}")
