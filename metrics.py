import torch
from transformers import pipeline, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

# Initialiser le modèle et le tokenizer avec pipeline pour text-generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id  # Définit pad_token_id pour éviter le warning
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

# Fonction pour évaluer une phrase sur toutes les métriques
def evaluer_phrase_sur_toutes_metrices(phrase_id, question, current_phrase, sections_resumees):
    evaluations = {}
    for metric, prompt_template in prompts.items():
        prompt = prompt_template.format(current_phrase=current_phrase, sections_resumees=sections_resumees)
        
        # Génération avec pipeline
        output = pipe(prompt, max_new_tokens=256)
        
        # Extraction de la réponse générée dans le champ 'content' en suivant l'exemple fourni
        generated_text = output[0]["content"].strip() if "content" in output[0] else output[0]["generated_text"].strip()

        evaluations[metric] = generated_text
    
    return {
        "id": phrase_id,
        "question": question,
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees,
        **evaluations
    }

# Fonction pour paralléliser l'évaluation des phrases sur toutes les métriques
def evaluer_phrase_parallele(rag_df):
    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for _, row in rag_df.iterrows():
            phrase_id = row['id']
            question = row['question']
            current_phrase = row['current_phrase']
            sections_resumees = row['sections_resumees']
            futures.append(executor.submit(
                evaluer_phrase_sur_toutes_metrices,
                phrase_id, question, current_phrase, sections_resumees
            ))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating phrases for all metrics"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Error evaluating phrase: {exc}")
    
    return pd.DataFrame(results)

# Fonction principale pour orchestrer le processus d'évaluation
def process_evaluation(chemin_questions_csv, rag_csv, resultats_csv):
    # Chargement des données
    rag_df = pd.read_csv(rag_csv)
    questions_df = pd.read_csv(chemin_questions_csv, usecols=['id', 'current_phrase'])
    rag_df = rag_df.merge(questions_df, on='id', how='left')

    # Évaluer chaque phrase pour toutes les métriques
    resultats = evaluer_phrase_parallele(rag_df)
    
    # Sauvegarder les résultats
    resultats.to_csv(resultats_csv, index=False, quotechar='"')
    print(f"Evaluation results saved in {resultats_csv}")