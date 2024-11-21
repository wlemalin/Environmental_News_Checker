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

# Définir le template de prompt pour la génération de questions
question_prompt_template = """
Vous êtes chargé de formuler une **question précise** pour vérifier les informations mentionnées dans un extrait spécifique d'un article de presse en consultant directement les rapports du GIEC (Groupe d'experts intergouvernemental sur l'évolution du climat).

Cette question sera utilisée dans un système de récupération d'information (RAG) pour extraire les sections pertinentes des rapports du GIEC et comparer les informations des rapports avec celles de l'article de presse.

**Objectif** : La question doit permettre de vérifier si les informations fournies dans la phrase de l'article sont corroborées ou contestées par les preuves scientifiques dans les rapports du GIEC. La question doit donc englober tous les sujets abordés par l'extrait de l'article de presse.

**Instructions** :

1. Analysez l'extrait et son contexte pour identifier les affirmations clés et/ou les informations à vérifier.
2. Formulez une **question claire et spécifique** orientée vers la vérification de ces affirmations ou informations à partir des rapports du GIEC. La question doit permettre de vérifier toutes les informations de l'extrait. La question peut être un ensemble de questions comme : "Quel est l'impact des activités humaines sur le taux de CO2 dans l'atmosphère ? Comment la concentration du CO2 dans l'atmosphère impacte l'agriculture ?"
3. La question doit être **directement vérifiable** dans les rapports du GIEC via un système RAG.
4. **IMPORTANT** : Répondez uniquement avec la question, sans ajouter d'explications ou de contexte supplémentaire.

Extrait de l'article de presse : {current_phrase}

Contexte : {context}

Générez uniquement la **question** spécifique qui permettrait de vérifier les informations mentionnées dans cette phrase en consultant les rapports du GIEC via un système de récupération d'information (RAG).
"""

# Fonction pour générer un prompt formaté
def generate_question_prompt(current_phrase, context):
    return question_prompt_template.format(current_phrase=current_phrase, context=context)

# Fonction pour générer une question avec le LLM
def generate_question_with_llm(current_phrase, context):
    prompt = generate_question_prompt(current_phrase, context)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Générer la réponse
    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7
    )

    # Décoder la réponse générée
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Extraire uniquement la réponse générée après le prompt
    response_only = response.split(prompt)[-1].strip()
    print(f"LLM Response: {response_only}")
    return response_only

# Fonction pour gérer la génération de questions en parallèle
def generate_questions_parallel(df):
    results = []

    with ThreadPoolExecutor(max_workers=2) as executor:  # Ajustez le nombre de workers si nécessaire
        futures = {
            executor.submit(generate_question_with_llm, row['current_phrase'], row['context']): row
            for _, row in df.iterrows()
        }

        # Traiter les résultats au fur et à mesure de leur achèvement
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating questions"):
            row = futures[future]
            current_phrase = row['current_phrase']
            context = row['context']

            try:
                # Obtenir la question générée
                question = future.result()
                results.append({
                    "current_phrase": current_phrase,
                    "context": context,
                    "question": question
                })

                # Afficher la question générée pour le débogage
                print(f"Phrase:\n{current_phrase}\nContext:\n{context}\nGenerated Question:\n{question}\n")

            except Exception as exc:
                print(f"Error generating question for phrase: {current_phrase} - {exc}")

    return pd.DataFrame(results)

# Exemple de fonction principale pour charger les données, générer des questions et sauvegarder les résultats
def question_generation_process(file_path, output_path):
    df = pd.read_csv(file_path)

    # Générer les questions en parallèle et sauvegarder les résultats
    questions_df = generate_questions_parallel(df)
    questions_df.to_csv(output_path, index=False)
    print(f"Questions saved to {output_path}")
