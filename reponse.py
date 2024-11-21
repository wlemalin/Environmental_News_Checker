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

# Définir le template de prompt pour la génération de réponses
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

# Fonction pour générer un prompt formaté
def generate_answer_prompt(question, consolidated_text):
    return answer_prompt_template.format(question=question, consolidated_text=consolidated_text)

# Fonction pour générer une réponse avec le LLM
def generate_answer_with_llm(question, consolidated_text):
    prompt = generate_answer_prompt(question, consolidated_text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Générer la réponse
    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=500,  # Ajustez la limite selon les besoins
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7
    )
    
    # Décoder la réponse générée
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    print(f"LLM Response: {response}")
    return response_only

# Fonction pour gérer la génération des réponses en parallèle
def answer_questions_parallel(df_questions):
    results = []

    with ThreadPoolExecutor(max_workers=1) as executor:  # Ajustez le nombre de workers si nécessaire
        futures = {
            executor.submit(generate_answer_with_llm, row['question'], row['resume_sections']): row
            for _, row in df_questions.iterrows()
        }

        # Traiter les résultats au fur et à mesure de leur achèvement
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating answers"):
            row = futures[future]
            question_id = row['id']
            question = row['question']
            resume_sections = row['resume_sections']
            sections_brutes = row['sections']

            try:
                # Obtenir la réponse générée
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

# Fonction principale pour exécuter le processus RAG
def process_reponses(chemin_questions_csv, chemin_resultats_csv):
    """
    Exécute le processus de récupération et génération de réponses (RAG) pour générer des réponses et les sauvegarder dans un fichier CSV.

    Args:
        chemin_questions_csv (str): Chemin du fichier CSV contenant les questions.
        chemin_resultats_csv (str): Chemin du fichier CSV de sortie où les résultats seront sauvegardés.

    Workflow :
        1. Charger les questions depuis le fichier CSV d'entrée dans un DataFrame pandas.
        2. Traiter les questions en parallèle pour générer les réponses à l'aide du LLM.
        3. Sauvegarder les réponses générées dans le fichier CSV de sortie.
        4. Afficher un message de confirmation indiquant où les résultats ont été sauvegardés.
    """
    questions_df = pd.read_csv(chemin_questions_csv)

    # Générer les réponses et les sauvegarder dans un fichier CSV
    answers_df = answer_questions_parallel(questions_df)
    answers_df.to_csv(chemin_resultats_csv, index=False, quotechar='"')
    print(f"Answers saved to file {chemin_resultats_csv}")
