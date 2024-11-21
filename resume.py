from embeddings_creation import embed_texts, generer_embeddings_rapport
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np
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

# Définir le template de prompt pour le résumé
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

# Fonction pour générer un prompt formaté
def generate_resume_prompt(question, retrieved_sections):
    return resume_prompt_template.format(question=question, retrieved_sections=retrieved_sections)

# Fonction pour générer un résumé avec le LLM
def generate_resume_with_llm(question, section):
    prompt = generate_resume_prompt(question, section)
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

# Fonction pour gérer la génération de résumés en parallèle
def generer_resume_parallel(df_questions):
    results = []

    with ThreadPoolExecutor(max_workers=1) as executor:  # Ajustez le nombre de workers si nécessaire
        futures = {
            executor.submit(generate_resume_with_llm, row['question'], section): (row['id'], row['question'], section)
            for _, row in df_questions.iterrows()
            for section in row['retrieved_sections']
        }

        # Traiter les résultats au fur et à mesure de leur achèvement
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating summaries"):
            (question_id, question, section) = futures[future]

            try:
                # Obtenir le résumé généré
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

# Fonction pour charger les données et les embeddings
def charger_donnees_et_modele(chemin_csv_questions, chemin_rapport_embeddings):
    df_questions = pd.read_csv(chemin_csv_questions)
    model_path = "/home2020/home/beta/aebeling/Models_local/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9"
    embed_model = SentenceTransformer(model_path, device="cpu")
    print("Model d'embeddings chargé avec succès.")
    embeddings_rapport, sections_rapport, titles_rapport = generer_embeddings_rapport(chemin_rapport_embeddings, embed_model)
    return df_questions, embeddings_rapport, sections_rapport, titles_rapport, embed_model

# Fonction pour filtrer les sections pertinentes pour chaque question
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

# Fonction principale pour le processus de génération de résumés
def process_resume(chemin_csv_questions, chemin_rapport_embeddings, chemin_resultats_csv, top_k=3):
    df_questions, embeddings_rapport, sections_rapport, _, embed_model = charger_donnees_et_modele(chemin_csv_questions, chemin_rapport_embeddings)
    df_questions = filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k)
    resultats = generer_resume_parallel(df_questions)

    # Regrouper les résultats par ID de question
    resultats_grouped = resultats.groupby('id').agg({
        'question': 'first',
        'sections': lambda x: ' '.join(x),
        'resume_sections': lambda x: ' '.join(x)
    }).reset_index()

    # Sauvegarder les résultats dans un fichier CSV
    resultats_grouped.to_csv(chemin_resultats_csv, index=False)
    print(f"Résumés sauvegardés dans le fichier {chemin_resultats_csv}")
