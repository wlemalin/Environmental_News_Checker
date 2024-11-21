from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import nltk
from nltk import sent_tokenize
from llms import parsed_responses
from generate_context_windows import generate_context_windows
import torch

# Configuration du modèle
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Configuration pour BitsAndBytes avec déchargement CPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # Activer le déchargement des poids sur le CPU
)

# Charger le modèle avec déchargement vers un dossier temporaire
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="balanced",  # Répartition automatique entre GPU et CPU
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    offload_folder="/tmp"  # Répertoire temporaire pour décharger les parties inutilisées
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Modèle chargé avec succès.")

# Définir le template de prompt
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
"""

# Fonction pour générer un prompt formaté
def generate_prompt(current_phrase, context):
    return prompt_template.format(current_phrase=current_phrase, context=context)

# Fonction pour analyser un paragraphe avec le modèle LLM
def analyze_paragraph_with_llm(current_phrase, context):
    prompt = generate_prompt(current_phrase, context)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Charger les données sur le GPU

    # Générer la réponse
    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=300,  # Ajustez selon les besoins
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7  # Contrôle de la diversité des réponses
    )

    # Décoder la réponse générée
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraire uniquement la réponse générée après le prompt
    response_only = response.split(prompt)[-1].strip()
    print(f"LLM Response: {response_only}")
    return response_only

# Fonction pour gérer l'analyse parallèle des paragraphes
def analyze_paragraphs_parallel(splitted_text):
    results = []

    # Using ThreadPoolExecutor pour gérer plusieurs tâches en parallèle
    with ThreadPoolExecutor(max_workers=1) as executor:  # Ajustez max_workers pour correspondre aux ressources
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
                # Récupérer le résultat de l'analyse
                analysis = future.result()

                # Ajouter uniquement les informations nécessaires pour économiser de la mémoire
                results.append({
                    "id": index,
                    "current_phrase": current_phrase,
                    "context": context,
                    "climate_related": analysis
                })

                # Print les résultats pour éviter de stocker trop de données en mémoire
                print(
                    f"ID: {index}\nPhrase:\n{current_phrase}\nContext:\n{context}\nLLM Response: {analysis}\n")

            except Exception as exc:
                print(f"Error analyzing phrase ID {index}: {current_phrase} - {exc}")

    return results

# Fonction principale pour identifier les extraits liés au GIEC
def identifier_extraits_sur_giec(file_path, output_path, output_path_improved):
    nltk.download("punkt")  # Télécharger le modèle de tokenisation

    # Charger et diviser le texte en phrases
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = sent_tokenize(text)  # Diviser le texte en phrases
    splitted_text = generate_context_windows(sentences)

    # Analyser les paragraphes avec Llama 3.2 en parallèle
    analysis_results = analyze_paragraphs_parallel(splitted_text)

    # Sauvegarder immédiatement les résultats dans un fichier CSV
    df = pd.DataFrame(analysis_results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Appliquer l'analyse pour extraire les sujets
    parsed_df_improved = parsed_responses(df)

    # Stream des résultats vers un CSV pour éviter de conserver de gros DataFrames en mémoire
    parsed_df_improved["subjects"] = parsed_df_improved["subjects"].apply(lambda x: ", ".join(x))
    parsed_df_improved.to_csv(output_path_improved, index=False)
    print(f"Improved results saved to {output_path_improved}")

    # Afficher quelques lignes du DataFrame final
    print(parsed_df_improved.head())
