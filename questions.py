from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import torch

# Configuration du mod�le LLM
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Charger le mod�le quantifi� avec d�chargement
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    offload_folder="/tmp"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Mod�le LLM charg� avec succ�s.")

# D�finir le template de prompt pour la g�n�ration de questions
question_prompt_template = """
Vous �tes charg� de formuler une **question pr�cise** pour v�rifier les informations mentionn�es dans un extrait sp�cifique d'un article de presse en consultant directement les rapports du GIEC (Groupe d'experts intergouvernemental sur l'�volution du climat).

Cette question sera utilis�e dans un syst�me de r�cup�ration d'information (RAG) pour extraire les sections pertinentes des rapports du GIEC et comparer les informations des rapports avec celles de l'article de presse.

**Objectif** : La question doit permettre de v�rifier si les informations fournies dans la phrase de l'article sont corrobor�es ou contest�es par les preuves scientifiques dans les rapports du GIEC. La question doit donc englober tous les sujets abord�s par l'extrait de l'article de presse.

**Instructions** :

1. Analysez l'extrait et son contexte pour identifier les affirmations cl�s et/ou les informations � v�rifier.
2. Formulez une **question claire et sp�cifique** orient�e vers la v�rification de ces affirmations ou informations � partir des rapports du GIEC. La question doit permettre de v�rifier toutes les informations de l'extrait. La question peut �tre un ensemble de questions comme : "Quel est l'impact des activit�s humaines sur le taux de CO2 dans l'atmosph�re ? Comment la concentration du CO2 dans l'atmosph�re impacte l'agriculture ?"
3. La question doit �tre **directement v�rifiable** dans les rapports du GIEC via un syst�me RAG.
4. **IMPORTANT** : R�pondez uniquement avec la question, sans ajouter d'explications ou de contexte suppl�mentaire.

Extrait de l'article de presse : {current_phrase}

Contexte : {context}

G�n�rez uniquement la **question** sp�cifique qui permettrait de v�rifier les informations mentionn�es dans cette phrase en consultant les rapports du GIEC via un syst�me de r�cup�ration d'information (RAG).
"""

# Fonction pour g�n�rer un prompt format�
def generate_question_prompt(current_phrase, context):
    return question_prompt_template.format(current_phrase=current_phrase, context=context)

# Fonction pour g�n�rer une question avec le LLM
def generate_question_with_llm(current_phrase, context):
    prompt = generate_question_prompt(current_phrase, context)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # G�n�rer la r�ponse
    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7
    )

    # D�coder la r�ponse g�n�r�e
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Extraire uniquement la r�ponse g�n�r�e apr�s le prompt
    response_only = response.split(prompt)[-1].strip()
    print(f"LLM Response: {response_only}")
    return response_only

# Fonction pour g�rer la g�n�ration de questions en parall�le
def generate_questions_parallel(df):
    results = []

    with ThreadPoolExecutor(max_workers=2) as executor:  # Ajustez le nombre de workers si n�cessaire
        futures = {
            executor.submit(generate_question_with_llm, row['current_phrase'], row['context']): row
            for _, row in df.iterrows()
        }

        # Traiter les r�sultats au fur et � mesure de leur ach�vement
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating questions"):
            row = futures[future]
            current_phrase = row['current_phrase']
            context = row['context']

            try:
                # Obtenir la question g�n�r�e
                question = future.result()
                results.append({
                    "current_phrase": current_phrase,
                    "context": context,
                    "question": question
                })

                # Afficher la question g�n�r�e pour le d�bogage
                print(f"Phrase:\n{current_phrase}\nContext:\n{context}\nGenerated Question:\n{question}\n")

            except Exception as exc:
                print(f"Error generating question for phrase: {current_phrase} - {exc}")

    return pd.DataFrame(results)

# Exemple de fonction principale pour charger les donn�es, g�n�rer des questions et sauvegarder les r�sultats
def question_generation_process(file_path, output_path):
    df = pd.read_csv(file_path)

    # G�n�rer les questions en parall�le et sauvegarder les r�sultats
    questions_df = generate_questions_parallel(df)
    questions_df.to_csv(output_path, index=False)
    print(f"Questions saved to {output_path}")
