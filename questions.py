import pandas as pd

from llms import create_questions_llm, generate_questions_parallel


def question_generation_process(file_path, output_path_questions):
    df = pd.read_csv(file_path)

    # Convertir la colonne 'binary_response' en texte (si elle est en format texte)
    df['binary_response'] = df['binary_response'].astype(str)

    # Filtrer uniquement les phrases identifiées comme liées à l'environnement (réponse binaire '1')
    df_environment = df[df['binary_response'] == '1']

    # Créer la LLMChain pour la génération des questions
    llm_chain = create_questions_llm()

    # Générer les questions pour les phrases liées à l'environnement
    questions_df = generate_questions_parallel(df_environment, llm_chain)

    # Sauvegarder les résultats dans un nouveau fichier CSV
    questions_df.to_csv(output_path_questions, index=False)
    print(f"Questions generated and saved to {output_path_questions}")
