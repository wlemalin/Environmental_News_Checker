import pandas as pd

from llms import generate_questions_parallel
from prompt import create_questions_llm


def question_generation_process(file_path, output_path_questions):
    """
    Generates questions from sentences identified as environmentally related and saves them to a CSV file.

    Args:
        file_path (str): Path to the input CSV file containing sentences and their binary response classification.
        output_path_questions (str): Path to the output CSV file where generated questions will be saved.

    Workflow:
        1. Load the input CSV file into a pandas DataFrame.
        2. Convert the 'binary_response' column to a string format.
        3. Filter the DataFrame to include only sentences related to the environment ('binary_response' is '1').
        4. Create an LLM chain using the `create_questions_llm` function.
        5. Generate questions for the filtered sentences using the LLM chain.
        6. Save the generated questions to the output CSV file.
        7. Print a confirmation message indicating where the questions have been saved.
    """
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
