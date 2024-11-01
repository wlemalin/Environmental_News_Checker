import nltk
import pandas as pd
from nltk import sent_tokenize

from llms import (analyze_paragraphs_parallel, parsed_responses)
from prompt import prompt_selection_phrase_pertinente
from topic_classifier import generate_context_windows


def identifier_extraits_sur_giec(file_path, output_path, output_path_improved):
    nltk.download('punkt')  # Téléchargez le modèle de tokenisation des phrases

    llm_chain = prompt_selection_phrase_pertinente()

    # Charger et regrouper le texte en phrases
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = sent_tokenize(text)  # Divise le texte en phrases
    splitted_text = generate_context_windows(sentences)

    # Analyser les paragraphes avec Llama 3.2 en parallèle
    analysis_results = analyze_paragraphs_parallel(splitted_text, llm_chain)

    # Sauvegarder les résultats dans un fichier CSV
    df = pd.DataFrame(analysis_results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Appliquer la méthode de parsing au DataFrame
    parsed_df_improved = parsed_responses(df)

    # Sauvegarder le DataFrame avec les résultats parsés
    parsed_df_improved['subjects'] = parsed_df_improved['subjects'].apply(
        lambda x: ', '.join(x))
    parsed_df_improved.to_csv(output_path_improved, index=False)

    # Affichage de quelques lignes du DataFrame final
    print(parsed_df_improved.head())
