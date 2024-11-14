import replicate
import pandas as pd
import nltk
from nltk import sent_tokenize
from llms import parsed_responses
from generate_context_windows import generate_context_windows
from llms import analyze_paragraphs_parallel_api

# Explicitly set the Replicate API key
replicate.api_token = "r8_KVdlDIHTh9T6xEuEJhDkNxvfCXleqe814zH72"


# Main function to identify excerpts related to IPCC
def identifier_extraits_sur_giec_api(file_path, output_path, output_path_improved):
    nltk.download('punkt')  # Download the sentence tokenizer model
    
    # Load and split text into sentences
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = sent_tokenize(text)  # Split text into sentences
    splitted_text = generate_context_windows(sentences)

    # Analyze paragraphs using Replicate API in parallel
    analysis_results = analyze_paragraphs_parallel_api(splitted_text)

    # Save the results to a CSV file
    df = pd.DataFrame(analysis_results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Apply the parsing method to the DataFrame
    parsed_df_improved = parsed_responses(df)

    # Save the parsed results DataFrame
    parsed_df_improved['subjects'] = parsed_df_improved['subjects'].apply(
        lambda x: ', '.join(x))
    parsed_df_improved.to_csv(output_path_improved, index=False)

    # Display a few rows of the final DataFrame
    print(parsed_df_improved.head())
