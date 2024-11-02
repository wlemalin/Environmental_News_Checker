#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 04:35:55 2024

@author: mateodib
"""

import pandas as pd
import re
import os

# Define the directory containing the input CSV files
input_directory = "/Users/mateodib/Desktop/Environmental_News_Checker-2/Data/resultats/resultats_intermediaires/evaluation/"

# Define the directory where the parsed output files will be saved
output_directory = "/Users/mateodib/Desktop/Environmental_News_Checker-2/Data/resultats/resultats_intermediaires/evaluation_parsed/"
os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist

# Function to manually extract 'score' and 'justifications' from simulated JSON text
def parse_simulated_json(response_text):
    """
    Parses simulated JSON-style text and extracts 'score' and 'justifications'.
    Returns None for both if parsing fails.
    """
    # Initialize the result with None values
    score = None
    justification = None
    
    # Extract score using regex
    score_match = re.search(r'"score"\s*:\s*(\d+)', response_text)
    if score_match:
        score = int(score_match.group(1))
    
    # Extract justification using regex
    justification_match = re.search(r'"justifications"\s*:\s*"(.*?)"', response_text, re.DOTALL)
    if justification_match:
        justification = justification_match.group(1).strip()
    
    return score, justification

# Process each CSV file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        # Construct the full file path
        chemin_fichier = os.path.join(input_directory, filename)
        
        # Load the data from the CSV file
        df = pd.read_csv(chemin_fichier)

        # Initialize a list to store the parsed data
        parsed_data = []

        # Iterate through each row to extract the information
        for idx, row in df.iterrows():
            # Extract information from each column (simulated JSON-format responses)
            accuracy_score, accuracy_justification = parse_simulated_json(row['accuracy'])
            bias_score, bias_justification = parse_simulated_json(row['bias'])
            tone_score, tone_justification = parse_simulated_json(row['tone'])
            clarity_score, clarity_justification = parse_simulated_json(row['clarity'])
            completeness_score, completeness_justification = parse_simulated_json(row['completeness'])
            objectivity_score, objectivity_justification = parse_simulated_json(row['objectivity'])
            alignment_score, alignment_justification = parse_simulated_json(row['alignment'])

            # Append the extracted data to the list, including other relevant fields from the original data
            parsed_data.append({
                'id': row['id'],
                'question': row['question'],
                'current_phrase': row['current_phrase'],
                'sections_resumees': row['sections_resumees'],
                'accuracy_score': accuracy_score,
                'accuracy_justification': accuracy_justification,
                'bias_score': bias_score,
                'bias_justification': bias_justification,
                'tone_score': tone_score,
                'tone_justification': tone_justification,
                'clarity_score': clarity_score,
                'clarity_justification': clarity_justification,
                'completeness_score': completeness_score,
                'completeness_justification': completeness_justification,
                'objectivity_score': objectivity_score,
                'objectivity_justification': objectivity_justification,
                'alignment_score': alignment_score,
                'alignment_justification': alignment_justification
            })

        # Convert the parsed data into a DataFrame for saving
        df_parsed = pd.DataFrame(parsed_data)

        # Define the output file path based on the original file name
        output_filename = f"{os.path.splitext(filename)[0]}_parsed.csv"
        chemin_sortie = os.path.join(output_directory, output_filename)
        
        # Save the parsed DataFrame to a new CSV file
        df_parsed.to_csv(chemin_sortie, index=False, quotechar='"')
        print(f"Parsed data saved in {chemin_sortie}")