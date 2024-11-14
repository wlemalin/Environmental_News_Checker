#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:49:28 2024

@author: mateodib
"""

import os
import pandas as pd
from llms import generate_questions_parallel_api

# Configure the Replicate API key
os.environ["REPLICATE_API_TOKEN"] = "r8_KVdlDIHTh9T6xEuEJhDkNxvfCXleqe814zH72"

def question_generation_process_api(file_path, output_path_questions):
    """
    Generates questions from sentences identified as environmentally related and saves them to a CSV file.
    """
    df = pd.read_csv(file_path)

    # Convert the 'binary_response' column to a string format if needed
    df['binary_response'] = df['binary_response'].astype(str)

    # Filter only sentences identified as environment-related (binary response '1')
    df_environment = df[df['binary_response'] == '1']

    # Generate questions for environment-related sentences using Replicate API
    questions_df = generate_questions_parallel_api(df_environment)

    # Save the results to a new CSV file
    questions_df.to_csv(output_path_questions, index=False)
    print(f"Questions generated and saved to {output_path_questions}")


