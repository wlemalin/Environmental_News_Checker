from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from langchain import PromptTemplate
from langchain_ollama import OllamaLLM
from tqdm import tqdm


# Function to evaluate a phrase for accuracy, bias, and tone using distinct LLMs and prompts
def evaluer_trois_taches_sur_phrase(phrase_id, question, current_phrase, sections_resumees,
                                    llm_sequence_exactitude, llm_sequence_biais, llm_sequence_ton):
    """
    Evaluates a given phrase on three different metrics: accuracy, bias, and tone.

    Parameters:
    - phrase_id (int): The identifier for the phrase being evaluated.
    - question (str): The question related to the current phrase.
    - current_phrase (str): The phrase to be evaluated.
    - sections_resumees (str): Summarized sections used as context for evaluation.
    - llm_sequence_exactitude (RunnableSequence): Chain of prompt template and LLM for accuracy evaluation.
    - llm_sequence_biais (RunnableSequence): Chain of prompt template and LLM for bias evaluation.
    - llm_sequence_ton (RunnableSequence): Chain of prompt template and LLM for tone evaluation.

    Returns:
    - dict: A dictionary containing the evaluation results for accuracy, bias, and tone for the given phrase.
    """
    # Ajoutez un dictionnaire avec `task_id` pour chaque tâche
    input_exactitude = {
        "task_id": "exactitude",
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees
    }
    input_biais = {
        "task_id": "biais",
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees
    }
    input_ton = {
        "task_id": "ton",
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees
    }
    
    # Vérifiez que `task_id` correspond à la tâche attendue pour chaque séquence

    # Évaluation de l'exactitude
    assert input_exactitude["task_id"] == "exactitude", "Erreur : mauvais task_id pour exactitude."
    response_exactitude = llm_sequence_exactitude.invoke(input_exactitude)
    exactitude = response_exactitude['text'].strip() if isinstance(response_exactitude, dict) else response_exactitude.strip()
    
    # Évaluation du biais
    assert input_biais["task_id"] == "biais", "Erreur : mauvais task_id pour biais."
    response_biais = llm_sequence_biais.invoke(input_biais)
    biais = response_biais['text'].strip() if isinstance(response_biais, dict) else response_biais.strip()
    
    # Évaluation du ton
    assert input_ton["task_id"] == "ton", "Erreur : mauvais task_id pour ton."
    response_ton = llm_sequence_ton.invoke(input_ton)
    ton = response_ton['text'].strip() if isinstance(response_ton, dict) else response_ton.strip()
    
    # Retourne les résultats pour cette phrase
    return {
        "id": phrase_id,
        "question": question,
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees,
        "exactitude": exactitude,
        "biais": biais,
        "ton": ton
    }
# Function to parallelize evaluation of phrases for each metric with distinct LLMs
def evaluer_phrase_parallele(rag_df, llm_sequence_exactitude, llm_sequence_biais, llm_sequence_ton):
    """
    Evaluates multiple phrases in parallel for accuracy, bias, and tone using distinct LLM models.

    Parameters:
    - rag_df (DataFrame): A pandas DataFrame containing phrases and contextual information for evaluation.
    - llm_sequence_exactitude (RunnableSequence): Chain of prompt template and LLM for accuracy evaluation.
    - llm_sequence_biais (RunnableSequence): Chain of prompt template and LLM for bias evaluation.
    - llm_sequence_ton (RunnableSequence): Chain of prompt template and LLM for tone evaluation.

    Returns:
    - DataFrame: A pandas DataFrame containing the evaluation results for each phrase.
    """
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:  # Adjusted max_workers to 3 for M2
        futures = []
        
        for _, row in rag_df.iterrows():
            phrase_id = row['id']
            question = row['question']
            current_phrase = row['current_phrase']
            sections_resumees = row['sections_resumees']
            
            # Submit evaluation for exactitude, biais, and ton with distinct LLMs
            futures.append(executor.submit(
                evaluer_trois_taches_sur_phrase,
                phrase_id, question, current_phrase, sections_resumees,
                llm_sequence_exactitude, llm_sequence_biais, llm_sequence_ton
            ))
        
        # Gather results as tasks complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating phrases"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Error evaluating phrase: {exc}")
    
    return pd.DataFrame(results)

# Main function to run the evaluation process with distinct LLMs for each metric
def process_evaluation(chemin_questions_csv, rag_csv, resultats_csv):
    """
    Main function that orchestrates the evaluation process of phrases.

    Parameters:
    - chemin_questions_csv (str): Path to the CSV file containing questions and phrases to be evaluated.
    - rag_csv (str): Path to the CSV file containing RAG information including summarized sections.
    - resultats_csv (str): Path where the evaluation results CSV file should be saved.

    Workflow:
    1. Reads input CSVs and merges relevant data.
    2. Initializes LLMs and creates evaluation chains for accuracy, bias, and tone.
    3. Evaluates each phrase for accuracy, bias, and tone using parallel processing.
    4. Saves the evaluation results to a CSV file.
    """
    # Load data
    rag_df = pd.read_csv(rag_csv)
    questions_df = pd.read_csv(chemin_questions_csv, usecols=['id', 'current_phrase'])
    rag_df = rag_df.merge(questions_df, on='id', how='left')
    rag_df = rag_df.head(1)
    print(rag_df.head())
    
    # Initialize separate LLMs for each metric
    llm_exactitude = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    llm_biais = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    llm_ton = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    
    # Define distinct prompts for each metric
    prompt_exactitude = PromptTemplate(template="Evaluate accuracy: {current_phrase} in context: {sections_resumees}", input_variables=["current_phrase", "sections_resumees"])
    llm_sequence_exactitude = prompt_exactitude | llm_exactitude
    
    prompt_biais = PromptTemplate(template="Evaluate bias: {current_phrase} in context: {sections_resumees}", input_variables=["current_phrase", "sections_resumees"])
    llm_sequence_biais = prompt_biais | llm_biais
    
    prompt_ton = PromptTemplate(template="Evaluate tone: {current_phrase} in context: {sections_resumees}", input_variables=["current_phrase", "sections_resumees"])
    llm_sequence_ton = prompt_ton | llm_ton
    
    # Evaluate each phrase for accuracy, bias, and tone
    resultats = evaluer_phrase_parallele(rag_df, llm_sequence_exactitude, llm_sequence_biais, llm_sequence_ton)
    
    # Save results
    resultats.to_csv(resultats_csv, index=False, quotechar='"')
    print(f"Evaluation results saved in {resultats_csv}")
