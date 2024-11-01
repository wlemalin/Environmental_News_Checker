from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from langchain import PromptTemplate
from langchain_ollama import OllamaLLM
from tqdm import tqdm
from prompt import creer_prompts_metrics

# Function to evaluate a phrase for accuracy, bias, and tone
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
    # Evaluate accuracy
    response_exactitude = llm_sequence_exactitude.invoke({
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees
    })
    exactitude = response_exactitude['text'].strip() if isinstance(response_exactitude, dict) else response_exactitude.strip()
    
    # Evaluate bias
    response_biais = llm_sequence_biais.invoke({
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees
    })
    biais = response_biais['text'].strip() if isinstance(response_biais, dict) else response_biais.strip()
    
    # Evaluate tone
    response_ton = llm_sequence_ton.invoke({
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees
    })
    ton = response_ton['text'].strip() if isinstance(response_ton, dict) else response_ton.strip()
    
    # Return results for this phrase
    return {
        "id": phrase_id,
        "question": question,
        "current_phrase": current_phrase,
        "sections_resumees": sections_resumees,
        "exactitude": exactitude,
        "biais": biais,
        "ton": ton
    }

# Function to parallelize evaluation of phrases for each metric with shared LLMs
def evaluer_phrase_parallele(rag_df, llm_sequence_exactitude, llm_sequence_biais, llm_sequence_ton):
    """
    Evaluates multiple phrases in parallel for accuracy, bias, and tone using shared LLM models.

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
            
            # Submit evaluation for exactitude, biais, and ton with shared LLMs
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

# Main function to run the evaluation process
def process_evaluation(chemin_questions_csv, rag_csv, resultats_csv):
    """
    Main function that orchestrates the evaluation process of phrases.

    Parameters:
    - chemin_questions_csv (str): Path to the CSV file containing questions and phrases to be evaluated.
    - rag_csv (str): Path to the CSV file containing RAG information including summarized sections.
    - resultats_csv (str): Path where the evaluation results CSV file should be saved.

    Workflow:
    1. Reads input CSVs and merges relevant data.
    2. Initializes an LLM model and creates evaluation chains for accuracy, bias, and tone.
    3. Evaluates each phrase for accuracy, bias, and tone using parallel processing.
    4. Saves the evaluation results to a CSV file.
    """
    rag_df = pd.read_csv(rag_csv)
    questions_df = pd.read_csv(chemin_questions_csv, usecols=['id', 'current_phrase'])
    rag_df = rag_df.merge(questions_df, on='id', how='left')
    
    # Initialize LLM model once and create chains for each evaluation task
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    prompt_exactitude, prompt_biais, prompt_ton = creer_prompts_metrics()
    llm_sequence_exactitude = PromptTemplate(template=prompt_exactitude, input_variables=["current_phrase", "sections_resumees"]) | llm
    llm_sequence_biais = PromptTemplate(template=prompt_biais, input_variables=["current_phrase", "sections_resumees"]) | llm
    llm_sequence_ton = PromptTemplate(template=prompt_ton, input_variables=["current_phrase", "sections_resumees"]) | llm
    
    # Evaluate each phrase for accuracy, bias, and tone
    resultats = evaluer_phrase_parallele(rag_df, llm_sequence_exactitude, llm_sequence_biais, llm_sequence_ton)
    
    # Save results
    resultats.to_csv(resultats_csv, index=False, quotechar='"')
    print(f"Evaluation results saved in {resultats_csv}")
