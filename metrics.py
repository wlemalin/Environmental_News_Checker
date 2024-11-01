from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from langchain import PromptTemplate
from langchain_ollama import OllamaLLM
from tqdm import tqdm
from prompt import creer_prompts_metrics

    

# Function to evaluate a phrase for accuracy, bias, and tone
def evaluer_trois_taches_sur_phrase(phrase_id, question, current_phrase, sections_resumees,
                                    llm_sequence_exactitude, llm_sequence_biais, llm_sequence_ton):
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
def process_evaluation(rag_csv, resultats_csv):
    # Load rag_results.csv
    rag_df = pd.read_csv(rag_csv)
    
    # Load final_climate_analysis_with_questions.csv to get "current_phrase" column
    questions_df = pd.read_csv("/Users/mateodib/Desktop/Environmental_News_Checker-Mateo/final_climate_analysis_with_questions.csv", usecols=['id', 'current_phrase'])
    
    # Merge rag_df with questions_df on 'id'
    rag_df = rag_df.merge(questions_df, on='id', how='left')
    
    # Initialize LLM model once and create chains for each evaluation task
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    
    # Create prompt templates for each task
    prompt_exactitude, prompt_biais, prompt_ton = creer_prompts_metrics()
    
    # Create RunnableSequences for each task using shared LLM instance
    llm_sequence_exactitude = PromptTemplate(template=prompt_exactitude, input_variables=["current_phrase", "sections_resumees"]) | llm
    llm_sequence_biais = PromptTemplate(template=prompt_biais, input_variables=["current_phrase", "sections_resumees"]) | llm
    llm_sequence_ton = PromptTemplate(template=prompt_ton, input_variables=["current_phrase", "sections_resumees"]) | llm
    
    # Evaluate each phrase for accuracy, bias, and tone
    resultats = evaluer_phrase_parallele(rag_df, llm_sequence_exactitude, llm_sequence_biais, llm_sequence_ton)
    
    # Save results
    resultats.to_csv(resultats_csv, index=False, quotechar='"')
    print(f"Evaluation results saved in {resultats_csv}")
