from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from prompt import creer_prompt_reponses

# Compare questions to the summarized report sections
def answer_questions_parallel(questions, llm_chain):
    """
    Processes multiple questions in parallel to generate answers using the LLM chain.

    Args:
        questions (DataFrame): A pandas DataFrame containing the questions and relevant sections.
        llm_chain: A language model chain for generating responses.

    Returns:
        list: A list of dictionaries containing question IDs, questions, summarized sections, original sections, and generated answers.
    """
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Comparing questions"):
            ID = row['id']
            question = row['question']
            resume_sections = row['resume_sections']
            sections_brutes = row['sections']

            futures.append(executor.submit(
                answer_question,
                question,
                resume_sections,
                sections_brutes,
                llm_chain,
                ID
            ))

        # Gather results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Retrieving answers"):
            try:
                question, resume_sections, generated_answer, ID, sections_brutes = future.result()
                results.append({
                    "id": ID,
                    "question": question,
                    "sections_resumees": resume_sections,
                    "retrieved_sections": sections_brutes,
                    "reponse": generated_answer
                })
            except Exception as exc:
                print(f"Error during RAG: {exc}")

    return results

# Function to generate a response
def answer_question(question, resume_sections, sections_brutes, llm_chain, ID):
    """
    Generates an answer for a given question using the provided LLM chain.

    Args:
        question (str): The question to be answered.
        resume_sections (str): Consolidated text summarizing the relevant report sections.
        sections_brutes (str): Original sections of the report.
        llm_chain: A language model chain used to generate responses.
        ID (int): The unique identifier for the question.

    Returns:
        tuple: Contains the question, summarized sections, generated answer, question ID, and original sections.
    """
    inputs = {
        "question": question,
        "consolidated_text": resume_sections
    }
    response = llm_chain.invoke(inputs)
    generated_answer = response['text'] if isinstance(
        response, dict) and "text" in response else response
    generated_answer = generated_answer.strip()
    return question, resume_sections, generated_answer, ID, sections_brutes

# Main function to execute the RAG process
def process_reponses(chemin_questions_csv, chemin_resultats_csv):
    """
    Executes the retrieval-augmented generation (RAG) process to generate answers and save them to a CSV file.

    Args:
        chemin_questions_csv (str): Path to the CSV file containing questions.
        chemin_resultats_csv (str): Path to the output CSV file where results will be saved.

    Workflow:
        1. Load questions from the input CSV file into a pandas DataFrame.
        2. Create an LLM chain using the `creer_prompt_reponses` function.
        3. Process the questions in parallel to generate answers using the LLM chain.
        4. Save the generated answers to the output CSV file.
        5. Print a confirmation message indicating where the results have been saved.
    """
    questions_df = pd.read_csv(chemin_questions_csv)
    llm_chain = creer_prompt_reponses()

    # Generate answers and save them to CSV
    mentions = answer_questions_parallel(questions_df, llm_chain)
    df_mentions = pd.DataFrame(mentions)
    df_mentions.to_csv(chemin_resultats_csv, index=False, quotechar='"')
    print(f"Mentions saved to file {chemin_resultats_csv}")
