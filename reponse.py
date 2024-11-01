from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from langchain import PromptTemplate
from langchain_ollama import OllamaLLM
from tqdm import tqdm


# Compare questions to the summarized report sections
def answer_questions_parallel(questions, llm_chain):
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
    questions_df = pd.read_csv(chemin_questions_csv)

    # Initialize LLM with the specified model
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")

    # Define the prompt for the LLM
    prompt_template = """
    Vous êtes un expert en climatologie. Répondez à la question ci-dessous en vous basant uniquement sur les sections pertinentes du rapport du GIEC.

    **Instructions** :
    1. Utilisez les informations des sections pour formuler une réponse précise et fondée.
    2. Justifiez votre réponse en citant les sections, si nécessaire.
    3. Limitez votre réponse aux informations fournies dans les sections.

    **Question** : {question}
    
    **Sections du rapport** : {consolidated_text}
    
    **Réponse** :
    - **Résumé de la réponse** : (Réponse concise)
    - **Justification basée sur le rapport** : (Citez et expliquez les éléments pertinents)
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "question", "consolidated_text"])
    llm_chain = prompt | llm  # Using simplified chaining without LLMChain

    # Generate answers and save them to CSV
    mentions = answer_questions_parallel(questions_df, llm_chain)
    df_mentions = pd.DataFrame(mentions)
    df_mentions.to_csv(chemin_resultats_csv, index=False, quotechar='"')
    print(f"Mentions saved to file {chemin_resultats_csv}")
