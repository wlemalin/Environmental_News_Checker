from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from langchain import LLMChain, PromptTemplate
from langchain_ollama import OllamaLLM
from tqdm import tqdm


# Generate answers using Llama3.2 and relevant sections of the report
def rag_answer_generation_with_llmchain(question, relevant_sections, llm_chain):
    inputs = {
        "question": question,
        "consolidated_text": relevant_sections
    }
    response = llm_chain.invoke(inputs)
    generated_answer = response['text'] if isinstance(
        response, dict) and "text" in response else response
    return generated_answer.strip()


def comparer_questions_rapport(questions_df, prompt_template):
    results = []

    # Use a limited number of threads to avoid memory overload
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Comparing questions"):
            ID = row['id']
            question = row['question']
            # Use the summarized sections specific to the question
            resume_sections = row['resume_sections']
            # Original sections for reference
            sections_brutes = row['sections']

            # Submit each question with its specific resume_sections for processing with a new model instance
            futures.append(executor.submit(
                trouver_sections_et_generer_reponse,
                question,
                resume_sections,
                sections_brutes,
                prompt_template,
                ID
            ))

        # Retrieve results as they complete
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

# Fonction pour trouver les sections pertinentes et générer une réponse


# Function to find relevant sections and generate a response
def trouver_sections_et_generer_reponse(question, resume_sections, sections_brutes, prompt_template, ID):
    # Create a new instance of OllamaLLM for each thread to ensure true parallelism
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    # Combine prompt with model to create a RunnableSequence
    llm_chain = prompt_template | llm
    # Generate the answer based on the summarized sections specific to this question
    generated_answer = rag_answer_generation_with_llmchain(
        question, resume_sections, llm_chain)
    return question, resume_sections, generated_answer, ID, sections_brutes


# Main function to execute the RAG process


# Main function to execute the RAG process with separate model instances for each thread
def rag_process(chemin_questions_csv, chemin_resultats_csv):
    # Load questions and relevant sections from rag_results.csv
    questions_df = pd.read_csv(chemin_questions_csv)

    # Define the prompt template for RAG
    prompt_template = PromptTemplate(
        template="""
        Vous êtes un expert en climatologie et votre rôle est d'analyser les informations provenant d'un article de presse sur le changement climatique. Votre tâche consiste à répondre à la question ci-dessous en utilisant uniquement les sections pertinentes du rapport du GIEC fournies.
        
        **Instructions** :
        1. Lisez attentivement la question posée par l'article.
        2. Utilisez les informations pertinentes des sections du rapport du GIEC pour formuler une réponse précise et fondée.
        3. Justifiez votre réponse en citant les sections du rapport du GIEC, si nécessaire.
        4. Limitez votre réponse aux informations présentes dans les sections fournies.
        
        **Question de l'article** : {question}
        
        **Sections pertinentes du rapport du GIEC** : {consolidated_text}
        
        **Réponse** :
        - **Résumé de la réponse** : (Donnez une réponse concise à la question)
        - **Justification basée sur le rapport du GIEC** : (Citez et expliquez les éléments pertinents du rapport)
        """,
        input_variables=["question", "consolidated_text"]
    )

    # Generate answers and save results
    mentions = comparer_questions_rapport(questions_df, prompt_template)
    df_mentions = pd.DataFrame(mentions)
    df_mentions.to_csv(chemin_resultats_csv, index=False)
    print(f"Mentions sauvegardées dans le fichier {chemin_resultats_csv}")
