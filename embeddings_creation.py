
import json

import tqdm


# Générer des embeddings pour les sections du rapport du GIEC
def generer_embeddings_rapport(chemin_rapport_json, embed_model):
    with open(chemin_rapport_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # Générer les embeddings pour chaque section de texte
    for section in tqdm(data, desc="Generating embeddings for report sections"):
        texte = section['text']
        # Ensure embeddings are on CPU
        embedding = embed_model.encode(
            texte, convert_to_tensor=True, device='cpu')
        # Convert tensor to list for JSON serialization
        section['embedding'] = embedding.cpu().numpy().tolist()
    # Sauvegarder les données avec les embeddings
    with open(chemin_rapport_json, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Embeddings generated and saved to {chemin_rapport_json}")


# Charger les embeddings des sections du rapport du GIEC après les avoir générés
def charger_embeddings_rapport(chemin_rapport_json):
    with open(chemin_rapport_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    sections = [section['text'] for section in data]
    embeddings = [section['embedding'] for section in data]
    titles = [section.get('title', 'Section sans titre')
              for section in data]  # Get section titles or default
    return embeddings, sections, titles


# Embedding function to efficiently process multiple texts
def embed_texts(texts, embed_model):
    # Ensure embedding is done on CPU
    embeddings = embed_model.encode(texts, convert_to_tensor=True, device='cpu')
    return embeddings


