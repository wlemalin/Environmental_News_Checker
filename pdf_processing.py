"""
Ce script fournit un pipeline de traitement pour extraire du texte à partir d'un PDF, le nettoyer et le découper en sections.
Il enregistre ensuite les sections dans un fichier JSON.

Fonctionnalités principales :
- Extraction de texte à partir de fichiers PDF
- Nettoyage du texte extrait pour supprimer les espaces et numéros de page
- Découpage du texte en sections
- Sauvegarde des sections dans un fichier JSON

"""

from pdfminer.high_level import extract_text
from file_utils import save_database
from txt_manipulation import clean_text, split_text_by_sections


# Pipeline principal de traitement
def process_pdf_to_index(chemin_rapport_pdf: str, chemin_output_json: str) -> None:
    """
    Traite un fichier PDF pour en extraire du texte, le nettoyer, le découper en sections, 
    puis sauvegarder les sections dans un fichier JSON.

    Args:
        chemin_rapport_pdf (str): Chemin vers le fichier PDF à traiter.
        chemin_output_json (str): Chemin du fichier JSON où sauvegarder les sections extraites.
    """
    # Extraction du texte du PDF
    raw_text = extract_text(chemin_rapport_pdf)

    # Nettoyage du texte extrait
    cleaned_text = clean_text(raw_text)

    # Découpage du texte en sections
    sections = split_text_by_sections(cleaned_text)

    # Sauvegarde des sections dans le fichier de sortie JSON
    save_database(sections, chemin_output_json)
