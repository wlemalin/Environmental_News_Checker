from file_utils import save_database, extract_text_from_pdf
from txt_manipulation import clean_text, split_text_by_sections


# Main processing pipeline
def process_pdf_to_index(chemin_rapport_pdf, chemin_output_json):
    # Process PDF
    raw_text = extract_text_from_pdf(chemin_rapport_pdf)
    cleaned_text = clean_text(raw_text)
    sections = split_text_by_sections(cleaned_text)
    # Save the sections to output
    save_database(sections, chemin_output_json)

