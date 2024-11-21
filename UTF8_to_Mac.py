import os
import chardet
from ftfy import fix_text

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

def read_and_correct_text(file_path):
    # Detecte l'encodage initial
    encoding = detect_encoding(file_path)
    print(f"Encodage detecte pour {file_path} : {encoding}")
    
    # Lire le fichier avec l'encodage detecte
    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
        text = file.read()
    
    # Correction automatique des erreurs Unicode
    corrected_text = fix_text(text)
    return corrected_text

def save_as_utf8(file_path, corrected_text):
    # Sauvegarde le texte corrige directement dans le fichier d'origine, ecrasant le contenu
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(corrected_text)
    print(f"Fichier sauvegarde en UTF-8 : {file_path}")

def process_folder(folder_path):
    # Parcourir chaque fichier du dossier
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Verifie si c'est bien un fichier et si l'extension est .py
        if os.path.isfile(file_path) and filename.endswith('.py'):
            # Lire, corriger et sauvegarder en UTF-8 (ecrase le fichier d'origine)
            corrected_text = read_and_correct_text(file_path)
            save_as_utf8(file_path, corrected_text)

# Utilisation du script
folder_path = os.getcwd()  # Utilise le repertoire actuel ou se trouve le script
process_folder(folder_path)
