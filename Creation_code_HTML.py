import os
import json

def generate_html_from_json(json_dir, output_html, articles_data_dir):
    # Lecture des fichiers JSON et stockage des données
    articles_data = {}
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
                article_key = os.path.splitext(filename)[0]
                articles_data[article_key] = article_data
                
    # Sauvegarde de chaque article comme fichier JSON individuel
    os.makedirs(articles_data_dir, exist_ok=True)
    for article_key, article_data in articles_data.items():
        with open(os.path.join(articles_data_dir, f"{article_key}.json"), 'w', encoding='utf-8') as f:
            json.dump(article_data, f, indent=4, ensure_ascii=False)

    # Création du contenu HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Article Analysis</title>
    <style>
        /* Styles du corps et du contenu */
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            background-color: #f4f4f9;
            color: #333;
            margin: 0;
            padding: 0;
        }}
        .container {{
            width: 90%;
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            background: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        /* Autres styles */
        /* Ajouter le reste des styles ici */
    </style>
</head>
<body>

<div class="container">
    <h1 id="article-title">Article Analysis</h1>
    <div id="article-content"></div>
</div>

<div style="margin-bottom: 20px; text-align: center;">
    <label for="article-select">Choose an article:</label>
    <select id="article-select" onchange="loadArticle(this.value)">
        <option value="">Select an article</option>
"""

    # Ajouter les options pour chaque article dans le menu déroulant
    for article_key in articles_data.keys():
        html_content += f'        <option value="{article_key}">{articles_data[article_key]["article_title"]}</option>\n'

    # Ajout du script et finalisation du contenu HTML
    html_content += """    </select>
</div>
"""
    # Insérer des données JSON pour chaque article
    for article_key, article_data in articles_data.items():
        html_content += f'<script type="application/json" id="article-data-{article_key}">\n'
        html_content += json.dumps(article_data, indent=4)
        html_content += "\n</script>\n"

    # Ajout du JavaScript
    html_content += """
<script>
    // Ajoutez ici le JavaScript nécessaire
</script>

</body>
</html>
"""

    # Écriture du fichier HTML de sortie
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML file created at {output_html}")
