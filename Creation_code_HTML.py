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
            /* General Body Styling */
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                background-color: #f4f4f9;
                color: #333;
                margin: 0;
                padding: 0;
            }}
            /* Container */
            .container {{
                width: 90%;
                max-width: 800px;
                margin: 20px auto;
                padding: 20px;
                background: #ffffff;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            /* Title */
            h1 {{
                text-align: center;
                font-size: 2em;
                color: #2c3e50;
                font-family: 'Georgia', serif;
                margin-bottom: 10px;
            }}
            /* Phrase Styling */
            .phrase {{
                display: block;
                margin: 10px 0;
                padding: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .phrase:hover {{
                background-color: #e0f7fa;
            }}
            /* Highlighted phrase */
            .highlighted-phrase {{
                color: #006400;
                font-weight: bold;
            }}
            /* Popup Styling */
            .popup {{
                display: none;
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                padding: 20px;
                background: white;
                border: 1px solid #ccc;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
                z-index: 10;
                width: 90%;
                max-width: 500px;
                max-height: 70%;
                overflow-y: auto;
                border-radius: 8px;
            }}
            /* Overlay */
            .overlay {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 5;
            }}
            /* Close Button */
            .close-btn {{
                color: #007bff;
                cursor: pointer;
                font-weight: bold;
                position: absolute;
                top: 10px;
                right: 15px;
            }}
            /* Justification Text */
            .justification-text {{
                display: none;
                margin-top: 10px;
                font-size: 0.9em;
                color: #333;
                border-top: 1px solid #ddd;
                padding-top: 10px;
            }}
            /* Score Color Coding */
            .score {{
                font-weight: bold;
                padding: 5px;
                border-radius: 4px;
            }}
            .score-high {{ color: green; }}
            .score-medium {{ color: orange; }}
            .score-low {{ color: red; }}
        </style>
    </head>
    <body>
    
    <div class="container">
        <h1 id="article-title">Article Analysis</h1>
        <div id="article-content"></div>
    </div>
    
    <!-- Popup -->
    <div class="overlay" id="overlay" onclick="closePopup()"></div>
    <div class="popup" id="popup">
        <span class="close-btn" onclick="closePopup()">X</span>
        <h2>Phrase Analysis</h2>
        <p><strong>Accuracy Score:</strong> <span id="accuracy-score" class="score"></span> <button onclick="toggleJustification('accuracy')">Show Justification</button></p>
        <p><strong>Bias Score:</strong> <span id="bias-score" class="score"></span> <button onclick="toggleJustification('bias')">Show Justification</button></p>
        <p><strong>Tone Score:</strong> <span id="tone-score" class="score"></span> <button onclick="toggleJustification('tone')">Show Justification</button></p>
        <p><strong>Clarity Score:</strong> <span id="clarity-score" class="score"></span> <button onclick="toggleJustification('clarity')">Show Justification</button></p>
        <p><strong>Completeness Score:</strong> <span id="completeness-score" class="score"></span> <button onclick="toggleJustification('completeness')">Show Justification</button></p>
        <p><strong>Objectivity Score:</strong> <span id="objectivity-score" class="score"></span> <button onclick="toggleJustification('objectivity')">Show Justification</button></p>
        <p><strong>Alignment Score:</strong> <span id="alignment-score" class="score"></span> <button onclick="toggleJustification('alignment')">Show Justification</button></p>
        
        <div id="justification-accuracy" class="justification-text"></div>
        <div id="justification-bias" class="justification-text"></div>
        <div id="justification-tone" class="justification-text"></div>
        <div id="justification-clarity" class="justification-text"></div>
        <div id="justification-completeness" class="justification-text"></div>
        <div id="justification-objectivity" class="justification-text"></div>
        <div id="justification-alignment" class="justification-text"></div>
    </div>
    
    <!-- Dropdown Menu for Selecting Article -->
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
    // Load selected article data from the embedded JSON
    function loadArticle(articleKey) {
        if (!articleKey) return;
        
        // Get the JSON data from the appropriate <script> tag
        const articleDataScript = document.getElementById(`article-data-${articleKey}`);
        if (!articleDataScript) return;
        
        const data = JSON.parse(articleDataScript.textContent);
        displayArticle(data);
    }
    
    function displayArticle(data) {
        // Clear previous content
        document.getElementById("article-content").innerHTML = '';
        
        // Display the article title
        document.getElementById("article-title").innerText = data.article_title;
        
        // Iterate over phrases and display them
        Object.keys(data.phrases).forEach(id => {
            const phraseData = data.phrases[id];
            const hasValidAnalysis = Object.values(phraseData.analysis).some(metric => metric.score !== null);
            
            const phraseElement = document.createElement("span");
            phraseElement.className = hasValidAnalysis ? "phrase highlighted-phrase" : "phrase";
            phraseElement.innerText = phraseData.text;
            phraseElement.dataset.index = id; // Store index for reference
            
            if (hasValidAnalysis) {
                phraseElement.onclick = function() {
                    showPopup(id, data.phrases);
                };
            }
            
            document.getElementById("article-content").appendChild(phraseElement);
            document.getElementById("article-content").appendChild(document.createElement("br"));
        });
    }
    
    // Show popup with details
    function showPopup(id, phrases) {
        const phraseData = phrases[id].analysis;
        setScoreColor(phraseData.accuracy.score, "accuracy-score");
        setScoreColor(phraseData.bias.score, "bias-score");
        setScoreColor(phraseData.tone.score, "tone-score");
        setScoreColor(phraseData.clarity.score, "clarity-score");
        setScoreColor(phraseData.completeness.score, "completeness-score");
        setScoreColor(phraseData.objectivity.score, "objectivity-score");
        setScoreColor(phraseData.alignment.score, "alignment-score");
        
        document.getElementById("justification-accuracy").textContent = phraseData.accuracy.justifications || "No justification available.";
        document.getElementById("justification-bias").textContent = phraseData.bias.justifications || "No justification available.";
        document.getElementById("justification-tone").textContent = phraseData.tone.justifications || "No justification available.";
        document.getElementById("justification-clarity").textContent = phraseData.clarity.justifications || "No justification available.";
        document.getElementById("justification-completeness").textContent = phraseData.completeness.justifications || "No justification available.";
        document.getElementById("justification-objectivity").textContent = phraseData.objectivity.justifications || "No justification available.";
        document.getElementById("justification-alignment").textContent = phraseData.alignment.justifications || "No justification available.";
        
        document.getElementById("overlay").style.display = "block";
        document.getElementById("popup").style.display = "block";
    }
    
    function setScoreColor(score, elementId) {
        const element = document.getElementById(elementId);
        element.textContent = score;
        element.classList.remove("score-high", "score-medium", "score-low");
        if (score >= 4) element.classList.add("score-high");
        else if (score >= 2) element.classList.add("score-medium");
        else element.classList.add("score-low");
    }
    
    function toggleJustification(type) {
        const justificationDiv = document.getElementById(`justification-${type}`);
        justificationDiv.style.display = justificationDiv.style.display === "none" ? "block" : "none";
    }
    
    function closePopup() {
        document.getElementById("overlay").style.display = "none";
        document.getElementById("popup").style.display = "none";
    }
    </script>
    
    </body>
    </html>
    """

    # Écriture du fichier HTML de sortie
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML file created at {output_html}")
