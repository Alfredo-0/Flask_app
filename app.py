from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import base64
from io import BytesIO
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd
from plotly.io import to_html
import numpy as np
import json
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

#get the enviromental variables
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/wordClouds')
def wordClouds():
    # List of words you want to display for selection
    words = ["mercantil","organización","colectivo","voluntario",
             "acumulacion capital","economia popular","servicio domestico",
            "reproduccion ampliada de la vida", "hogar","remunerado"]
    return render_template('wordClouds.html', words=words)

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    
    keyword = request.form['keyword']
    if keyword == "reproduccion ampliada de la vida":
        keyword = "reproduccion vida"
    images = []
    for i in range(1,4):  # Assuming three topics after reduction
        with open(f'wordclouds_data/{keyword}_topic_{i}_word_scores.json', 'r') as f:
            word_score_dict = json.load(f)
            word_score_dict.pop(keyword, None)
            # Apply softmax to the loaded scores
            words = list(word_score_dict.keys())
            scores = list(word_score_dict.values())
            scores = softmax(scores)
            word_score_dict = dict(zip(words, scores))

            wordcloud = WordCloud(width=1600, height=400, background_color="black").generate_from_frequencies(word_score_dict)

            img = BytesIO()
            wordcloud.to_image().save(img, format='PNG')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
            images.append(f"data:image/png;base64,{img_base64}")    
    return jsonify({'images': images})

@app.route('/visualization')
def visualization():
    proj_3d = np.loadtxt('visualization_data/umap_proj_3d.txt')
    cluster_labels = np.loadtxt('visualization_data/hdbscan_cluster_labels.txt', dtype=int)

    with open('visualization_data/top_3_topic_words.txt', 'r') as file:
        doc_top_3_words = [line.strip().split('; ') for line in file]


    # Prepare the DataFrame
    df = pd.DataFrame({
        'x': proj_3d[:,0],
        'y': proj_3d[:,1],
        'z': proj_3d[:,2],
        'Top 3 Topic Words': ['; '.join(words) for words in doc_top_3_words],  # Joining the top 3 words for hover info
        'Cluster': cluster_labels
    })

    # Generate the 3D scatter plot
    fig_3d = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='Cluster',
        hover_data=['Top 3 Topic Words']  # Include the top 3 topic words in the hover data
    )

    # Adjust marker size for better visibility
    fig_3d.update_traces(marker_size=1.5)  

    # Convert the figure to HTML
    plot_html = to_html(fig_3d, full_html=True)

    # Render it in the template
    return render_template("visualization.html", plot_html=plot_html)

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run()