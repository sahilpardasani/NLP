# Latent Dirichlet Allocation (LDA) is a popular topic modeling technique in NLP
# It identifies abstract topics within a collection of documents by discovering groups of words that frequently appear together
# Each document is modeled as a mixture of topics, and each topic is characterized by a distribution of words

import os
import re
import json
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

app = Flask(__name__)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Perform LDA on the input text
def perform_lda(documents, n_topics=3, n_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    dt_matrix = vectorizer.fit_transform(documents)
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(dt_matrix)
    
    words = vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(lda_model.components_):
        top_words = [words[i] for i in topic.argsort()[-n_words:][::-1]]
        topics[f"Topic {idx + 1}"] = top_words
    return topics

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    text = request.form.get("text")
    n_topics = int(request.form.get("n_topics", 3))
    n_words = int(request.form.get("n_words", 5))
    
    if not text.strip():
        return {"error": "Input text cannot be empty."}, 400

    documents = [preprocess_text(doc) for doc in text.split("\n") if doc.strip()]
    topics = perform_lda(documents, n_topics=n_topics, n_words=n_words)
    return json.dumps(topics)

if __name__ == "__main__":
    app.run(debug=True)