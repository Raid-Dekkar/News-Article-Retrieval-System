import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import os
import nltk
import matplotlib.pyplot as plt
from collections import Counter

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize Flask app
app = Flask(__name__)

# Load semantic search model
semantic_model = SentenceTransformer("all-mpnet-base-v2")

# Load dataset
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, sep="\t")
        df.dropna(subset=["title", "content"], inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Expand query with synonyms
def expand_query_with_synonyms(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return " ".join(list(synonyms.union(query.split())))

# Compute TF-IDF matrix
def compute_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["content"])
    return vectorizer, tfidf_matrix

# Ranked search
def ranked_search(query, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_doc_ids = scores.argsort()[::-1]
    return ranked_doc_ids, scores

# Semantic search
def semantic_search(query, df, top_k=5):
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    corpus_embeddings = semantic_model.encode(df["content"].tolist(), convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    return [(hit["corpus_id"], hit["score"]) for hit in hits]

# Generate visualizations
def generate_visualizations(df):
    all_words = Counter(word for content in df["content"] for word in preprocess_text(content))
    most_common = all_words.most_common(10)
    words, counts = zip(*most_common)

    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color="skyblue")
    plt.title("Top 10 Most Frequent Terms in the Dataset")
    plt.xlabel("Terms")
    plt.ylabel("Frequency")
    plt.savefig("static/term_frequency.png")
    plt.close()

# Flask routes
df, vectorizer, tfidf_matrix = None, None, None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    global df, vectorizer, tfidf_matrix
    query = request.form.get("query")
    search_type = request.form.get("search_type")
    use_synonyms = request.form.get("use_synonyms") == "on"

    if use_synonyms:
        query = expand_query_with_synonyms(query)

    results = []
    if search_type == "ranked":
        ranked_doc_ids, scores = ranked_search(query, vectorizer, tfidf_matrix)
        for rank, doc_id in enumerate(ranked_doc_ids[:5], 1):
            results.append({
                "rank": rank,
                "title": df.iloc[doc_id]["title"],
                "snippet": df.iloc[doc_id]["content"][:150] + "...",
                "score": scores[doc_id],
            })
    elif search_type == "semantic":
        semantic_results = semantic_search(query, df, top_k=5)
        for rank, (doc_id, score) in enumerate(semantic_results, 1):
            results.append({
                "rank": rank,
                "title": df.iloc[doc_id]["title"],
                "snippet": df.iloc[doc_id]["content"][:150] + "...",
                "score": score,
            })

    return render_template("results.html", query=query, results=results, search_type=search_type)

@app.route("/visualize")
def visualize():
    generate_visualizations(df)
    return render_template("visualization.html")

if __name__ == "__main__":
    dataset_path = "./dataset/bbc-news-data.csv"
    if not os.path.exists(dataset_path):
        print("Dataset not found! Ensure 'bbc-news-data.csv' is in the correct directory.")
        exit()
    df = load_data(dataset_path)
    vectorizer, tfidf_matrix = compute_tfidf(df)
    app.run(debug=True)
