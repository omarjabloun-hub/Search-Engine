from fastapi import FastAPI, Query
from typing import List, Dict, Any
import pandas as pd
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from collections import defaultdict, Counter

# Download data once if needed
nltk.download('stopwords')
nltk.download('punkt')

app = FastAPI()

df = None
inverted_index = None
idf = None
doc_vectors = None

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def combine_text(row):
    title = str(row['Job Title'])
    description = str(row['Job Description'])
    return title + " " + description

def build_inverted_index(df: pd.DataFrame):
    inverted_index = defaultdict(lambda: defaultdict(int))
    for doc_id, row in df.iterrows():
        text = row["Text"]
        tokens = preprocess_text(text)
        freq_counter = Counter(tokens)
        for token, freq in freq_counter.items():
            inverted_index[token][doc_id] += freq
    return dict(inverted_index)

def compute_idf(inverted_idx, total_docs):
    idf_dict = {}
    for token, posting in inverted_idx.items():
        df_t = len(posting)
        idf_dict[token] = math.log10(total_docs / df_t) if df_t else 0
    return idf_dict

def build_doc_vectors(inverted_idx, idf_dict):
    vectors = defaultdict(dict)
    for token, posting in inverted_idx.items():
        for doc_id, freq in posting.items():
            tf = 1 + math.log10(freq) if freq > 0 else 0
            vectors[doc_id][token] = tf * idf_dict[token]
    return dict(vectors)

def build_query_vector(tokens, idf_dict):
    counter = Counter(tokens)
    q_vec = {}
    for token, freq in counter.items():
        if token in idf_dict:
            tf = 1 + math.log10(freq)
            q_vec[token] = tf * idf_dict[token]
    return q_vec


def cosine_similarity(q_vec, doc_vec):
    dot = 0.0
    for token, q_val in q_vec.items():
        d_val = doc_vec.get(token, 0.0)
        dot += q_val * d_val
    norm_q = math.sqrt(sum(v*v for v in q_vec.values()))
    norm_d = math.sqrt(sum(v*v for v in doc_vec.values()))
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)

def rank_documents(query_tokens, doc_vectors, idf_dict):
    q_vec = build_query_vector(query_tokens, idf_dict)
    scores = {}
    for doc_id, doc_vec in doc_vectors.items():
        score = cosine_similarity(q_vec, doc_vec)
        scores[doc_id] = score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

@app.on_event("startup")
def startup_event():
    global df, inverted_index, idf, doc_vectors

    # Load dataset
    df = pd.read_csv("jobs_dataset.csv")
    
    # Combine text
    df["Text"] = df.apply(combine_text, axis=1)
    
    # Build inverted index
    inverted_index = build_inverted_index(df)
    
    # Compute IDF
    N = len(df)
    idf = compute_idf(inverted_index, N)
    
    # Build doc vectors
    doc_vectors = build_doc_vectors(inverted_index, idf)
    
    print("TF-IDF model built successfully.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Job API with direct search results!"}

@app.get("/search")
def search(query: str):
    """
    1. Preprocess query
    2. Rank documents using TF-IDF + Cosine Similarity
    3. Return top results (with snippet, etc.)
    """
    query_tokens = preprocess_text(query)
    ranked_docs = rank_documents(query_tokens, doc_vectors, idf)

    # Let's just return the top 5
    top_k = 5
    results = []
    for doc_id, score in ranked_docs[:top_k]:
        job_title = df.loc[doc_id, "Job Title"]
        snippet = str(df.loc[doc_id, "Job Description"])[:150]
        results.append({
            "doc_id": doc_id,
            "score": score,
            "title": job_title,
            "snippet": snippet
        })

    return {
        "query": query,
        "tokens": query_tokens,
        "results_count": len(ranked_docs),
        "top_results": results
    }

@app.get("/docs")
def get_docs(ids: List[int] = Query(...)):
    """
    An optional endpoint to retrieve multiple docs by ID,
    returning (title + snippet). 
    """
    output = []
    for doc_id in ids:
        if doc_id in df.index:
            title = df.loc[doc_id, "Job Title"]
            snippet = str(df.loc[doc_id, "Job Description"])[:150]
            output.append({
                "doc_id": doc_id,
                "title": title,
                "snippet": snippet
            })
        else:
            output.append({"doc_id": doc_id, "error": "Invalid doc_id"})
    return output

@app.get("/docs/{doc_id}")
def get_doc_by_id(doc_id: int):
    """
    Return full details for a single doc.
    """
    if doc_id in df.index:
        title = df.loc[doc_id, "Job Title"]
        description = df.loc[doc_id, "Job Description"]
        return {
            "doc_id": doc_id,
            "title": title,
            "description": description
        }
    else:
        return {
            "doc_id": doc_id,
            "error": "Invalid doc_id"
        }
