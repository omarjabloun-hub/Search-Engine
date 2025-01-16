from fastapi import FastAPI, Query
from typing import List
import math
import nltk
import json
import re
from collections import defaultdict, Counter
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download these once if you haven't already
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = FastAPI()

# ----------------------
# GLOBAL VARIABLES
# ----------------------
articles = {}         # { doc_number(str): { "url": "...", "content": "..." } }
inverted_index = {}   # { token: { doc_number(str): raw_freq OR tf-idf } }
idf = {}              # { token: idf_value }
doc_norms = {}        # { doc_number(str): sqrt of sum of TF-IDF^2 for that doc }

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# ----------------------
# TEXT CLEANING FUNCTIONS
# ----------------------
def clean_text(text: str) -> str:
    """
    1) Lowercase
    2) Replace hyphens with space
    3) Remove underscores
    4) Remove punctuation
    5) Remove digits
    6) Remove newlines
    7) Remove single-char words
    8) Strip extra spaces
    """
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\b\w{1}\b', '', text)
    text = text.strip()
    text = text.replace("  ", " ")
    return text

def remove_stopwords(text: str) -> str:
    """
    Remove common English stopwords using NLTK.
    """
    if not isinstance(text, str):
        return text
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

def apply_pos_tagging(tokens: List[str]) -> List[tuple]:
    """
    Apply NLTK POS tagging to a list of tokens.
    """
    return pos_tag(tokens)

def apply_lemmatization_on_tokens(tokens_with_pos: List[tuple]) -> List[str]:
    """
    Lemmatize tokens according to their POS tag.
    """
    lemmatized = []
    for word, tag in tokens_with_pos:
        # Map NLTK POS tags to WordNet POS
        if tag.startswith('J'):
            wn_pos = 'a'  # adjective
        elif tag.startswith('V'):
            wn_pos = 'v'  # verb
        elif tag.startswith('N'):
            wn_pos = 'n'  # noun
        elif tag.startswith('R'):
            wn_pos = 'r'  # adverb
        else:
            wn_pos = 'n'  # default to noun
        lemmatized.append(lemmatizer.lemmatize(word, wn_pos))
    return lemmatized

# ----------------------
# QUERY PROCESSING
# ----------------------
def process_query(query: str, idf_dict: dict, query_scheme : str = "ltc") -> dict:
    """
    1) Clean and remove stopwords
    2) Tokenize
    3) POS tag + lemmatize
    4) Build a TF-IDF vector for the query
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

    # 1) Clean + remove stopwords
    query = clean_text(query)
    query = remove_stopwords(query)
    print(query)
    # 2) Tokenize
    tokens = word_tokenize(query)

    # 3) POS tagging + lemmatization
    pos_tags = apply_pos_tagging(tokens)
    lemmatized = apply_lemmatization_on_tokens(pos_tags)
    print(lemmatized)
    # 4) Build query TF-IDF
    freq_counter = Counter(lemmatized)
    q_vector = {}
    print(idf_dict)
    for term, freq in freq_counter.items():
        if term in idf_dict:
            # TF = 1 + log(freq)
            tf_val = 1 + math.log(freq, 10)
            # Multiply by IDF
            q_vector[term] = tf_val * idf_dict[term]

    return q_vector

# ----------------------
# IDF & DOC WEIGHTS
# ----------------------
def compute_idf(inverted_idx: dict, total_docs: int) -> dict:
    """
    IDF(token) = log10( total_docs / doc_freq )
    """
    idf_dict = {}
    for token, posting in inverted_idx.items():
        df_t = len(posting)  # how many docs contain this token
        if df_t > 0:
            idf_dict[token] = math.log10(total_docs / df_t)
        else:
            idf_dict[token] = 0
    return idf_dict

# ----------------------
# COSINE SIMILARITY
# ----------------------
def cosine_similarity(query_vector: dict) -> List[tuple]:
    """
    Score all docs by:
        sum_{t in query} [ (query TF-IDF) * (doc TF-IDF) ] / doc_norm
    We do not divide by query norm here, but you can if you want.

    doc TF-IDF was precomputed and stored in the global inverted_index (overwritten).
    doc_norms[doc] = sqrt( sum of doc's (TF-IDF)^2 ).
    """
    scores = defaultdict(float)

    # 1) Accumulate dot product for each doc
    for term, q_weight in query_vector.items():
        # If term is in the inverted_index, then we have doc -> TF-IDF
        if term in inverted_index:
            for doc_id, doc_weight in inverted_index[term].items():
                scores[doc_id] += q_weight * doc_weight

    # 2) Normalize by doc_norm
    #    doc_norms[doc_id] was computed at startup
    for doc_id in scores:
        denominator = doc_norms.get(doc_id, 1e-9)
        scores[doc_id] /= denominator

    # 3) Sort by descending score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

# ----------------------
# FASTAPI EVENTS & ENDPOINTS
# ----------------------
@app.on_event("startup")
def startup_event():
    global articles, inverted_index, idf, doc_norms

    # 1. Load the articles from JSON
    with open("articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to dict: { "doc_number": {"url":..., "content":...}, ... }
    for article in data:
        doc_str = str(article["doc_number"])
        articles[doc_str] = {
            "url": article["url"],
            "content": article["content"]
        }

    # 2. Load the pre-built inverted index (currently raw freq)
    with open("inverted_index.json", "r", encoding="utf-8") as f:
        inv_idx = json.load(f)

    # 3. Compute IDF for each token
    N = len(articles)
    print(len(inv_idx))
    local_idf = compute_idf(inv_idx, N)

    # 4. Convert raw freq -> TF-IDF in the inverted index,
    #    and compute doc_norms = sqrt( sum of (doc TF-IDF)^2 ).
    local_doc_norms = defaultdict(float)
    for token, posting in inv_idx.items():
        for doc_id, raw_freq in posting.items():
            # TF
            tf_val = 1 + math.log(raw_freq, 10) if raw_freq > 0 else 0
            # TF-IDF
            tf_idf_val = tf_val * local_idf[token]

            # Overwrite the posting with TF-IDF, no longer just freq
            posting[doc_id] = tf_idf_val

            # Accumulate square for norm
            local_doc_norms[doc_id] += (tf_idf_val ** 2)

    # Now take sqrt to finalize doc_norms
    for doc_id, accum in local_doc_norms.items():
        local_doc_norms[doc_id] = math.sqrt(accum)

    # 5. Assign back to global variables
    inverted_index = inv_idx     # now it holds TF-IDF instead of raw freq
    idf = local_idf
    doc_norms = dict(local_doc_norms)

    print("TF-IDF model built successfully from pre-built inverted index.")

@app.get("/")
def read_root():
    return {"message": "Wikipedia-based TF-IDF model is ready!"}

@app.get("/search")
def search(query: str,query_scheme : str = "ltc",  top_k: int = 5):
    """
    1. process_query => build query TF-IDF vector
    2. cosine_similarity => get doc scores
    3. Return top_k docs
    """
    query_vector = process_query(query, idf, query_scheme)
    print(query_vector)
    results = cosine_similarity(query_vector)

    top_results = []
    for doc_num, score in results[:top_k]:
        article_info = articles.get(doc_num, {})
        url = article_info.get("url", "")
        content = article_info.get("content", "")
        top_results.append({
            "doc_number": doc_num,
            "score": score,
            "url": url,
            "snippet": content[:150]  # first 150 chars
        })

    return {
        "query": query,
        "tokens": query_vector,
        "results_count": len(results),
        "top_results": top_results
    }

@app.get("/doc/{doc_num}")
def get_full_doc(doc_num: str):
    """
    Return the full content for a single document by doc_number.
    """
    article_info = articles.get(doc_num)
    if article_info:
        return {
            "doc_number": doc_num,
            "url": article_info["url"],
            "content": article_info["content"]
        }
    else:
        return {"error": f"No article found with doc_number={doc_num}"}
