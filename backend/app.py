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
smart_schemes = {
    "ltc": ("logarithmic_tf", "idf", "cosine_normalization"),
    "lnc": ("logarithmic_tf", "none", "cosine_normalization"),
    "ntc": ("natural_tf", "idf", "cosine_normalization"),
    "anc": ("augmented_tf", "none", "cosine_normalization"),
    # Add other SMART combinations as needed
}
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


def apply_smart_scheme(tf, idf, scheme):
    """
    Apply SMART weighting scheme to compute term weighting.
    """
    tf_scheme, idf_scheme, normalization_scheme = smart_schemes[scheme]

    # Term Frequency (TF) transformation
    if tf_scheme == "logarithmic_tf":
        tf = 1 + math.log(tf)
    elif tf_scheme == "natural_tf":
        tf = tf  # No change
    elif tf_scheme == "augmented_tf":
        tf = 0.5 + (0.5 * tf / max(tf, 1))
    elif tf_scheme == "boolean_tf":
        tf = 1 if tf > 0 else 0

    # Inverse Document Frequency (IDF) transformation
    if idf_scheme == "idf":
        tf *= idf
    elif idf_scheme == "prob_idf":
        tf *= max(0, math.log(len(inverted_index) / idf))

    # Normalization is handled separately later
    return tf


# ----------------------
# QUERY PROCESSING
# ----------------------
def process_query(query, idf, query_scheme):
    """
    Process a query with the given SMART weighting scheme.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")


    query = clean_text(query)


    query = remove_stopwords(query)


    tokens = word_tokenize(query)


    pos_tags = apply_pos_tagging(tokens)


    lemmatized_tokens = apply_lemmatization_on_tokens(pos_tags)

    query_tf = Counter(lemmatized_tokens)

    query_vector = {}
    for term, tf in query_tf.items():
        if term in idf:
            query_vector[term] = apply_smart_scheme(tf, idf[term], query_scheme)

    return query_vector

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
def cosine_similarity(query_vector, doc_scheme):
    """
    Compute cosine similarity between the query vector and the document vectors
    using different SMART weighting schemes for query and documents.
    """
    scores = defaultdict(float)

    for term, weight in query_vector.items():
        if term in inverted_index:
            for doc, doc_tf in inverted_index[term].items():
                doc_weight = apply_smart_scheme(doc_tf, idf[term], doc_scheme)
                scores[doc] += weight * doc_weight

    # Normalize scores using cosine normalization
    for doc in scores:
        scores[doc] /= doc_norms[doc]

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ----------------------
# FASTAPI EVENTS & ENDPOINTS
# ----------------------
@app.on_event("startup")
def startup_event():
    global articles, inverted_index, idf, doc_norms

    # 1. Load the articles from JSON
    with open("./articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to dict: { "doc_number": {"url":..., "content":...}, ... }
    for article in data:
        doc_str = str(article["doc_number"])
        articles[doc_str] = {
            "url": article["url"],
            "content": article["content"]
        }

    # 2. Load the pre-built inverted index (currently raw freq)
    with open("./inverted_index.json", "r", encoding="utf-8") as f:
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
def search(query: str,query_scheme : str = "ltc",doc_scheme : str = "lnc",  top_k: int = 5):
    """
    1. process_query => build query TF-IDF vector
    2. cosine_similarity => get doc scores
    3. Return top_k docs
    """
    query_vector = process_query(query, idf, query_scheme)
    print(query_vector)
    results = cosine_similarity(query_vector,doc_scheme)

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
    print(top_results)
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
