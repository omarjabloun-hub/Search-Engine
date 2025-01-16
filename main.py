import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from collections import defaultdict, Counter

# Download any missing NLTK data
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Lowercase the text, remove non-alphanumeric characters, tokenize,
    remove stopwords, and apply stemming.
    """
    text = text.lower()
    # Remove non-alphanumeric except whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    #tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def combine_text(row):
    """
    Concatenate the "Job Title" and "Job Description" columns
    into a single string for indexing.
    """
    title = str(row['Job Title'])
    description = str(row['Job Description'])
    return title + " " + description

def build_inverted_index(df):
    """
    Create an inverted index from the DataFrame.
    The index structure is:
        { token: { doc_id: frequency_in_that_doc, ...}, ... }
    """
    inverted_index = defaultdict(lambda: defaultdict(int))

    for doc_id, row in df.iterrows():
        text = row["Text"]
        tokens = preprocess_text(text)
        freq_counter = Counter(tokens)

        for token, freq in freq_counter.items():
            inverted_index[token][doc_id] += freq

    # Convert to regular dict before returning
    return dict(inverted_index)

def main():
    # 1. Load dataset (ensure 'jobs_dataset.csv' is in your current folder, or give the correct path)
    df = pd.read_csv("jobs_dataset.csv")

    # 2. Combine columns into one text column
    df["Text"] = df.apply(combine_text, axis=1)

    # 3. Build the inverted index
    index = build_inverted_index(df)

    # 4. Prompt the user for the search word
    user_input = input("Enter the word you want to search: ").strip()
    if not user_input:
        print("You didn't enter a search term.")
        return

    # 5. Because we use stemming, we have to stem the user’s input as well
    stemmed_search_word = user_input

    # 6. Look up the stemmed word in the index
    if stemmed_search_word in index:
        postings = index[stemmed_search_word]
        # Convert the dict items to a list
        sorted_postings = sorted(postings.items(), key=lambda x: x[1], reverse=True)

        # Slice to get the first 5 documents
        top_5_docs = sorted_postings[:5]
        print(f"\nDocuments containing the token '{stemmed_search_word}':")
        for doc_id, freq in top_5_docs:
            print(f"  doc_id={doc_id}, frequency={freq}")

            # Optionally, show snippet of the original job post
            job_title = df.loc[doc_id, "Job Title"]
            snippet = df.loc[doc_id, "Job Description"][:100]  # first 100 chars
            print(f"    Title: {job_title}")
            print(f"    Snippet: {snippet}...\n")
    else:
        print(f"\nToken '{user_input}' (stemmed as '{stemmed_search_word}') not found in index.")

if __name__ == "__main__":
    main()
