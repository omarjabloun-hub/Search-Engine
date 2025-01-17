
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

def clean_text(text):
    """
    Clean and preprocess text data.
    This function performs several cleaning operations:
    - Lowercases the text (Case Folding)
    - Removes punctuation, replacing hyphens with space
    - Removes numbers
    - Removes newline characters
    - Removes underscores
    - Removes lone characters (length 1 words)
    - Removes leading and trailing spaces

    Parameters:
    text (str): A string containing text data.
    Returns:
    str: A cleaned text string.
    """
    if not isinstance(text, str):
        return text

    # Lowercase the text
    text = text.lower()

    # Replace hyphens with space
    text = re.sub(r'-', ' ', text)

    # Remove underscores
    text = re.sub(r'_', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d', '', text)

    # Remove newlines
    text = re.sub(r'\n', ' ', text)

    # Remove lone characters (length 1 words)
    text = re.sub(r'\b\w{1}\b', '', text)

    # Remove all types of parentheses and their content
    text = re.sub(r'[\(\)\{\}\[\]\<\>]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    """
    Remove stopwords from text data.

    This function filters out common stopwords from the text data.
    Stopwords are removed based on the NLTK's English stopwords list.

    Parameters:
    text (str): A string containing text data.

    Returns:
    str: A string with stopwords removed.
    """
    if not isinstance(text, str):
        return text

    filtered_text = " ".join(word for word in text.split() if word not in stop_words)
    return filtered_text

def apply_pos_tagging(tokens):
    """
    Apply POS tagging to tokenized text.

    Parameters:
    tokens (list): A list of tokenized words.

    Returns:
    list: A list of tuples, each containing a token and its corresponding POS tag.
    """
    pos_tags = pos_tag(tokens)

    return pos_tags

lemmatizer = WordNetLemmatizer()

def apply_lemmatization_on_tokens(tokens_with_pos):
    """
    Apply lemmatization to a list of tokenized words using NLTK's WordNetLemmatizer,
    considering their POS tags.

    Parameters:
    tokens_with_pos (list): A list of tuples containing token and its POS tag.

    Returns:
    list: A list of lemmatized words.
    """
    lemmatized_tokens = []

    for word, tag in tokens_with_pos:
        # Convert NLTK POS tags to WordNet POS tags for lemmatizer
        if tag.startswith('J'):  # Adjectives (JJ, JJR, JJS)
            pos = 'a'
        elif tag.startswith('V'):  # Verbs (VB, VBD, VBG, VBN, VBP, VBZ)
            pos = 'v'
        elif tag.startswith('N'):  # Nouns (NN, NNS, NNP, NNPS)
            pos = 'n'
        elif tag.startswith('R'):  # Adverbs (RB, RBR, RBS)
            pos = 'r'
        else:
            pos = 'n'  # Default to noun if unsure


        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos))

    return lemmatized_tokens

def main():
    """
    Main function to create and save the inverted index.
    
    This function:
    1. Loads the Wikipedia articles
    2. Processes each document (cleaning, tokenization, etc.)
    3. Creates the inverted index
    4. Saves the index and related data structures with timestamps
    """

    # Load articles from JSON file
    with open('./articles.json', 'r', encoding='utf-8') as f:
        df = json.load(f)

    df["cleaned_content"] = df["content"].apply(clean_text)
    df["filtered_content"] = df["cleaned_content"].apply(remove_stopwords)
    df["tokenized_content"] = df["filtered_content"].apply(lambda x: word_tokenize(x))
    df["pos_tagging_content"] = df["tokenized_content"].apply(apply_pos_tagging)
    df["lemmatized_content"] = df["pos_tagging_content"].apply(apply_lemmatization_on_tokens)
    

    file_path = f"./preprocessed_articles.csv"
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")

if __name__ == "__main__":
    main()

