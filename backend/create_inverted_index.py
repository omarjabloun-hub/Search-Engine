import pandas as pd
import json
from collections import defaultdict

def main():
    with open('./preprocessed_articles.csv', 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)
    inverted_index = defaultdict(lambda: defaultdict(int))

    for index, row in df.iterrows():
        doc_number = row['doc_number']
        tokens = row['lemmatized_content']

        for token in tokens:
            inverted_index[token][doc_number] += 1
    inverted_index = dict(sorted(inverted_index.items()))
    index_path = "./inverted_index.json"
    with open(index_path, "w") as f:
        json.dump(inverted_index, f, indent=4)

print("Inverted index created and saved.")
if __name__ == "__main__":
    main()