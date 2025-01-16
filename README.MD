Information Retrieval Project: Vector Space Model
=================================================

Table of Contents
-----------------

*   [1\. Introduction](#1-introduction)
    
*   [2\. Project Structure](#2-project-structure)
    
*   [3\. Requirements](#3-requirements)
    
*   [4\. Data Preparation](#4-data-preparation)
    
*   [5\. Preprocessing and Inverted Index](#5-preprocessing-and-inverted-index)
    
*   [6\. Building TF-IDF Vectors](#6-building-tf-idf-vectors)
    
*   [7\. Querying: Vector Space Model](#7-querying-vector-space-model)
    
*   [8\. FastAPI Endpoints](#8-fastapi-endpoints)
    
    *   [8.1 /search (Vector Space Model)](#81-search-vector-space-model)
        
    *   [8.2 /docs (Multiple IDs)](#82-docs-multiple-ids)
        
    *   [8.3 /docs/{doc\_id} (Single Document)](#83-docsdoc_id-single-document)
        
*   [9\. How to Run](#9-how-to-run)
    
*   [10\. Example Usage](#10-example-usage)
    
*   [11\. Future Improvements](#11-future-improvements)
    
*   [12\. Conclusion](#12-conclusion)
    

1\. Introduction
----------------

This project implements a **Vector Space Model** for document retrieval using **TF-IDF** scoring and **cosine similarity**. We use **FastAPI** to expose search endpoints. The main goals:

1.  **Preprocess** and store a dataset of job descriptions.
    
2.  **Build** an inverted index, then compute **TF-IDF** vectors for each document.
    
3.  **Allow** users to **search** for terms and receive the most relevant job descriptions, ranked by cosine similarity.
    

2\. Project Structure
---------------------

A typical folder layout might look like:

```bash
 project/  
 ├── app.py                # FastAPI application (endpoints, logic)  
 ├── jobs_dataset.csv      # CSV dataset (Job Title, Job Description)  
 ├── requirements.txt      # Dependencies  
 ├── README.md             # Project documentation (this file)  
 └── venv/ (optional)      # Virtual environment   
 ```
3\. Requirements
----------------

*   **Python 3.7+**
    
*   **FastAPI** (for the web service)
    
*   **Uvicorn** (ASGI server to run FastAPI)
    
*   **NLTK** (for text processing)
    
*   **Pandas** (for CSV data handling)
    
Install with:
```bash
pip install -r requirements.txt
```

4\. Data Preparation
--------------------

We have a CSV file named jobs\_dataset.csv containing at least two columns:

*   **Job Title**
    
*   **Job Description**
    

Each row represents one “document.”

Example row:


| Job Title          | Job Description                                            |
|--------------------|------------------------------------------------------------|
| Software Engineer  | We are looking for a Software Engineer with strong Python… |


We combine **Job Title** and **Job Description** into a single text for indexing.

5\. Preprocessing and Inverted Index
------------------------------------
# 5. Preprocessing and Inverted Index

**Preprocessing** transforms raw text into a normalized form to improve retrieval accuracy. The common steps are:

1. **Lowercase** the text to reduce case-variation (e.g., “Engineer” vs. “engineer”).
2. **Remove non-alphanumeric characters** (punctuation, symbols) to keep only letters/numbers/whitespace.
3. **Tokenize** the text, splitting sentences into individual words or tokens.
4. **Stopword removal** to drop common words (like “the”, “and”, etc.) that don’t add contextual meaning.
5. **Stemming** (e.g., with the PorterStemmer) or **lemmatization** to reduce words to their base form.

---

### Inverted Index

Once the text is preprocessed, we build the **inverted index**, which maps each token to the documents where it appears, along with frequencies or positions. A common structure is:


```json  
{ 
    "token_1": { 
        "doc_id_1": frequency_in_that_doc, 
        "doc_id_2": frequency_in_that_doc, ... 
        },
    "token_2": {
        "doc_id_3": frequency_in_that_doc, 
        ... 
        },
    ...
}  
```


- **Token**: a stemmed/lowercased word from the text.
- **doc_id**: an identifier for each document (e.g., row index in a DataFrame).
- **frequency**: how many times that token appears in the doc.

This data structure provides quick lookups of which documents contain a given token, enabling more efficient retrieval algorithms later (like TF-IDF scoring or vector space ranking).


6\. Building TF-IDF Vectors
---------------------------


### 6.1 IDF Calculation

\[
    \begin{aligned}
  IDF(t) = \log_{10}\Bigl(\frac{N}{\text{df}(t)}\Bigr)
    \end{aligned}
\]


idf(t)=log⁡10(Ndf(t)) \\text{idf}(t) = \\log\_{10}\\left(\\frac{N}{\\text{df}(t)}\\right)idf(t)=log10​(df(t)N​)

Where:

*   NNN is total number of documents.
    
*   df(t)\\text{df}(t)df(t) is number of docs containing token ttt.
    

### 6.2 Document Vectors

For each (token, doc\_id, freq) in the inverted index:

tf-idf(t,d)=(1+log⁡10(freq))×idf(t) \\text{tf-idf}(t, d) = (1 + \\log\_{10}(\\text{freq})) \\times \\text{idf}(t)tf-idf(t,d)=(1+log10​(freq))×idf(t)

Store the results in a structure like:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopydoc_vectors = {    doc_id: {      token: tf_idf_value,      ...    },    ...  }   `

7\. Querying: Vector Space Model
--------------------------------

When a user enters a query (e.g. "software engineer"):

1.  **Preprocess** the query tokens (same steps: lowercasing, stopwords, stemming).
    
2.  **Compute TF-IDF** for each query token (using the same IDF values).
    
3.  **Cosine Similarity**: compare query vector to each doc\_vector.
    

cos⁡(q⃗,d⃗)=∑t(qt×dt)∑t(qt2)×∑t(dt2) \\cos(\\vec{q}, \\vec{d}) = \\frac{\\sum\_{t}(q\_t \\times d\_t)}{\\sqrt{\\sum\_{t}(q\_t^2)} \\times \\sqrt{\\sum\_{t}(d\_t^2)}}cos(q​,d)=∑t​(qt2​)​×∑t​(dt2​)​∑t​(qt​×dt​)​

Finally, sort documents by descending similarity.

8\. FastAPI Endpoints
---------------------

### 8.1 /search (Vector Space Model)

*   **Method**: GET
    
*   **Query Parameter**: query (string)
    
*   **Steps**:
    
    1.  Preprocess query.
        
    2.  Build TF-IDF query\_vector.
        
    3.  Compute cosine similarity for each document.
        
    4.  Sort and return top results.
        
*   **Response**: JSON array of top documents, each with:
    
    *   doc\_id
        
    *   score (cosine similarity)
        
    *   title
        
    *   snippet (first 150 chars)
        

**Example**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   sqlCopyGET /search?query=software+engineer   `

**Response (JSON)**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   jsonCopy{    "query": "software engineer",    "tokens": ["softwar", "engin"],    "results_count": 12,    "top_results": [      {        "doc_id": 0,        "score": 0.556,        "title": "Software Engineer",        "snippet": "We are looking for a Software Engineer..."      },      ...    ]  }   `

### 8.2 /docs (Multiple IDs)

*   **Method**: GET
    
*   **Query Parameter**: ids (list of integers)
    
*   **Returns**: a JSON list, each element with doc\_id, title, snippet.
    

**Example**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyGET /docs?ids=0&ids=2&ids=5   `

### 8.3 /docs/{doc\_id} (Single Document)

*   **Method**: GET
    
*   **Path Parameter**: doc\_id
    
*   **Returns**: full title and description.
    

**Example**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyGET /docs/5   `

9\. How to Run
--------------

1.  bashCopypip install -r requirements.txt
    
2.  bashCopyuvicorn app:app --reload
    
3.  **Open** a browser at http://127.0.0.1:8000 for a welcome message.
    
4.  **Swagger UI** is at http://127.0.0.1:8000/docs where you can interact with the endpoints.
    

10\. Example Usage
------------------

### 10.1 Searching

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   sqlCopyGET /search?query=data+scientist   `

Possible response:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   jsonCopy{    "query": "data scientist",    "tokens": ["data", "scientist"],    "results_count": 10,    "top_results": [      {        "doc_id": 3,        "score": 0.8132,        "title": "Data Scientist",        "snippet": "We are seeking a Data Scientist to build..."      },      {        "doc_id": 7,        "score": 0.4950,        "title": "Senior Data Engineer",        "snippet": "Candidate will handle large scale data pipelines..."      },      ...    ]  }   `

### 10.2 Retrieving Multiple Docs by ID

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyGET /docs?ids=0&ids=1&ids=4   `

JSON response:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   jsonCopy[    {      "doc_id": 0,      "title": "Software Engineer",      "snippet": "We are looking for a Software Engineer..."    },    {      "doc_id": 1,      "title": "Data Scientist",      "snippet": "We are seeking a Data Scientist..."    },    {      "doc_id": 4,      "error": "Invalid doc_id"    }  ]   `

### 10.3 Full Description for a Single Doc

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyGET /docs/2   `

Response:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   jsonCopy{    "doc_id": 2,    "title": "DevOps Engineer",    "description": "Responsible for CI/CD pipelines, infrastructure automation..."  }   `

11\. Future Improvements
------------------------

1.  **Advanced Ranking**: Instead of basic TF-IDF, explore **BM25** or other IR models.
    
2.  **Query Expansion**: Use synonyms or phrase detection to improve relevance.
    
3.  **Phrase Queries**: Modify indexing to handle multi-word “phrases” exactly.
    
4.  **Pagination**: Return only the top 10 or 20, with optional page and size parameters.
    
5.  **GUI**: Build a simple front-end or incorporate a React/Vue/Angular UI to query the API.
    

12\. Conclusion
---------------

We have implemented a **Vector Space Model** using **TF-IDF** weighting and **cosine similarity** in a **FastAPI** application. This allows users to **search** job postings in a dataset and retrieve the most relevant results. The steps outlined—from data loading and preprocessing to serving results via REST endpoints—provide a **foundation** for further enhancements in Information Retrieval systems.


import nltk




nltk.download('punkt_tab')