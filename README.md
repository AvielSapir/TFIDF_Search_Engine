# TFIDF_Search_Engine

A zero-dependency Python search engine built on the Vector Space Model (VSM). It indexes `.txt` documents into an inverted index, runs a lightweight NLP pipeline, and ranks search results using **TF-IDF** weighting and **Cosine Similarity**.

## How it works

1. **Text preprocessing (`preprocessor.py`)** – Lowercases text, tokenizes with regex, removes common English stopwords, and normalizes words with a custom lightweight Porter Stemmer.
2. **Inverted index (`indexer.py`)** – Maps each stemmed term to its posting list (`doc_id -> frequency`) and tracks document frequencies ($\text{DF}$) so queries only touch relevant candidate documents.
3. **Scoring & ranking (`engine.py`)** – Precomputes document $L_2$ norms at index time and ranks matching documents for any query.

### Scoring math

To prevent long or repetitive documents from dominating results, term frequency uses logarithmic scaling (for $f_{t, d} > 0$):

$$
\text{TF}(t, d) = 1 + \ln(f_{t, d})
$$

Rare terms carry more weight via smooth Inverse Document Frequency, where $N$ is the total number of documents and $\text{DF}(t)$ is the number of documents containing term $t$:

$$
\text{IDF}(t) = \ln\left(\frac{N + 1}{\text{DF}(t) + 1}\right) + 1
$$

Query and document vectors are compared using **Cosine Similarity** over their shared terms:

$$
\text{Sim}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \, \|\vec{d}\|_2} = \frac{\sum_{t \in q \cap d} w_{t, q} \cdot w_{t, d}}{\sqrt{\sum_{t \in q} w_{t, q}^2} \cdot \sqrt{\sum_{t \in d} w_{t, d}^2}}
$$

## Quick start

Run the test suite:
```bash
python test_tfidf.py
```

Run a single query against a folder of `.txt` files:
```bash
python main.py --docs path/to/documents --query "your search query" --top 5
```

Or launch interactive search mode:
```bash
python main.py --docs path/to/documents
```

## Project structure

- `main.py` – CLI for directory indexing, single queries, and interactive search
- `preprocessor.py` – Regex tokenizer, stopword filter, and Porter Stemmer
- `indexer.py` – `InvertedIndex` and postings list management
- `engine.py` – `MathEngine` (TF-IDF, cosine similarity) and `SearchEngine`
- `test_tfidf.py` – Unit tests for preprocessing, indexing, and ranking
