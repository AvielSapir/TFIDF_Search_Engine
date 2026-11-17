import os
import argparse
from preprocessor import TextPreprocessor
from indexer import InvertedIndex
from engine import SearchEngine

def index_directory(directory: str) -> SearchEngine:
    preprocessor = TextPreprocessor()
    index = InvertedIndex()
    
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                tokens = preprocessor.preprocess(content)
                index.add_document(filename, tokens)
                
    return SearchEngine(index)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="documents")
    parser.add_argument("--query", default=None)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    if not os.path.isdir(args.docs):
        print(f"[ERROR] Directory '{args.docs}' not found.")
        return

    engine = index_directory(args.docs)
    preprocessor = TextPreprocessor()

    if args.query:
        tokens = preprocessor.preprocess(args.query)
        results = engine.query(tokens, top_k=args.top)
        print(f"Results for query: '{args.query}'")
        for doc_name, score, breakdown in results:
            print(f"  * {doc_name}: Cosine Similarity = {score:.4f}")
            print(f"    Terms: {breakdown}")
        return

    try:
        while True:
            q = input("\nSearch Query (or 'exit'): ")
            if q.strip().lower() == "exit":
                break
            if not q.strip():
                continue
            tokens = preprocessor.preprocess(q)
            results = engine.query(tokens, top_k=args.top)
            if not results:
                print("No relevant documents found.")
            else:
                for doc_name, score, breakdown in results:
                    print(f"  * {doc_name}: Cosine Similarity = {score:.4f}")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
