import math
from typing import Dict, List, Set, Tuple

class InvertedIndex:
    def __init__(self):
        self.doc_names: List[str] = []
        self.doc_lengths: List[int] = []
        self.postings: Dict[str, Dict[int, int]] = {}
        self.vocabulary: Set[str] = set()

    def add_document(self, doc_name: str, tokens: List[str]) -> int:
        doc_id = len(self.doc_names)
        self.doc_names.append(doc_name)
        self.doc_lengths.append(len(tokens))
        
        term_counts: Dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1
            self.vocabulary.add(token)
            
        for term, count in term_counts.items():
            if term not in self.postings:
                self.postings[term] = {}
            self.postings[term][doc_id] = count
            
        return doc_id

    @property
    def total_docs(self) -> int:
        return len(self.doc_names)

    def get_term_doc_freq(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def get_term_freq_in_doc(self, term: str, doc_id: int) -> int:
        return self.postings.get(term, {}).get(doc_id, 0)
