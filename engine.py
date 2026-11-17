import math
from typing import Dict, List, Tuple
from indexer import InvertedIndex

class MathEngine:
    @staticmethod
    def compute_tf(raw_count: int, doc_len: int) -> float:
        if raw_count == 0 or doc_len == 0:
            return 0.0
        return 1.0 + math.log(raw_count)

    @staticmethod
    def compute_smooth_idf(doc_freq: int, total_docs: int) -> float:
        return math.log((total_docs + 1.0) / (doc_freq + 1.0)) + 1.0

    @staticmethod
    def cosine_similarity(query_weights: Dict[str, float], doc_weights: Dict[str, float], doc_norm: float) -> float:
        if not query_weights or not doc_weights or doc_norm == 0.0:
            return 0.0
            
        dot_product = 0.0
        query_norm_sq = 0.0
        
        for term, q_val in query_weights.items():
            query_norm_sq += q_val ** 2
            if term in doc_weights:
                dot_product += q_val * doc_weights[term]
                
        query_norm = math.sqrt(query_norm_sq)
        if query_norm == 0.0:
            return 0.0
            
        return dot_product / (query_norm * doc_norm)

class SearchEngine:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.doc_weights: Dict[int, Dict[str, float]] = {}
        self.doc_norms: Dict[int, float] = {}
        self._build_model()

    def _build_model(self):
        total_docs = self.index.total_docs
        for doc_id in range(total_docs):
            self.doc_weights[doc_id] = {}
            norm_sq = 0.0
            doc_len = self.index.doc_lengths[doc_id]
            
            for term in self.index.vocabulary:
                raw_count = self.index.get_term_freq_in_doc(term, doc_id)
                if raw_count > 0:
                    tf = MathEngine.compute_tf(raw_count, doc_len)
                    df = self.index.get_term_doc_freq(term)
                    idf = MathEngine.compute_smooth_idf(df, total_docs)
                    weight = tf * idf
                    self.doc_weights[doc_id][term] = weight
                    norm_sq += weight ** 2
                    
            self.doc_norms[doc_id] = math.sqrt(norm_sq)

    def query(self, query_tokens: List[str], top_k: int = 5) -> List[Tuple[str, float, Dict[str, float]]]:
        if not query_tokens:
            return []
            
        total_docs = self.index.total_docs
        query_term_counts: Dict[str, int] = {}
        for token in query_tokens:
            query_term_counts[token] = query_term_counts.get(token, 0) + 1
            
        query_weights: Dict[str, float] = {}
        for term, count in query_term_counts.items():
            tf = 1.0 + math.log(count)
            df = self.index.get_term_doc_freq(term)
            idf = MathEngine.compute_smooth_idf(df, total_docs)
            query_weights[term] = tf * idf

        candidate_doc_ids = set()
        for term in query_weights:
            if term in self.index.postings:
                candidate_doc_ids.update(self.index.postings[term].keys())

        results = []
        for doc_id in candidate_doc_ids:
            doc_w = self.doc_weights[doc_id]
            doc_norm = self.doc_norms[doc_id]
            score = MathEngine.cosine_similarity(query_weights, doc_w, doc_norm)
            if score > 0.0:
                breakdown = {t: query_weights[t] * doc_w.get(t, 0.0) for t in query_weights if t in doc_w}
                results.append((self.index.doc_names[doc_id], score, breakdown))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
