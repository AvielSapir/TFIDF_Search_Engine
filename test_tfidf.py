import unittest
from preprocessor import TextPreprocessor, PorterStemmerLight
from indexer import InvertedIndex
from engine import MathEngine, SearchEngine

class TestPreprocessor(unittest.TestCase):
    def test_stemming(self):
        self.assertEqual(PorterStemmerLight.stem("running"), "runn")
        self.assertEqual(PorterStemmerLight.stem("cats"), "cat")

    def test_preprocessing_pipeline(self):
        pipe = TextPreprocessor()
        tokens = pipe.preprocess("The quick brown fox jumps over the lazy dogs!")
        self.assertNotIn("the", tokens)
        self.assertIn("fox", tokens)

class TestInvertedIndex(unittest.TestCase):
    def test_postings(self):
        idx = InvertedIndex()
        idx.add_document("doc1.txt", ["cyber", "defense", "security"])
        idx.add_document("doc2.txt", ["cyber", "intelligence"])
        self.assertEqual(idx.get_term_doc_freq("cyber"), 2)
        self.assertEqual(idx.get_term_doc_freq("security"), 1)

class TestSearchEngine(unittest.TestCase):
    def test_search_retrieval(self):
        idx = InvertedIndex()
        idx.add_document("d1.txt", ["deepmind", "artificial", "intelligence", "agent"])
        idx.add_document("d2.txt", ["cryptography", "network", "security", "packet"])
        idx.add_document("d3.txt", ["machine", "learning", "artificial", "intelligence"])
        
        engine = SearchEngine(idx)
        results = engine.query(["artificial", "intelligence"])
        self.assertGreater(len(results), 0)
        top_doc = results[0][0]
        self.assertIn(top_doc, ["d1.txt", "d3.txt"])

if __name__ == "__main__":
    unittest.main()
