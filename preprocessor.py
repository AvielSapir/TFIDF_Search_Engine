import re
import string
from typing import List, Set

DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

class PorterStemmerLight:
    @staticmethod
    def stem(word: str) -> str:
        if len(word) <= 3:
            return word
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif word.endswith("ss"):
            pass
        elif word.endswith("s"):
            word = word[:-1]
            
        if word.endswith("eed"):
            if len(word) > 4:
                word = word[:-1]
        elif word.endswith("ed") and len(word) > 3:
            word = word[:-2]
        elif word.endswith("ing") and len(word) > 4:
            word = word[:-3]
            
        if word.endswith("ly") and len(word) > 3:
            word = word[:-2]
            
        return word

class TextPreprocessor:
    def __init__(self, stopwords: Set[str] = None):
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self.stemmer = PorterStemmerLight()
        self.tokenizer_re = re.compile(r"\b[a-zA-Z0-9_]+\b")

    def preprocess(self, text: str) -> List[str]:
        text = text.lower()
        tokens = self.tokenizer_re.findall(text)
        processed = []
        for token in tokens:
            if token not in self.stopwords and len(token) > 1:
                stemmed = self.stemmer.stem(token)
                processed.append(stemmed)
        return processed
