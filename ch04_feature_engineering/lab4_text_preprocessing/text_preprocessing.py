import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def tokenize_text(text, method='split'):
    if method == 'split':
        return text.split()
    elif method == 'regex':
        return re.findall(r'\b\w+\b', text)
    elif method == 'nltk':
        return nltk.word_tokenize(text)
    return text.split()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def stem_text(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words]

def build_tfidf_matrix(documents, max_features=None):
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer

def build_count_matrix(documents, max_features=None):
    vectorizer = CountVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer

def preprocess_pipeline(documents):
    processed = []
    for doc in documents:
        norm = normalize_text(doc)
        tokens = tokenize_text(norm, 'regex')
        no_stop = remove_stopwords(tokens)
        lemmas = lemmatize_text(no_stop)
        processed.append(" ".join(lemmas))
    return processed

def build_simple_embeddings(documents, dim):
    matrix, vectorizer = build_tfidf_matrix(documents)
    n_components = min(dim, matrix.shape[1] - 1) if matrix.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(matrix)
    return embeddings
