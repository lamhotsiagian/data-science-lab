import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
        i += (chunk_size - overlap)
        if i >= len(words) or chunk_size - overlap <= 0:
            break
    return chunks

def create_tfidf_embeddings(chunks):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks).toarray()
    return vectorizer, embeddings

def build_index(embeddings):
    return embeddings

def retrieve(query, index, chunks, vectorizer, top_k=3):
    q_vec = vectorizer.transform([query]).toarray()
    sims = cosine_similarity(q_vec, index)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(chunks[i], sims[i]) for i in top_indices]

class RAGPipeline:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.index = None

    def add_documents(self, texts, chunk_size=10, overlap=2):
        for text in texts:
            self.chunks.extend(chunk_text(text, chunk_size, overlap))
        if self.chunks:
            self.vectorizer, self.index = create_tfidf_embeddings(self.chunks)

    def query(self, question, top_k=3):
        if self.index is None:
            return []
        return retrieve(question, self.index, self.chunks, self.vectorizer, top_k)

def evaluate_retrieval(pipeline, queries, expected_chunks):
    correct = 0
    for q, exp in zip(queries, expected_chunks):
        res = pipeline.query(q, top_k=3)
        if any(exp in r[0] for r in res):
            correct += 1
    return correct / len(queries) if queries else 0

def generate_sample_corpus():
    return [
        "Data science is an interdisciplinary field that uses scientific methods to extract knowledge.",
        "Machine learning algorithms build a model based on sample data.",
        "Python is a popular programming language for data science and AI.",
        "Neural networks are a series of algorithms that endeavor to recognize underlying relationships.",
        "Transformers are a deep learning architecture that relies on self-attention mechanisms."
    ]
