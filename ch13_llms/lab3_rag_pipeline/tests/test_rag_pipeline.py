import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rag_pipeline import *

def test_chunking():
    text = "A B C D E F G H"
    chunks = chunk_text(text, 4, 2)
    assert len(chunks) >= 3
    assert chunks[0] == "A B C D"

def test_rag_pipeline():
    corpus = generate_sample_corpus()
    pipe = RAGPipeline()
    pipe.add_documents(corpus, chunk_size=5, overlap=1)
    
    res = pipe.query("What is Python?", top_k=2)
    assert len(res) == 2
    assert "Python" in res[0][0] or "Python" in res[1][0]
    
    acc = evaluate_retrieval(pipe, ["Python"], ["Python"])
    assert acc == 1.0
