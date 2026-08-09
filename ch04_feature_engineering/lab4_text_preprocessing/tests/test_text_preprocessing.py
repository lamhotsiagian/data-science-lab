import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import text_preprocessing

def test_normalization():
    text = "Hello, World!  "
    assert text_preprocessing.normalize_text(text) == "hello world"

def test_stopwords():
    tokens = ["this", "is", "a", "test"]
    assert text_preprocessing.remove_stopwords(tokens) == ["test"]

def test_tfidf_matrix():
    docs = ["hello world", "world of python"]
    matrix, vec = text_preprocessing.build_tfidf_matrix(docs)
    assert matrix.shape == (2, 4)

def test_pipeline():
    docs = ["This is a TEST!"]
    processed = text_preprocessing.preprocess_pipeline(docs)
    assert processed == ["test"]
