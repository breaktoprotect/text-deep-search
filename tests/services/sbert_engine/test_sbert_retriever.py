import numpy as np
import pytest
from services.sbert_engine import sbert_retriever


def test_semantic_search_returns_top_k():
    # Simulate a query vector and corpus with known match pattern
    corpus_embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    query_embedding = np.array([1.0, 0.0, 0.0])  # Should match best with index 0 or 1

    results = sbert_retriever.get_top_cosine_matches(
        query_embedding, corpus_embeddings, top_k=2
    )

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0][0] in [0, 1]  # Top match must be index 0 or 1
    assert isinstance(results[0][1], float)
    assert 0.0 <= results[0][1] <= 1.0


def test_semantic_search_invalid_input_shape():
    corpus = np.random.rand(5, 3)
    query = np.random.rand(2, 3)  # Invalid: 2D query

    with pytest.raises(Exception):  # Should fail due to shape
        sbert_retriever.get_top_cosine_matches(query, corpus)
