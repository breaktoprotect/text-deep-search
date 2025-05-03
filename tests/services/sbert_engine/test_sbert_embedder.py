import pytest
import numpy as np
from services.sbert_engine import sbert_embedder


def test_list_supported_models():
    models = sbert_embedder.list_supported_models()
    assert isinstance(models, list)
    assert "MiniLM-L6-v2" in models
    assert "MPNet-base-v2" in models


def test_get_model_returns_model_instance():
    model = sbert_embedder.get_model("MiniLM-L6-v2")
    assert model is not None
    assert hasattr(model, "encode")


def test_get_model_invalid_key_raises():
    with pytest.raises(ValueError):
        sbert_embedder.get_model("non-existent-model")


def test_embed_sentences_output_shape():
    sentences = ["Enable firewall", "Disable guest login"]
    embeddings = sbert_embedder.embed_sentences(sentences, model_key="MiniLM-L6-v2")
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2  # 2 sentences
    assert embeddings.shape[1] > 0  # some embedding dimension


def test_embed_query_output_shape():
    query = "How to enforce password complexity?"
    embedding = sbert_embedder.embed_query(query, model_key="MiniLM-L6-v2")
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0  # embedding dimension
