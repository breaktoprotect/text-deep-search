from services.sbert_engine import sbert_model_registry


def test_list_supported_models():
    models = sbert_model_registry.list_supported_models()
    assert isinstance(models, list)
    assert "MiniLM-L6-v2" in models
    assert "MPNet-base-v2" in models


def test_default_model_id_in_supported_models():
    default_model = sbert_model_registry.DEFAULT_MODEL_ID
    assert default_model in sbert_model_registry.SUPPORTED_MODELS
