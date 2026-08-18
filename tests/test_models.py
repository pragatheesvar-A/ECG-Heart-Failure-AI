import torch
from deephhf.models import ECGEncoder, RecordingTransformer


def test_encoder_shapes():
    model = ECGEncoder([8, 16, 24, 32], embedding_dim=20)
    logits, embedding = model(torch.randn(3, 1, 3840))
    assert logits.shape == (3,)
    assert embedding.shape == (3, 20)


def test_transformer_shapes_and_attention():
    model = RecordingTransformer(embedding_dim=20, heads=4, layers=1, ff_dim=32, max_windows=10)
    score = model(torch.randn(2, 6, 20))
    assert score.shape == (2,)
    score.sum().backward()
    assert model.gradient_attention().shape == (2, 6)
