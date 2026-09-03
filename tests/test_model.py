import torch

from src.model import AttentionHead, MiniTransformer


def test_transformer_output_shape():
    model = MiniTransformer(
        vocab_size=20,
        embedding_size=16,
        num_heads=4,
        num_layers=2,
        block_size=8,
        dropout=0.0
    )

    inputs = torch.randint(0, 20, (2, 8))
    targets = torch.randint(0, 20, (2, 8))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, 20)
    assert torch.isfinite(loss)


def test_causal_mask_blocks_future_tokens():
    head = AttentionHead(
        embedding_size=16,
        head_size=4,
        block_size=8
    )

    inputs = torch.randn(2, 8, 16)
    _, weights = head(inputs, return_weights=True)

    future_positions = torch.triu(
        torch.ones(8, 8, dtype=torch.bool),
        diagonal=1
    )

    assert torch.all(
        weights[:, future_positions] == 0
    )


def test_text_generation_length():
    model = MiniTransformer(
        vocab_size=20,
        embedding_size=16,
        num_heads=4,
        num_layers=2,
        block_size=8,
        dropout=0.0
    )

    starting_tokens = torch.tensor([[1]])
    output = model.generate(
        starting_tokens,
        max_new_tokens=10
    )

    assert output.shape == (1, 11)