import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size, block_size):
        super().__init__()

        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.scale = head_size ** -0.5

    def forward(self, x, return_weights=False):
        _, sequence_length, _ = x.shape

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores * self.scale

        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        output = weights @ values

        if return_weights:
            return output, weights

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, block_size):
        super().__init__()

        head_size = embedding_size // num_heads

        self.heads = nn.ModuleList([
            AttentionHead(embedding_size, head_size, block_size)
            for _ in range(num_heads)
        ])

        self.output_projection = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x, return_weights=False):
        outputs = []
        all_weights = []

        for head in self.heads:
            output, weights = head(x, return_weights=True)
            outputs.append(output)
            all_weights.append(weights)

        combined = torch.cat(outputs, dim=-1)
        projected = self.output_projection(combined)

        if return_weights:
            return projected, torch.stack(all_weights, dim=1)

        return projected


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_heads,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_size,
            num_heads,
            block_size
        )

        self.feed_forward = FeedForward(embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(
            self.attention(self.norm1(x))
        )

        x = x + self.dropout(
            self.feed_forward(self.norm2(x))
        )

        return x


class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        num_heads,
        num_layers,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embeddings = nn.Embedding(
            vocab_size,
            embedding_size
        )

        self.position_embeddings = nn.Embedding(
            block_size,
            embedding_size
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embedding_size,
                num_heads,
                block_size,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_size)
        self.output_layer = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, inputs, targets=None):
        _, sequence_length = inputs.shape

        positions = torch.arange(
            sequence_length,
            device=inputs.device
        )

        x = (
            self.token_embeddings(inputs)
            + self.position_embeddings(positions)
        )

        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=0.8):
        self.eval()

        for _ in range(max_new_tokens):
            context = tokens[:, -self.block_size:]
            logits, _ = self(context)

            next_logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens