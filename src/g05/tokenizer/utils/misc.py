import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import constraints
from typing import Union


IMAGE_TOKEN_INDEX = 1
PROPRIO_TOKEN_INDEX = 2
ACTION_TOKEN_INDEX = 3
TEXT_TOKEN_INDEX = 4


def left_to_right_align(x, input_mask, labels=None):
    """
    x: [B, N]
    input_mask: [B, N]
    Returns: x and input_mask right-aligned.
    """
    assert x.dim() == 2
    assert input_mask.dim() == 2
    B, N = x.shape
    assert input_mask.shape == (B, N)

    if labels is not None:
        assert labels.dim() == 2
        assert labels.shape == (B, N)

    arange = torch.arange(N, device=x.device)
    seqlen = ((input_mask > 0) * arange).max(dim=1).values + 1  # [B]

    # Vectorized roll: indices[i, j] = (j + seqlen[i]) % N.
    indices = (arange.unsqueeze(0) + seqlen.unsqueeze(1)) % N  # [B, N]

    x_aligned = torch.gather(x, 1, indices)
    input_mask_aligned = torch.gather(input_mask, 1, indices)
    labels_aligned = torch.gather(labels, 1, indices) if labels is not None else None
    return x_aligned, input_mask_aligned, labels_aligned


def resize_embeddings_with_distribution_init(
    embedding: Union[nn.Embedding, torch.Tensor],
    num_new_tokens: int,
    padding_idx: int = None,
    epsilon: float = 1e-5,
) -> nn.Embedding:
    """
    Resize an embedding layer or weight tensor by adding new tokens and initializing their embeddings
    from a multivariate normal distribution estimated from the existing embeddings.
    Falls back to mean initialization if covariance is not positive definite.

    Args:
        embedding (Union[nn.Embedding, torch.Tensor]): The existing embedding layer or its weight tensor.
        num_new_tokens (int): Number of tokens to add.
        padding_idx (int, optional): Padding token index to exclude from statistics.
        epsilon (float): Regularization term to ensure numerical stability of covariance.

    Returns:
        nn.Embedding: A new embedding layer with extended vocabulary and initialized weights.
    """
    # Normalize input
    if isinstance(embedding, nn.Embedding):
        weight = embedding.weight.detach().clone()
        emb_dim = embedding.embedding_dim
        original_padding_idx = embedding.padding_idx
    elif isinstance(embedding, torch.Tensor):
        weight = embedding.detach().clone()
        emb_dim = weight.size(1)
        original_padding_idx = padding_idx
    else:
        raise TypeError("embedding must be nn.Embedding or torch.Tensor")

    vocab_size = weight.size(0)

    # Exclude padding if present
    if original_padding_idx is not None:
        mask = torch.ones(vocab_size, dtype=torch.bool, device=weight.device)
        mask[original_padding_idx] = False
        used_weights = weight[mask]
    else:
        used_weights = weight

    # Compute mean and covariance
    mean = used_weights.mean(dim=0)
    centered = used_weights - mean
    cov = centered.T @ centered / used_weights.size(0)  # Empirical covariance

    # Try multivariate normal sampling
    try:
        is_psd = constraints.positive_definite.check(epsilon * cov).all()
        if is_psd:
            mvn = MultivariateNormal(
                mean, covariance_matrix=cov + epsilon * torch.eye(emb_dim, device=weight.device)
            )
            new_weights = mvn.sample((num_new_tokens,))
        else:
            raise ValueError("Covariance not PSD")
    except Exception:
        # Fall back to mean initialization
        new_weights = mean.unsqueeze(0).repeat(num_new_tokens, 1)

    # Concatenate and create new embedding
    expanded_weights = torch.cat([weight, new_weights.to(weight.dtype)], dim=0)
    # new_embedding = nn.Embedding(vocab_size + num_new_tokens, emb_dim, padding_idx=original_padding_idx)
    # with torch.no_grad():
    #     new_embedding.weight.copy_(expanded_weights)

    return expanded_weights


if __name__ == "__main__":
    # Suppose we have an embedding layer or tensor.
    embedding_layer = nn.Embedding(30000, 768, padding_idx=0)

    # Add 100 new tokens.
    resized = resize_embeddings_with_distribution_init(embedding_layer, 100)

    print(resized.weight.shape)  # torch.Size([30100, 768])
