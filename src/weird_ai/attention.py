import torch
import torch.nn as nn
import random as rand


class SimpleSelfAttention(nn.Module):
    """
    A simple self-attention module without trainable query, key, and value projections.

    This version is primarily for learning.
    """

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, embedding_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        # TODO:
        # 1. Compute attention scores using matrix multiplication.
        torch.manual_seed(8675309)
        inputs = torch.tensor([ 
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],])

        query = inputs[1]
        attn_scores=inputs @ inputs.T
        """attn_scores= torch.empty(inputs.shape[0])for index, embedding in enumerate(inputs):
        attn_scores[i] = torch.dot(index, query)
        print(attn_scores) """

        # 2. Normalize scores with softmax.
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 3. Compute context vectors as weighted sums of input vectors.
        contextVectors = attn_weights @ inputs
        #raise NotImplementedError("Implement simple self-attention.")
        print(attn_weights)
        print(attn_weights.shape)
        print(contextVectors)
        print(contextVectors.shape)
        print()


        return contextVectors, attn_weights



string = "hdfhjfdwkhjd"
model = SimpleSelfAttention()
model(string)

class SelfAttention(nn.Module):
    """
    Trainable self-attention using query, key, and value projections.
    """

    def __init__(self, embedding_dim, output_dim, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, output_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        # TODO:
        # 1. Compute queries, keys, and values.
        # 2. Compute scaled attention scores.
        # 3. Apply softmax.
        # 4. Compute context vectors.

        raise NotImplementedError("Implement trainable self-attention.")

class CausalAttention(nn.Module):
    """
    Self-attention with a causal mask so tokens cannot attend to future tokens.
    """

    def __init__(self, embedding_dim, output_dim, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (batch_size, num_tokens, output_dim)
        """

        # TODO:
        # 1. Compute keys, queries, and values.
        # 2. Compute scaled attention scores.
        # 3. Mask future tokens.
        # 4. Apply softmax.
        # 5. Apply dropout.
        # 6. Compute context vectors.

        raise NotImplementedError("Implement causal attention.")