"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        self.weights = Parameter(tensor_from_numpy(np.random.uniform(0, 1, size=(num_embeddings, embedding_dim)), backend=backend))
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        oh = one_hot(x, self.num_embeddings)
        oh_flat = oh.view(bs * seq_len, self.num_embeddings)
        emb_flat = oh_flat @ self.weights.value
        emb = emb_flat.view(bs, seq_len, self.embedding_dim)
        return emb
        # raise NotImplementedError
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN ASSIGN3_2
        if self.p_dropout == 0. or not self.training:
            return x
        rands = np.random.uniform(0, 1, size=x.to_numpy().shape)
        masks = (rands > self.p_dropout).astype(np.float32)
        return (x * tensor_from_numpy(masks)) / (1.0 - self.p_dropout)
        # raise NotImplementedError
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        import math
        bound = 1.0 / math.sqrt(in_size)
        w = np.random.uniform(-bound, bound, size=(in_size, out_size))
        self.weights = Parameter(tensor_from_numpy(w, backend=backend, requires_grad=True))
        self.apply_bias = bias
        if bias:
            b = np.random.uniform(-bound, bound, size=(out_size,))
            self.bias = Parameter(tensor_from_numpy(b, backend=backend, requires_grad=True))
        else:
            b = zeros_tensor_from_numpy(shape=(out_size,), backend=backend)
            b.requires_grad_(True)
            self.bias = Parameter(b)
        # raise NotImplementedError
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        ### BEGIN ASSIGN3_2
        if len(x.shape) == 3:
            batch, seq_len, in_size = x.shape
            x_ = x.view(batch * seq_len, in_size)
            w = self.weights.value.view(in_size, self.out_size)
            out_ = x_ @ w
            if self.apply_bias:
                b = self.bias.value.view(1, self.out_size)
                out_ = out_ + b
            out = out_.view(batch, seq_len, self.out_size)
        else:
            batch, in_size = x.shape
            w = self.weights.value.view(in_size, self.out_size)
            out = x @ w
            if self.apply_bias:
                b = self.bias.value.view(1, self.out_size)
                out = out + b

        return out
        
        # raise NotImplementedError
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend, use_fused_kernels=True):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        wt = ones_tensor_from_numpy((dim,), backend=backend)
        wt.requires_grad_(True)
        self.weights = Parameter(wt)
        b = zeros_tensor_from_numpy((dim,), backend=backend)
        b.requires_grad_(True)
        self.bias = Parameter(b)
        self.use_fused_kernels = use_fused_kernels
        # raise NotImplementedError
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        if self.use_fused_kernels:
            ln = x.layernorm(self.weights.value, self.bias.value)
        else:
            ### BEGIN ASSIGN3_2
            mean = x.mean(dim=1).view(batch, 1)
            var = x.var(dim=1).view(batch, 1)
            x_normalized = (x - mean) / (var + self.eps) ** 0.5
            ln = self.weights.value * x_normalized + self.bias.value
        return ln.view(batch, dim)
        # raise NotImplementedError
        ### END ASSIGN3_2
