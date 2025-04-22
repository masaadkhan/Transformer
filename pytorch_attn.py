import torch
import torch.nn as nn
import numpy as np
import attention_cuda

if not torch.cuda.is_available():
  raise RuntimeError("CUDA is not available. This module requires a GPU with CUDA support.")

DEVICE = torch.device("cuda")

class MHALayer(nn.Module):
  """
  Wrapper for PyTorch's built-in MultiheadAttention.
  Args:
      embed_dim (int): total dimension of the model.
      num_heads (int): number of attention heads.
      dropout (float): dropout probability on attention weights.
  """
  def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
    super().__init__()
    self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

  def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
    # x: (batch, seq_len, embed_dim)
    # PyTorch expects (seq_len, batch, embed_dim)
    x = x.transpose(0, 1)
    # attn_mask expects shape (seq_len, seq_len)
    attn_output, attn_weights = (
        self.attn(x, x, x, attn_mask=mask)
        if mask is not None
        else self.attn(x, x, x)
    )
    # return to (batch, seq_len, embed_dim)
    attn_output = attn_output.transpose(0, 1)
    return attn_output, attn_weights

class AttnDotProdLayer(nn.Module):
  """
  Wrapper for PyTorch's built-in MultiheadAttention.
  Args:
      embed_dim (int): total dimension of the model.
      num_heads (int): number of attention heads.
      dropout (float): dropout probability on attention weights.
  """
  def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
    super().__init__()
    self.attn = nn.functional.scaled_dot_product_attention(embed_dim, num_heads, dropout=dropout)

  def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
    # x: (batch, seq_len, embed_dim)
    # PyTorch expects (seq_len, batch, embed_dim)
    x = x.transpose(0, 1)
    # attn_mask expects shape (seq_len, seq_len)
    attn_output, attn_weights = (
        self.attn(x, x, x, attn_mask=mask)
        if mask is not None
        else self.attn(x, x, x)
    )
    # return to (batch, seq_len, embed_dim)
    attn_output = attn_output.transpose(0, 1)
    return attn_output, attn_weights

if __name__ == "__main__":
    torch.set_default_device('cuda') 

    embed_dim = 4
    num_heads = 1
    dropout = 0.1

    mha_layer = MHALayer(embed_dim, num_heads, dropout)
    # attndot_layer = AttnDotProdLayer(embed_dim, num_heads, dropout)

    batch_size = 1
    seq_len = 4

    x = torch.rand(batch_size, seq_len, embed_dim, device=DEVICE)
    # print(f"{x=}")
    x_cpu = x.cpu()
    x_np = x_cpu.numpy()
    # print(f"{x_np=}")
    x_pycuda = attention_cuda.kl.cuda.mem_alloc(x_np.shape[1] * x_np.shape[2] * np.float32().nbytes)
    attention_cuda.kl.cuda.memcpy_htod(x_pycuda, x_np)

    # mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE)).bool()

    mha_output, mha_weights = mha_layer(x)
    # attndot_output, attndot_weights = attndot_layer(x)

    # print(f"PyTorch MHA output:\n{mha_output=}\n")
    # print(f"PyTorch Dot Product Attention output: {attndot_output=}")
    cuda_attn = attention_cuda.attention_cuda(x_pycuda, x_np.shape[1], x_np.shape[2])
    # print(f"Custom CUDA attention output:\n{cuda_attn=}\n")
