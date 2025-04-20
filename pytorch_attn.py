import torch
import torch.nn as nn
import attention_cuda

if not torch.cuda.is_available():
  raise RuntimeError("CUDA is not available. This module requires a GPU with CUDA support.")

DEVICE = torch.device("cuda")

class AttentionLayer(nn.Module):
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

if __name__ == "__main__":
    torch.set_default_device('cuda') 
    # Example usage
    embed_dim = 4
    num_heads = 1
    dropout = 0.1
    
    # Instantiate the attention layer
    attention_layer = AttentionLayer(embed_dim, num_heads, dropout)

    # Create dummy input: (batch_size, seq_len, embed_dim)
    batch_size = 1
    seq_len = 4
    x = torch.rand(batch_size, seq_len, embed_dim, device=DEVICE)

    from torch.utils.dlpack import to_dlpack
    import cupy as cp
    import pycuda.autoinit  # initializes CUDA driver and context
    import pycuda.gpuarray as gpuarray

    # print(x.device)
    # print(cp.cuda.runtime.getDevice())

    # print("torch:", torch.__version__, torch.version.cuda)
    # print("cupy :", cp.__version__, hex(cp.cuda.runtime.runtimeGetVersion()))

    dlpack_tensor = to_dlpack(x)
    cp_array = cp.from_dlpack(dlpack_tensor)
    gpu_x = gpuarray.GPUArray(cp_array.shape, cp_array.dtype, gpudata=cp_array.data.ptr)

    # print(f"{x=}")
    # print(f"{gpu_x=}")

    # Optional causal mask to prevent attention to future tokens
    # True indicates positions to attend, False to mask
    # mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE)).bool()

    # print(f"{mask.device=}")

    # Forward pass
    output, weights = attention_layer(x)

    print(f"PyTorch attention output: {output=}")
    print(f"Custom CUDA attention output: {attention_cuda.attention_cuda(gpu_x, cp_array.shape[1], cp_array.shape[2])}")
