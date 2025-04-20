# %%
from matrix import Matrix, kl, np
import math

def attention_cuda(embeddings_gpu, num_rows, num_cols):
  embeddings_w_pos = Matrix(num_rows, num_cols, np.float32, gpu=True)
  embeddings_w_pos.set_gpu_matrix(embeddings_gpu)

  weights_t = Matrix(embeddings_w_pos.num_cols, 3 * embeddings_w_pos.num_cols, np.float32, gpu=True)
  weights_t.alloc_on_gpu()
  weights_scale = kl.xavier_uniform(weights_t.num_rows, weights_t.num_cols)

  weights_t.init_uniform_rand(weights_scale)

  # QKV matrix = [vocab_size, 3 * token_dims]
  QKV = embeddings_w_pos * weights_t

  bias_scale = kl.xavier_uniform(QKV.num_cols, 1)

  b = Matrix(QKV.num_cols, 1, np.float32, gpu=True)
  b.alloc_on_gpu()
  b.init_uniform_rand(bias_scale)

  QKV_b = Matrix(QKV.num_rows, QKV.num_cols, np.float32, gpu=True)
  QKV_b.alloc_on_gpu()

  kl.add_matrix_w_vector(QKV.a_gpu,
                      b.a_gpu,
                      np.int32(QKV.num_rows),
                      np.int32(QKV.num_cols),
                      QKV_b.a_gpu,
                      block=(QKV.num_cols, QKV.num_rows, 1))

  Q = Matrix(QKV_b.num_rows, QKV_b.num_cols / 3, np.float32, gpu=True)
  Q.set_gpu_matrix(QKV_b.a_gpu, stride=QKV_b.num_cols, start_idx=(0 * QKV_b.num_cols) / 3)

  K = Matrix(QKV_b.num_rows, QKV_b.num_cols / 3, np.float32, gpu=True)
  K.set_gpu_matrix(QKV_b.a_gpu, stride=QKV_b.num_cols, start_idx=(1 * QKV_b.num_cols) / 3)

  V = Matrix(QKV_b.num_rows, QKV_b.num_cols / 3, np.float32, gpu=True)
  V.set_gpu_matrix(QKV_b.a_gpu, stride=QKV_b.num_cols, start_idx=(2 * QKV_b.num_cols) / 3)

  score_scaled = (Q * K.transpose()) / math.sqrt(Q.num_cols)

  score_softmaxed = Matrix(score_scaled.num_rows, score_scaled.num_cols, np.float32, gpu=True)
  score_softmaxed.alloc_on_gpu()
  score_softmaxed.init_incremental()

  matrix_bytes = score_scaled.num_elements() * score_scaled.dtype().nbytes
  shared_mem_bytes = int((3 * matrix_bytes) / 2)

  kl.fused_softmax(score_scaled.a_gpu,
                np.int32(score_scaled.num_rows),
                np.int32(score_scaled.num_cols),
                score_softmaxed.a_gpu,
                block=(score_scaled.num_cols, score_scaled.num_rows, 1),
                shared=shared_mem_bytes)

  attention = score_softmaxed * V
  add = embeddings_w_pos + attention

  return add

