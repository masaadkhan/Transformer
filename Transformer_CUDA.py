# %%
#%load_ext autoreload
#%autoreload 2

from kernel_lib import *
from matrix import Matrix
import math

# %%
vocab = ["This", "is", "a", "sentence"]
vocab_size = len(vocab)

pos_enc_sequence_len = 10
token_dims = vocab_size

# %%
pos_encodings = Matrix(pos_enc_sequence_len, token_dims, np.float32, gpu=True)
pos_encodings.alloc_on_gpu()
gen_pos_encodings(pos_encodings.a_gpu,
                  np.int32(pos_encodings.num_rows),
                  np.int32(pos_encodings.num_cols),
                  block=(pos_encodings.num_cols, pos_encodings.num_rows, 1))
cuda.Context.synchronize()

print(f"{pos_encodings=}")

# %%
sentence = "This is a sentence"
sentence_toks = [0, 1, 2, 3] # Straight forward
word2tok = {"This" : 0, "is" : 1, "a" : 2, "sentence" : 3}

# %%
embeddings = Matrix(vocab_size, token_dims, np.float32, gpu=True)
embeddings.alloc_on_gpu()
embeddings.init_uniform_rand(math.sqrt(6.0 / (embeddings.num_rows + embeddings.num_cols)))

cuda.Context.synchronize()

print(f"{embeddings=}")


# %%
#TODO: Add a function that adds two matrices but has the ability to "scale" down matrices based on which one is bigger, etc
# Special "trimmed" matrix add...
kernel_code = """
// Assumes embedding matrix has been sized such that Dim(embedding_matrix) < Dim(pos_enc)
extern "C" __global__ void add_pos_enc_and_embed(float* embedding_matrix, float* pos_enc, float* output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = embedding_matrix[idx] + pos_enc[idx];
  }
}
"""

mod = SourceModule(kernel_code,
                   no_extern_c=True,
                   options=["-std=c++11",
                           "-Xcompiler",
                           "-fPIC"])

add_pos_enc_and_embed = mod.get_function("add_pos_enc_and_embed")

# %%
embeddings_w_pos = Matrix(embeddings.num_rows, embeddings.num_cols, np.float32, gpu=True)
embeddings_w_pos.alloc_on_gpu()
add_pos_enc_and_embed(embeddings.a_gpu,
                      pos_encodings.a_gpu,
                      embeddings_w_pos.a_gpu,
                      np.int32(embeddings.num_elements()),
                      block=(embeddings.num_elements(), 1, 1))
cuda.Context.synchronize()

print(f"{embeddings_w_pos=}")

# %%
test_a = Matrix(4,4,np.float32,gpu=True)
test_a.alloc_on_gpu()
test_a.init_incremental()

test_b = Matrix(4,4,np.float32,gpu=True)
test_b.alloc_on_gpu()
test_b.init_incremental()

test_c = test_a * test_b
print(test_c)

# %%
print(f"Initially: {test_c=}")
test_c = test_c / 2
print(f"After scalar divide: {test_c=}")

# %%
# Embedding matrix = [vocab_size, token_dims]
# Technically you can make swap the dimensions of this and it will still work
# One way requires a transpose, the other doesn't
# Weights: [3 * token_dims, token_dims]

#TODO: Make a weight transpose instead for learnings... Even though a bit less efficient...
# weights = Matrix(3 * embeddings.num_cols, embeddings.num_cols, np.float32, gpu=True)
# weights.alloc_on_gpu()
weights_t = Matrix(embeddings_w_pos.num_cols, 3 * embeddings_w_pos.num_cols, np.float32, gpu=True)
weights_t.alloc_on_gpu()

# QKV matrix = [vocab_size, 3 * token_dims]
QKV = Matrix(embeddings_w_pos.num_rows, weights_t.num_cols, np.float32, gpu=True)
QKV.alloc_on_gpu()

regular_matmul(embeddings_w_pos.a_gpu,
               weights_t.a_gpu,
               np.int32(QKV.num_rows),
               np.int32(QKV.num_cols),
               np.int32(embeddings_w_pos.num_cols),
               QKV.a_gpu,
               block=(QKV.num_cols, QKV.num_rows, 1))

b = Matrix(QKV.num_cols, 1, np.float32, gpu=True)
b.alloc_on_gpu()
# TODO: Fix the scale here..?
b.init_uniform_rand(math.sqrt(6.0 / (embeddings.num_rows + embeddings.num_cols)))

QKV_b = Matrix(QKV.num_rows, QKV.num_cols, np.float32, gpu=True)
QKV_b.alloc_on_gpu()

add_matrix_w_vector(QKV.a_gpu,
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

score = (Q * K) / math.sqrt(Q.num_cols)
score_scaled = score / math.sqrt(Q.num_cols)

# %%
test_a = Matrix(4,4,np.float32,gpu=True)
test_a.alloc_on_gpu()
test_a.init_incremental()

test_output = Matrix(4,1,np.float32,gpu=True)
test_output.alloc_on_gpu()
test_output.init_incremental()

print(f"{test_a=}")
print(f"Before: {test_output=}")

matrix_row_wise_add(test_a.a_gpu,
        np.int32(test_a.num_rows),
        np.int32(test_a.num_cols),
        test_output.a_gpu,
        block=(test_a.num_cols,test_a.num_rows,1),
        shared=test_a.num_elements() * test_a.dtype().nbytes)

print(f"After: {test_output=}")

# %%
print(f"{score=}")
print(f"{score_scaled=}")

score_scaled_row_sum = Matrix(score_scaled.num_rows, 1, np.float32, gpu=True)
score_scaled_row_sum.alloc_on_gpu()

matrix_row_wise_add(score_scaled.a_gpu,
        np.int32(score_scaled.num_rows),
        np.int32(score_scaled.num_cols),
        score_scaled_row_sum.a_gpu,
        block=(score_scaled.num_cols, score_scaled.num_rows, 1),
        shared=score_scaled.num_elements() * score_scaled.dtype().nbytes)

# %%
print(f"{score_scaled_row_sum=}")

score_scaled_softmaxed = Matrix(score_scaled.num_rows, score_scaled.num_cols, np.float32, gpu=True)
score_scaled_softmaxed.alloc_on_gpu()

softmax(score_scaled.a_gpu,
        score_scaled_row_sum.a_gpu,
        np.int32(score_scaled.num_rows),
        np.int32(score_scaled.num_cols),
        score_scaled_softmaxed.a_gpu,
        block=(score_scaled_softmaxed.num_cols, score_scaled_softmaxed.num_rows, 1))

print(f"{score_scaled_softmaxed=}")


