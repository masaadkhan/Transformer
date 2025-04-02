kernel_code = """
// Assumes embedding matrix has been sized such that Dim(embedding_matrix) < Dim(pos_enc)
extern "C" __global__ void add_matrix(float* embedding_matrix, float* pos_enc, float* output, int N) {
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

add_matrix = mod.get_function("add_matrix")

pos_encoded_emb_gpu = cuda.mem_alloc(embedding_size_bytes)
add_matrix(embedding_matrix_gpu,
           pos_encodings_gpu,
           pos_encoded_emb_gpu,
           np.int32(embedding_num_elements),
           block=(embedding_num_elements, 1, 1))
cuda.Context.synchronize()

print_gpu_array(pos_encoded_emb_gpu,
                "pos_encoded_emb",
                embedding_num_elements,
                shape=[vocab_size, token_dims])

linear_layer_code = """
// extern "C" __device__ void 

// Inputs x is a matrix and w is a vector
// Dereference the vector in x and vector multiply by w
extern "C" __global__ void linear_layer(float* x, float* w, int num_rows, int num_cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {

  }
}
"""

# Linear layer
# We need weights and input X
weights_num_elements = vocab_size * weights_dim
weights_size_bytes = weights_num_elements * np.float32().nbytes
weights_matrix_gpu = cuda.mem_alloc(weights_size_bytes)

# input X in this case would be the embedding vectors?

# Embedding matrix = [vocab_len, token_dims]
# Weight matrix = [vocab_len, 3*token_dims]
# Output matrix (Q,K,V) = [vocab_len, 3*token_dims]
# The output of this linear layer
