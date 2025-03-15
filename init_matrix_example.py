import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np

def print_gpu_array(a_gpu, num_elements):
  a_host = np.empty(num_elements, np.float32)
  cuda.memcpy_dtoh(a_host, a_gpu)
  print(f"{a_host=}")

kernel_code = """
extern "C" __global__ void init_matrix(float* matrix, int num_rows, int num_cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < num_rows) && (col < num_cols)) {
    int idx = row * num_rows + col;
    matrix[idx] = idx;
  }
}
"""

mod = SourceModule(kernel_code,
                   no_extern_c=True,
                   options=["-std=c++11",
                           "-Xcompiler",
                           "-fPIC"])

init_matrix = mod.get_function("init_matrix")

num_vectors = 3
vector_dims = 3

num_elements = num_vectors * vector_dims
matrix_size_bytes = num_elements * np.float32().nbytes
matrix_gpu = cuda.mem_alloc(matrix_size_bytes)

init_matrix(matrix_gpu, np.int32(num_vectors), np.int32(vector_dims), block=(num_vectors, vector_dims, 1))

print_gpu_array(matrix_gpu, num_elements)
