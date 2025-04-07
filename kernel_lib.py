import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np

def print_gpu_array(a_gpu, var_name, num_elements, shape=None, verbose=False):
  a_host = np.empty(num_elements, np.float32)
  cuda.memcpy_dtoh(a_host, a_gpu)
  np.set_printoptions(threshold=np.inf, precision=4, linewidth=120)
  if (shape != None):
    a_host = a_host.reshape(shape[0], shape[1])
  print(f"{var_name}={a_host}")

kernel_lib_code = f"""
extern "C" __global__ void init_array(float* array, int N) {{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {{
    array[idx] = idx;
  }}
}}

// Init array with some provided value
extern "C" __global__ void init_array_w_val(float* arr, int val, int N) {{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {{
    arr[idx] = val;
  }}
}}

extern "C" __global__ void
add_matrix_w_vector(float* mat, float* vec, int num_rows, int num_cols, float* c) {{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if ((row < num_rows) && (col < num_cols)) {{
    int idx = row * num_cols + col;

    c[idx] = mat[idx] + vec[col];
  }}
}}

extern "C" __global__ void
scalar_divide(float* mat, float div, int N, float* c) {{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {{
    c[idx] = mat[idx] / div;
  }}
}}

extern "C" __global__ void
regular_matmul(float* a, float* b, int num_rows, int num_cols, int inner_dim, float* c) {{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  //printf("Row %d and Col %d\\n", row, col);
  if ((row < num_rows) && (col < num_cols)) {{
    //printf("Row %d and Col %d\\n", row, col);
    int idx = row * num_cols + col;

    float sum = 0;
    for (int i = 0; i < inner_dim; i++) {{
      float a_val = a[row * inner_dim + i];
      float b_val = b[i * num_cols + col];
      if (row == 4 && col == 0) {{
        //printf("Value of a: %f, Value of b: %f\\n", a_val, b_val);
        //printf("Value of multiply: %f\\n", a_val * b_val);
      }}
      sum += a_val * b_val;
    }}
    c[idx] = sum;
    if (row == 4 && col == 0) {{
      //printf("Value of c: %f\\n", c[idx]);
    }}
  }}
}}

extern "C" __global__ void
shared_matmul(float* a, float* b, int num_rows, int num_cols, int inner_dim, float* c) {{
  // Dynamically store tensor for matmul
  extern __shared__ float f[];

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if ((row < num_rows) && (col < num_cols)) {{
    int idx = row * num_rows + col;

    // Set the shared memory values of a and b
    f[idx] = a[idx];
    f[idx + num_rows * num_cols] = b[idx];

    // Sync to ensure all threads finished their writes...
    __syncthreads();

    float* a_shared = &f[0];
    float* b_shared = &f[num_rows * num_cols];

    float sum = 0;
    for (int i = 0; i < num_rows; i++) {{
      sum += a_shared[row * num_cols + i] * b_shared[i * num_rows + col];
    }}
    c[idx] = sum;
  }}
}}
"""

lib = SourceModule(
  kernel_lib_code,
  no_extern_c=True,
  options=["-std=c++11",
          "-Xcompiler",
          "-fPIC"])

init_array = lib.get_function("init_array")
init_array_w_val = lib.get_function("init_array_w_val")
regular_matmul = lib.get_function("regular_matmul")
shared_matmul = lib.get_function("shared_matmul")
add_matrix_w_vector = lib.get_function("add_matrix_w_vector")
scalar_divide = lib.get_function("scalar_divide")
