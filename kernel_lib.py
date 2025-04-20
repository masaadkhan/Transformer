import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import math

gen_pos_encodings = None
generate_uniform_random = None
init_array = None
init_array_w_val = None
regular_matmul = None
shared_matmul = None
add_matrix_w_vector = None
scalar_divide = None
matrix_row_wise_add = None
matrix_row_wise_max = None
fused_softmax = None
matrix_transpose = None
regular_add = None
gather = None

def xavier_uniform(fan_in, fan_out):
  return math.sqrt(6 / (fan_in + fan_out))

def print_gpu_array(a_gpu, var_name, num_elements, shape=None, verbose=False):
  a_host = np.empty(num_elements, np.float32)
  cuda.memcpy_dtoh(a_host, a_gpu)
  np.set_printoptions(threshold=np.inf, precision=4, linewidth=120)
  if (shape != None):
    a_host = a_host.reshape(shape[0], shape[1])
  print(f"{var_name}={a_host}")

kernel_lib_code = f"""
#include <curand_kernel.h>

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
regular_add(float* a, float* b, int N, float* c) {{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < N) {{
    c[idx] = a[idx] + b[idx];
  }}
}}

extern "C" __global__ void
regular_matmul(float* a, float* b, int num_rows, int num_cols, int inner_dim, float* c) {{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if ((row < num_rows) && (col < num_cols)) {{
    int idx = row * num_cols + col;

    float sum = 0;
    for (int i = 0; i < inner_dim; i++) {{
      float a_val = a[row * inner_dim + i];
      float b_val = b[i * num_cols + col];
      sum += a_val * b_val;
    }}

    c[idx] = sum;
  }}
}}

extern "C" __global__ void
shared_matmul(float* a, float* b, int num_rows, int num_cols, int inner_dim, float* c) {{
  // Dynamically store tensor for matmul
  extern __shared__ float f[];

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int idx = row * num_rows + col;

  if ((row < num_rows) && (col < num_cols)) {{
    // Set the shared memory values of a and b
    f[idx] = a[idx];
    f[idx + num_rows * num_cols] = b[idx];
  }}

  // Sync to ensure all threads finished their writes...
  __syncthreads();

  if ((row < num_rows) && (col < num_cols)) {{
    float* a_shared = &f[0];
    float* b_shared = &f[num_rows * num_cols];

    float sum = 0;
    for (int i = 0; i < num_rows; i++) {{
      sum += a_shared[row * num_cols + i] * b_shared[i * num_rows + col];
    }}

    c[idx] = sum;
  }}
}}

  // Let's think about how this code should work
  // Initially I have a 4 element vector that I am trying to sum
  // I am doing a tree reduction in this case
  // To do 4 element reduction, I need 2 threads running in this case
  // 1 2 3 4
  // Thread0 will sum 1 + 2
  // Thread1 will sum 3 + 4
  // Thread0 will store this sum into sum[0]
  // Thread1 will store this sum into sum[1]
  // Then we will need to sum
  // 3 7
  // Thread0 will sum 3 + 7
  // Store into sum[0]
  // 10
// Takes a matrix and performs row-wise add reduction and outputs a vector representing each row's sum
extern "C" __global__ void matrix_row_wise_add(float* matrix, int num_rows, int num_cols, float* output_vec) {{
  extern __shared__ float f[];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < num_rows && col < num_cols) {{
    int idx = row * num_cols + col;
    f[idx] = matrix[idx];
  }}

  __syncthreads();

  if (row < num_rows && col < num_cols) {{
    float* row_ptr = &f[row * num_cols];

    // TODO: Revisit this for non-power-of-two num_cols
    for (int i = num_cols; i > 1; i = i / 2) {{
      if (threadIdx.x < i) {{
        int idx1 = threadIdx.x;
        int idx2 = threadIdx.x + i / 2;

        float a_ptr = row_ptr[idx1];
        float b_ptr = row_ptr[idx2];

        //printf("threadIdx.x: %d, idx1: %d, a: %f idx2: %d, b: %f\\n", threadIdx.x, idx1, row_ptr[idx1], idx2, row_ptr[idx2]);
        //printf("sum: %f\\n", a_ptr + b_ptr);

        row_ptr[idx1] = a_ptr + b_ptr;
      }}
      // Sync all threads before doing next iteration...
      __syncthreads();
    }}

    if (threadIdx.x == 0) {{
      output_vec[row] = row_ptr[threadIdx.x];
    }}
  }}
}}

extern "C" __global__ void matrix_row_wise_max(float* matrix, int num_rows, int num_cols, float* output_vec) {{
  extern __shared__ float f[];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = row * num_cols + col;

  if (row < num_rows && col < num_cols) {{
    f[idx] = matrix[idx];
  }}
  __syncthreads();

  if (row < num_rows && col < num_cols) {{
    float* row_ptr = &f[row * num_cols];

    for (int i = num_cols; i > 1; i = i / 2) {{
      if (threadIdx.x < i) {{
        int idx1 = threadIdx.x;
        int idx2 = threadIdx.x + i / 2;

        float a_ptr = row_ptr[idx1];
        float b_ptr = row_ptr[idx2];

        //printf("threadIdx.x: %d, idx1: %d, a: %f idx2: %d, b: %f\\n", threadIdx.x, idx1, a_ptr, idx2, b_ptr);
        row_ptr[idx1] = (a_ptr > b_ptr) ? a_ptr : b_ptr;
        //printf("output: %f\\n", row_ptr[idx1]);
      }}
      __syncthreads();
    }}

    if (threadIdx.x == 0) {{
      output_vec[row] = row_ptr[threadIdx.x];
    }}
  }}
}}

// Takes the provided row-wise sum and divides each element respectively
extern "C" __global__ void fused_softmax(float* matrix, int num_rows, int num_cols, float* output) {{
  extern __shared__ float f[];

  float* exp_scratch = f;
  float* sum_scratch = &f[num_rows * num_cols + num_cols];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = row * num_cols + col;

  if (row < num_rows && col < num_cols) {{
    f[idx] = matrix[idx];
  }}
  __syncthreads();

  for (int i = num_cols; i > 0; i /= 2) {{
    if (row < num_rows && col < num_cols) {{
      if (threadIdx.x < (i / 2)) {{
        float* exp_row = &exp_scratch[row * num_cols];
        float* matrix_row = &matrix[row * num_cols];

        float a = matrix_row[threadIdx.x];
        float b = matrix_row[threadIdx.x + i / 2];
        int store_idx = threadIdx.y * num_cols + threadIdx.x;

        exp_scratch[store_idx] = (a > b) ? a : b; // TODO: Check this out, could this not be threadIdx.x?... exp_row...
      }}
    }}
    __syncthreads();
  }}

  if (row < num_rows && col < num_cols) {{
    float row_max = f[row * num_cols];
    exp_scratch[idx] = expf(matrix[idx] - row_max);
    // printf("(%d,%d): exp_scratch: %f\\n", row, col, exp_scratch[idx]);
  }}
  __syncthreads();

  // Sum the rows and divide by the sum
  int scratch_cols = num_cols;
  for (int i = scratch_cols; i > 0; i /= 2) {{
    if (row < num_rows && col < scratch_cols) {{
      if (col < (i / 2)) {{
        int a_idx = col;
        int b_idx = col + i / 2;

        float* exp_row = &exp_scratch[row * num_cols];
        float* sum_row = &sum_scratch[row * num_cols];

        float a = (i == scratch_cols) ? exp_row[a_idx] : sum_row[a_idx];
        float b = (i == scratch_cols) ? exp_row[b_idx] : sum_row[b_idx];
        int store_idx = row * (scratch_cols / 2) + col;

        // printf("threadIdx.y: %d threadIdx.x: %d a_idx: %d b_idx: %d sum[%d] = %f + %f\\n", row, col, a_idx, b_idx, store_idx, a, b);
        sum_row[a_idx] = a + b;
      }}
    }}
  }}
  __syncthreads();

  if (row < num_rows && col < num_cols) {{
    output[idx] = exp_scratch[idx] / sum_scratch[row * num_cols];
  }}
}}

extern "C" __global__ void calc_positional_encoding(float* pos_enc, int num_rows, int num_cols) {{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < num_rows && col < num_cols) {{
    int idx = row * num_cols + col;
    
    int token_idx = row;
    int current_dim = col / 2;
    int token_dims = num_cols;
    float expo = (2.0 * current_dim) / token_dims;

    if ((col & 1) == 0) {{
      pos_enc[idx] = sinf(token_idx / powf(10000, expo));
    }} else {{
      pos_enc[idx] = cosf(token_idx / powf(10000, expo));
    }}
  }}
}}

extern "C" __global__ void generate_uniform_random(float* numbers, float scale, int seed, int N) {{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {{
    curandState state;
    curand_init(seed, idx, 0, &state);
    numbers[idx] = (curand_uniform(&state) * 2 * scale) - scale;
  }}
}}

extern "C" __global__ void
matrix_transpose(float* mat, int num_rows, int num_cols, float* c) {{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if ((row < num_rows) && (col < num_cols)) {{
    int idx = row * num_cols + col;

    int t_cols = num_rows;
    int t_row = col;
    int t_col = row;

    int t_idx = t_row * t_cols + t_col;

    c[t_idx] = mat[idx];
  }}
}}

extern "C" __global__ void
gather(float* mat, int start_idx, int stride, int output_rows, int output_cols, float* c) {{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < output_rows && col < output_cols) {{
    int idx = row * output_cols + col;
    int stride_idx = row * stride + start_idx + col;
    c[idx] = mat[stride_idx];
    //printf("(%d,%d): Copying %f from %d, start_idx: %d\\n", row, col, c[idx], stride_idx, start_idx);
  }}
}}
"""

def compile_kernels():
  global gen_pos_encodings
  global generate_uniform_random
  global init_array
  global init_array_w_val
  global regular_matmul
  global shared_matmul
  global add_matrix_w_vector
  global scalar_divide
  global matrix_row_wise_add
  global matrix_row_wise_max
  global fused_softmax
  global matrix_transpose
  global regular_add
  global gather

  lib = SourceModule(
    kernel_lib_code,
    no_extern_c=True,
    options=["-std=c++11",
            "-Xcompiler",
            "-fPIC"])

  gen_pos_encodings = lib.get_function("calc_positional_encoding")
  generate_uniform_random = lib.get_function("generate_uniform_random")
  init_array = lib.get_function("init_array")
  init_array_w_val = lib.get_function("init_array_w_val")
  regular_matmul = lib.get_function("regular_matmul")
  shared_matmul = lib.get_function("shared_matmul")
  add_matrix_w_vector = lib.get_function("add_matrix_w_vector")
  scalar_divide = lib.get_function("scalar_divide")
  matrix_row_wise_add = lib.get_function("matrix_row_wise_add")
  matrix_row_wise_max = lib.get_function("matrix_row_wise_max")
  fused_softmax = lib.get_function("fused_softmax")
  matrix_transpose = lib.get_function("matrix_transpose")
  regular_add = lib.get_function("regular_add")
  gather = lib.get_function("gather")
