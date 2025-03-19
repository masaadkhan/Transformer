import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np

def print_gpu_array(a_gpu, var_name, num_elements):
  a_host = np.empty(num_elements, np.float32)
  cuda.memcpy_dtoh(a_host, a_gpu)
  print(f"{var_name}={a_host}")

# Assume square
def check_matmul(a_gpu, b_gpu, c_gpu, num_rows, num_cols, num_elements):
  a_host = np.empty(num_elements, np.float32)
  cuda.memcpy_dtoh(a_host, a_gpu)
  b_host = np.empty(num_elements, np.float32)
  cuda.memcpy_dtoh(b_host, b_gpu)
  c_host = np.empty(num_elements, np.float32)
  cuda.memcpy_dtoh(c_host, c_gpu)

  a_mat = a_host.reshape(num_rows, num_cols)
  b_mat = b_host.reshape(num_rows, num_cols)
  c_mat = c_host.reshape(num_rows, num_cols)

  expected_result = a_mat @ b_mat

  # print(f"{c_mat=}")
  # print(f"{expected_result=}")

  if (np.allclose(c_mat, expected_result)):
    print("Matmul successful\n")
  else:
    print("Matmul did not match GPU...\n")

a_rows = 32
a_cols = 32
a_elements = a_rows * a_cols
a_size_bytes = a_elements * np.float32().nbytes

b_rows = 32
b_cols = 32
b_elements = b_rows * b_cols
b_size_bytes = b_elements * np.float32().nbytes

c_size_bytes = a_size_bytes

N = a_rows * a_cols

kernel_code = f"""
extern "C" __global__ void init_array(float* array, int N) {{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {{
    array[idx] = idx;
  }}
}}

extern "C" __global__ void regular_matmul(float* a, float* b, int num_rows, int num_cols, float* c) {{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if ((row < num_rows) && (col < num_cols)) {{
    int idx = row * num_rows + col;
    int sum = 0;
    for (int i = 0; i < num_rows; i++) {{
      sum += a[row * num_cols + i] * b[i * num_rows + col];
    }}
    c[idx] = sum;
  }}
}}

extern "C" __global__ void shared_matmul(float* a, float* b, int num_rows, int num_cols, float* c) {{
  // Dynamically store tensor for matmul
  extern __shared__ float f[];

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if ((row < num_rows) && (col < num_cols)) {{
    int idx = row * num_rows + col;

    // Set the shared memory values of a and b
    f[idx] = a[idx];
    f[idx + {N}] = b[idx];

    // Sync to ensure all threads finished their writes...
    __syncthreads();

    float* a_shared = &f[0];
    float* b_shared = &f[{N}];

    int sum = 0;
    for (int i = 0; i < num_rows; i++) {{
      sum += a_shared[row * num_cols + i] * b_shared[i * num_rows + col];
    }}
    c[idx] = sum;
  }}
}}
"""

mod = SourceModule(
  kernel_code,
  no_extern_c=True,
  options=["-std=c++11",
          "-Xcompiler",
          "-fPIC"])

init_array = mod.get_function("init_array")
regular_matmul = mod.get_function("regular_matmul")
shared_matmul = mod.get_function("shared_matmul")

a_gpu = cuda.mem_alloc(a_size_bytes)
b_gpu = cuda.mem_alloc(b_size_bytes)
c_gpu = cuda.mem_alloc(c_size_bytes)

init_array(a_gpu, np.int32(a_elements), block=(a_elements, 1, 1))
init_array(b_gpu, np.int32(b_elements), block=(b_elements, 1, 1))

start = cuda.Event()
end = cuda.Event()

regular_sum = 0
for i in range(10000):
  start.record()
  regular_matmul(a_gpu, b_gpu, np.int32(a_rows), np.int32(a_cols),
        c_gpu, block=(a_rows, b_cols, 1))
  end.record()
  end.synchronize()

  regular_matmul_time_ms = start.time_till(end)
  regular_sum += regular_matmul_time_ms
  # check_matmul(a_gpu, b_gpu, c_gpu, a_rows, a_cols, a_elements)

shared_sum = 0
for i in range(10000):
  start.record()
  shared_matmul(a_gpu, b_gpu, np.int32(a_rows), np.int32(a_cols),
        c_gpu, block=(a_rows, b_cols, 1), shared=a_size_bytes + b_size_bytes)
  end.record()
  end.synchronize()

  shared_matmul_time_ms = start.time_till(end)
  shared_sum += shared_matmul_time_ms
  # check_matmul(a_gpu, b_gpu, c_gpu, a_rows, a_cols, a_elements)

cuda.Context.synchronize()

print(f"It took {regular_sum=} and {shared_sum=}\n")
