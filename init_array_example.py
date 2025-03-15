import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np

def print_gpu_array(a_gpu, num_elements):
  a_host = np.empty(num_elements, np.float32)
  cuda.memcpy_dtoh(a_host, a_gpu)
  print(f"{a_host=}")

kernel_code = """
extern "C" __global__ void init_array(float* array, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    array[idx] = idx;
  }
}
"""

mod = SourceModule(kernel_code,
                   no_extern_c=True,
                   options=["-std=c++11",
                           "-Xcompiler",
                           "-fPIC"])

init_array = mod.get_function("init_array")

num_elements = 9
array_size_bytes = num_elements * np.float32().nbytes
array_gpu = cuda.mem_alloc(array_size_bytes)

init_array(array_gpu, np.int32(num_elements), block=(num_elements, 1, 1))

print_gpu_array(array_gpu, num_elements)
