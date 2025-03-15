import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np

CUDA_THREADS_PER_BLOCK = 1024

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

def print_gpu_array(a_gpu, num_elements):
  a_host = np.empty(num_elements, np.float32)
  cuda.memcpy_dtoh(a_host, a_gpu)
  print(f"{a_host=}")

def run_one_iter(array_gpu, num_elements, num_threads, num_blocks):
  print(f"Attempting to set {num_threads=} {num_blocks=}")
  start = cuda.Event()
  end = cuda.Event()

  start.record()
  init_array(array_gpu, np.int32(num_elements), block=(num_threads, 1, 1), grid=(num_blocks, 1, 1))
  end.record()
  end.synchronize()

  elapsed_time = start.time_till(end)

  print_gpu_array(array_gpu, num_elements)
  print(f"{num_threads=} {num_blocks=} {elapsed_time=}")
  return elapsed_time

init_array = mod.get_function("init_array")

if __name__ == "__main__":
  num_elements = CUDA_THREADS_PER_BLOCK * 10
  array_size_bytes = num_elements * np.float32().nbytes
  array_gpu = cuda.mem_alloc(array_size_bytes)

  # start with 1024 threads
  num_threads = CUDA_THREADS_PER_BLOCK
  num_blocks = (num_elements + CUDA_THREADS_PER_BLOCK - 1) // CUDA_THREADS_PER_BLOCK

  high_block_cfg = (1, num_elements)
  mid_block_cfg = (32, 320)
  low_block_cfg = (CUDA_THREADS_PER_BLOCK, 10)

  while (True):
    high_ = run_one_iter(array_gpu, num_elements, high_block_cfg[0], high_block_cfg[1])
    mid_ = run_one_iter(array_gpu, num_elements, mid_block_cfg[0], mid_block_cfg[1])
    low_ = run_one_iter(array_gpu, num_elements, low_block_cfg[0], low_block_cfg[1])

    high_diff = mid_ - high_
    low_diff = mid_ - low_

    if (high_diff < 0 and low_diff < 0):
      print(f"mid seems to be best {mid_} {mid_block_cfg}")
      break

    # We're looking for the largest positive diff
    if (high_diff >= low_diff):
      low_block_cfg = (mid_block_cfg[0] // 2, mid_block_cfg[1] * 2)
      mid_block_cfg = (high_block_cfg[0] // 2, high_block_cfg[1] * 2)
    elif (low_diff >= high_diff):
      if ((low_block_cfg[0] * 2) > CUDA_THREADS_PER_BLOCK):
        print(f"{low_block_cfg=} was the best...")
        break
      else:
        high_block_cfg = (mid_block_cfg[0] * 2, (mid_block_cfg[1] + 2 - 1) // 2)
        mid_block_cfg = (low_block_cfg[0] * 2, (low_block_cfg[1] + 2 - 1) // 2)
    else:
      print(f"mid2 seems to be best {mid_} {mid_block_cfg}")
      break
