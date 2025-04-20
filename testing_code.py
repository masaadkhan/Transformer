from matrix import Matrix
from kernel_lib import *

kernel_code = """
// Takes a matrix and performs row-wise add reduction and outputs a vector representing each row's sum
extern "C" __global__ void matrix_row_wise_add(float* matrix, int num_rows, int num_cols, float* output_vec) {
  extern __shared__ float f[];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < num_rows && col < num_cols) {
    int idx = row * num_cols + col;
    f[idx] = matrix[idx];
    __syncthreads();

    // printf("f[%d] = %f %f\\n", idx, f[idx], matrix[idx]);
    float* row_ptr = &f[row * num_cols];

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

    for (int i = num_cols; i > 1; i = i / 2) {
      if (threadIdx.x < i) {
        int idx1 = threadIdx.x;
        int idx2 = threadIdx.x + i / 2;

        float a_ptr = row_ptr[idx1];
        float b_ptr = row_ptr[idx2];

        //printf("threadIdx.x: %d, idx1: %d, a: %f idx2: %d, b: %f\\n", threadIdx.x, idx1, row_ptr[idx1], idx2, row_ptr[idx2]);
        //printf("sum: %f\\n", a_ptr + b_ptr);

        row_ptr[idx1] = a_ptr + b_ptr;
        if (i == 2) {
          output_vec[row] = row_ptr[idx1];
        }
      }
    }
  }
}
"""

mod = SourceModule(kernel_code,
                   no_extern_c=True,
                   options=["-std=c++11",
                           "-Xcompiler",
                           "-fPIC"])

matrix_row_wise_add = mod.get_function("matrix_row_wise_add")

test_a = Matrix(4,4,np.float32,gpu=True)
test_a.alloc_on_gpu()
test_a.init_incremental()

test_output = Matrix(4,1,np.float32,gpu=True)
test_output.alloc_on_gpu()
test_output.init_incremental()

print(f"{test_a=}")
print(f"Before: {test_output=}")

# print(f"Kicked off {test_a.num_cols * test_a.num_rows} threads!")
# print(f"Allocated {test_a.num_elements()*test_a.dtype().nbytes} bytes")

matrix_row_wise_add(test_a.a_gpu,
                    np.int32(test_a.num_rows),
                    np.int32(test_a.num_cols),
                    test_output.a_gpu,
                    block=(test_a.num_cols,test_a.num_rows,1),
                    shared=test_a.num_elements() * test_a.dtype().nbytes)

print(f"After: {test_output=}")

# TODO(MASAAD): Ideally make this coalesced to the warp....
kernel_code = """
// Takes the provided row-wise sum and divides each element respectively
extern "C" __global__ void softmax(float* matrix, float* vector, int num_rows, int num_cols, float* output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < num_rows && col < num_cols) {
    int idx = row * num_cols + col;
    matrix[idx] /= vector[row];
  }
}
"""

mod = SourceModule(kernel_code,
                   no_extern_c=True,
                   options=["-std=c++11",
                           "-Xcompiler",
                           "-fPIC"])

softmax = mod.get_function("softmax")

matrix_row_wise_add(test_a.a_gpu,
                    np.int32(test_a.num_rows),
                    np.int32(test_a.num_cols),
                    test_output.a_gpu,
                    block=(test_a.num_cols,test_a.num_rows,1),
                    shared=test_a.num_elements() * test_a.dtype().nbytes)

softmax(test_a.a_gpu,
        test_output.a_gpu,
        np.int32(test_a.num_rows),
        np.int32(test_a.num_cols),
        test_a.a_gpu,
        block=(test_a.num_cols,test_a.num_rows,1))

print(test_a)