from matrix import Matrix
from kernel_lib import *

test_a = Matrix(4,4,np.float32,gpu=True)
test_a.alloc_on_gpu()
test_a.init_incremental()

row_sum = Matrix(4,1,np.float32,gpu=True)
row_sum.alloc_on_gpu()
row_sum.init_incremental()

print(f"{test_a=}")
print(f"Before: {row_sum=}")

matrix_row_wise_add(test_a.a_gpu,
        np.int32(test_a.num_rows),
        np.int32(test_a.num_cols),
        row_sum.a_gpu,
        block=(test_a.num_cols, test_a.num_rows, 1),
        shared=test_a.num_elements() * test_a.dtype().nbytes)

print(f"After: {row_sum=}")

