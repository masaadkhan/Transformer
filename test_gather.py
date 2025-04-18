from kernel_lib import *
from matrix import Matrix
import math

test_big_matrix = Matrix(4, 12, np.float32, gpu=True)
test_big_matrix.alloc_on_gpu()
test_big_matrix.init_incremental()
test_big_matrix.copy_d_to_h()

small_1 = Matrix(4, 4, np.float32, gpu=True)
small_1.set_gpu_matrix(test_big_matrix.a_gpu, stride=test_big_matrix.num_cols, start_idx=(0 * test_big_matrix.num_cols) / 3)

small_2 = Matrix(4, 4, np.float32, gpu=True)
small_2.set_gpu_matrix(test_big_matrix.a_gpu, stride=test_big_matrix.num_cols, start_idx=(1 * test_big_matrix.num_cols) / 3)

small_3 = Matrix(4, 4, np.float32, gpu=True)
small_3.set_gpu_matrix(test_big_matrix.a_gpu, stride=test_big_matrix.num_cols, start_idx=(2 * test_big_matrix.num_cols) / 3)

print(f"{small_1.start_idx=} Beforehand...")
print(f"{small_2.start_idx=} Beforehand...")
print(f"{small_3.start_idx=} Beforehand...")

# print(f"{small_1=}")
print(f"{small_2=}")
# print(f"{small_3=}")

split_dim = int(test_big_matrix.a_host.shape[1] / 3)

small_1_np = test_big_matrix.a_host[:, : split_dim]
small_2_np = test_big_matrix.a_host[:, split_dim : 2 * split_dim]
small_3_np = test_big_matrix.a_host[:, 2 * split_dim :]

# print(f"{small_1_np=}")
print(f"{small_2_np=}")
# print(f"{small_3_np=}")
