from kernel_lib import *

a_rows = 4
a_cols = 4

b_cols = 4

c_rows = 4
c_cols = 4

dummy_a = cuda.mem_alloc(a_rows * a_cols * 4)
dummy_b = cuda.mem_alloc(b_cols * 4)
dummy_c = cuda.mem_alloc(c_rows * c_cols * 4)

init_array(dummy_a, np.int32(a_rows * a_cols), block=(a_rows * a_cols,1,1))
init_array(dummy_b, np.int32(b_cols), block=(b_cols,1,1))

add_matrix_w_vector(dummy_a, dummy_b, np.int32(a_rows), np.int32(a_cols), dummy_c, block=(c_cols, c_rows, 1))

print_gpu_array(dummy_a, "dummy_a", a_rows * a_cols, shape=[a_rows,a_cols])
print_gpu_array(dummy_b, "dummy_b", b_cols, shape=[1,b_cols])
print_gpu_array(dummy_c, "dummy_c", c_rows * c_cols, shape=[c_rows,c_cols])
