from kernel_lib import *

a_rows = 4
a_cols = 4

b_rows = 4
b_cols = 4

c_rows = 4
c_cols = 4

dummy_a = cuda.mem_alloc(a_rows * a_cols * 4)
dummy_b = cuda.mem_alloc(b_rows * b_cols * 4)
dummy_c = cuda.mem_alloc(c_rows * c_cols * 4)

init_array(dummy_a, np.int32(a_rows * a_cols), block=(a_rows * a_cols,1,1))
init_array(dummy_b, np.int32(b_rows * b_cols), block=(b_rows * b_cols,1,1))
init_array(dummy_c, np.int32(c_rows * c_cols), block=(c_rows * c_cols,1,1))

cuda.Context.synchronize()

host_a = np.empty(a_rows * a_cols, np.float32)
host_b = np.empty(b_rows * b_cols, np.float32)
host_c = np.empty(c_rows * c_cols, np.float32)

cuda.memcpy_dtoh(host_a, dummy_a)
cuda.memcpy_dtoh(host_b, dummy_b)

regular_matmul(dummy_a, dummy_b, np.int32(c_rows), np.int32(c_cols), np.int32(a_cols), dummy_c, block=(c_rows,c_cols,1))
cuda.Context.synchronize()
cuda.memcpy_dtoh(host_c, dummy_c)

# print_gpu_array(dummy_c, "dummy_c", c_rows * c_cols, shape=[c_rows,c_cols])

a_h = host_a.reshape(a_rows,a_cols)
b_h = host_b.reshape(b_rows,b_cols)
c_h = host_c.reshape(c_rows,c_cols)

print(f"{a_h=}")
print(f"{b_h=}")

expected_result = a_h @ b_h

print(f"{c_h=}")
print(f"{expected_result=}")

if (np.allclose(c_h, expected_result)):
  print("Matmul successful")
else:
  print("Matmul did not match GPU...")