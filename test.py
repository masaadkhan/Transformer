from kernel_lib import *

# a_rows = 3
# a_cols = 2

# b_rows = 2
# b_cols = 4

# c_rows = 3
# c_cols = 4

def func(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
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

  num_row_threads = max(a_rows, b_rows)
  num_col_threads = max(a_cols, b_cols)

  print(f"Running {num_row_threads} * {num_col_threads} threads")

  regular_matmul(dummy_a, dummy_b, np.int32(c_rows), np.int32(c_cols), np.int32(a_cols), dummy_c, block=(num_col_threads,num_row_threads,1))
  cuda.Context.synchronize()
  cuda.memcpy_dtoh(host_c, dummy_c)

  # print_gpu_array(dummy_c, "dummy_c", c_rows * c_cols, shape=[c_rows,c_cols])

  a_h = host_a.reshape(a_rows,a_cols)
  b_h = host_b.reshape(b_rows,b_cols)
  c_h = host_c.reshape(c_rows,c_cols)

  expected_result = a_h @ b_h

  if (np.allclose(c_h, expected_result)):
    print("Matmul successful")
    return 0
  else:
    print(f"{a_h=}")
    print(f"{b_h=}")
    print(f"{c_h=}")
    print(f"{expected_result=}")

    print("Matmul did not match GPU...")
    return 1

# M N K
# 0 0 0
print("\nRunning first test\n")
if (func(5, 5,
         5, 5,
         5, 5)):
  exit("First test failed")

# 0 0 1
print("\nRunning first test\n")
if (func(5, 10,
         10, 5,
         5, 5)):
  exit("First test failed")

print("\nRunning first test\n")
if (func(5, 4,
         4, 5,
         5, 5)):
  exit("First test failed")

# 0 1 0
print("\nRunning first test\n")
if (func(5, 5,
         5, 4,
         5, 4)):
  exit("First test failed")

print("\nRunning first test\n")
if (func(5, 5,
         5, 6,
         5, 6)):
  exit("First test failed")

# 0 1 1
print("\nRunning first test\n")
if (func(5, 4,
         4, 4,
         5, 4)):
  exit("First test failed")

print("\nRunning first test\n")
if (func(5, 6,
         6, 6,
         5, 6)):
  exit("First test failed")

# 1 0 0
print("\nRunning first test\n")
if (func(10, 5,
         5, 5,
         10, 5)):
  exit("First test failed")

print("\nRunning first test\n")
if (func(4, 5,
         5, 5,
         4, 5)):
  exit("First test failed")

# 1 0 1
print("\nRunning first test\n")
if (func(10, 8,
         8, 5,
         10, 5)):
  exit("First test failed")

print("\nRunning first test\n")
if (func(4, 4,
         4, 5,
         4, 5)):
  exit("First test failed")

# 1 1 0
print("\nRunning first test\n")
if (func(10, 5,
         5, 4,
         10, 4)):
  exit("First test failed")

print("\nRunning first test\n")
if (func(10, 5,
         5, 10,
         10, 10)):
  exit("First test failed")

print("\nRunning first test\n")
if (func(4, 5,
         5, 4,
         4, 4)):
  exit("First test failed")

# 1 1 1
print("\nRunning first test\n")
if (func(15, 8,
         8, 6,
         15, 6)):
  exit("First test failed")

print("\nRunning first test\n")
if (func(4, 4,
         4, 4,
         4, 4)):
  exit("First test failed")

# print("\nRunning first test\n")
# if (func(4, 5,
#          5, 5,
#          4, 5)):
#   exit("First test failed")

# print("\nRunning first test\n")
# if (func(4, 5,
#          5, 5,
#          4, 5)):
#   exit("First test failed")

# # print("\nRunning fourth test\n")
# # if (func(5, 5,
# #          10, 5,
# #          5, 5)):
# #   exit("Fourth test failed")

# # print("\nRunning fourth test\n")
# # if (func(10, 5,
# #          5, 10,
# #          10, 10)):
# #   exit("Fourth test failed")

# # print("\nRunning fourth test\n")
# # if (func(6, 20,
# #          20, 5,
# #          6, 5)):
# #   exit("Fourth test failed")

# # print("\nRunning fourth test\n")
# # if (func(5, 20,
# #          20, 6,
# #          5, 6)):
# #   exit("Fourth test failed")

# # print("\nRunning second test\n")
# # if (func(20, 4,
# #          4, 4,
# #          20, 4)):
# #   exit("Second test failed")