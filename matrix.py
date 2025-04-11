from kernel_lib import *

# Basically a Tensor class definition...
class Matrix():
  def __init__(self, num_rows, num_cols, dtype, gpu=False):
    self.num_rows = int(num_rows)
    self.num_cols = int(num_cols)
    self.dtype = dtype
    self.gpu = gpu
    self.allocated_on_gpu = False
    self.allocated_on_host = False
    self.a_gpu = None
    self.a_host = None
    self.stride = 1
    self.start_index = 0
    self.shape = [self.num_rows, self.num_cols]
    # __rmul__ = __mul__

  def __str__(self):
    if (self.allocated_on_gpu):
      a_host = np.empty(self.num_rows * self.num_cols, np.float32)
      cuda.memcpy_dtoh(a_host, self.a_gpu)
      # np.set_printoptions(threshold=np.inf, precision=4, linewidth=120)
      a_host = a_host.reshape(self.num_rows, self.num_cols)
      return f"{a_host}"
  
  def __repr__(self):
    return self.__str__()

  #TODO: Change to matmul rather than mul...
  def __mul__(self, other):
    if (isinstance(other, Matrix)):
      if (self.allocated_on_gpu and other.allocated_on_host or
          self.allocated_on_host and other.allocated_on_gpu):
        raise MemoryError("Only support arrays on the same device type...")

      if (self.allocated_on_gpu and other.allocated_on_gpu):
        max_num_rows = max(self.num_rows, other.num_rows)
        max_num_cols = max(self.num_cols, other.num_cols)

        output = Matrix(self.num_rows, other.num_cols, self.dtype, gpu=True)
        output_gpu = cuda.mem_alloc(self.num_rows * self.num_cols * self.dtype().nbytes)
        output.set_gpu_matrix(output_gpu)

        regular_matmul(self.a_gpu, other.a_gpu,
                       np.int32(self.num_rows),
                       np.int32(other.num_cols),
                       np.int32(self.num_cols),
                       output_gpu, block=(max_num_cols, max_num_rows, 1))

        return output

      elif (self.allocated_on_host and other.allocated_on_host):
        raise ValueError("Not implemented arrays on hosts...")
      else:
        raise MemoryError("ERROR: Not sure if this is possible...")
    else:
      raise ValueError("Not implemented multiplies with different data types other than Matrix")

  def __truediv__(self, other):
    if (isinstance(other, int) or isinstance(other, float)):
      other = np.float32(other)
      if (self.allocated_on_gpu):
        output = Matrix(self.num_rows, self.num_cols, self.dtype, gpu=True)
        output_gpu = cuda.mem_alloc(self.num_rows * self.num_cols * self.dtype().nbytes)
        output.set_gpu_matrix(output_gpu)

        scalar_divide(self.a_gpu,
                      other,
                      np.int32(self.num_elements()),
                      output_gpu,
                      block=(self.num_elements(), 1, 1))

        return output
      elif (self.allocated_on_host):
        pass

    elif (isinstance(other, Matrix)):
      max_num_rows = max(self.num_rows, other.num_rows)
      max_num_cols = max(self.num_cols, other.num_cols)

      if (self.allocated_on_gpu and other.allocated_on_host or
          self.allocated_on_host and other.allocated_on_gpu):
        raise MemoryError("Gpus")

      if (self.allocated_on_gpu and other.allocated_on_gpu):
        output = Matrix(self.num_rows, other.num_cols, self.dtype, gpu=True)
        output_gpu = cuda.mem_alloc(self.num_rows * self.num_cols * self.dtype().nbytes)
        output.set_gpu_matrix(output_gpu)

        regular_matmul(self.a_gpu, other.a_gpu,
                       np.int32(self.num_rows),
                       np.int32(other.num_cols),
                       np.int32(self.num_cols),
                       output_gpu, block=(max_num_cols, max_num_rows, 1))

        return output

      elif (self.allocated_on_host and other.allocated_on_host):
        raise ValueError("Not implemented")
      else:
        raise MemoryError("ERROR: Not sure if this is possible...")
    else:
      raise ValueError("Not implemented")

  def num_elements(self):
    return self.num_rows * self.num_cols

  # Here's a thought experiment, is stride really the most
  # optimized storage strategy?...
  def set_gpu_matrix(self, a_gpu, stride=1, start_idx=0):
    self.allocated_on_gpu = True
    self.a_gpu = a_gpu
    self.stride = stride
    self.start_idx = start_idx

  def set_host_matrix(self, a_host, stride=1, start_idx=0):
    print("Allocated on host!")
    self.allocated_on_host = True
    self.a_host = a_host
    self.stride = stride
    self.start_idx = start_idx

  def alloc_on_gpu(self):
    self.allocated_on_gpu = True
    self.a_gpu = cuda.mem_alloc(self.num_elements() * self.dtype().nbytes)

  def alloc_on_host(self):
    # print("Allocated on host!")
    self.allocated_on_host = True
    self.a_host = np.empty(self.num_elements(), self.dtype)
    # print(f"Set {self.a_host=}")

  def copy_d_to_h(self):
    # Assumes D already allocated
    # print(f"{self.allocated_on_host=}")
    if not self.allocated_on_host:
      print(f"Allocating on host!")
      self.alloc_on_host()
    # print(f"{self.a_host=} {self.a_gpu=}")
    cuda.memcpy_dtoh(self.a_host, self.a_gpu)
    self.a_host = self.a_host.reshape(self.num_rows, self.num_cols)

  def copy_h_to_d(self):
    # Assumes H already allocated
    if not self.allocated_on_gpu:
      self.alloc_on_gpu()
    cuda.memcpy_htod(self.a_gpu, self.a_host)
  
  def init_ones(self):
    if (self.allocated_on_gpu):
      init_array_w_val(self.a_gpu, np.int32(1), np.int32(self.num_elements()), block=(self.num_elements(),1,1))
    elif (self.allocated_on_host):
      raise MemoryError("Not implemented")
  
  def init_incremental(self):
    if (self.allocated_on_gpu):
      init_array(self.a_gpu, np.int32(self.num_elements()), block=(self.num_elements(),1,1))
    elif (self.allocated_on_host):
      raise MemoryError("Not implemented")
  
  def init_uniform_rand(self, scale):
    if (self.allocated_on_gpu):
      seed = 0
      generate_uniform_random(self.a_gpu,
                        np.float32(scale),
                        np.int32(seed),
                        np.int32(self.num_elements()),
                        block=(self.num_elements(), 1, 1))
    else:
      raise MemoryError("Not implemented")

  def transpose(self):
    if (self.allocated_on_gpu):
      output = Matrix(self.num_cols, self.num_rows, self.dtype, gpu=True)
      output_gpu = cuda.mem_alloc(self.num_elements() * self.dtype().nbytes)
      output.set_gpu_matrix(output_gpu)

      matrix_transpose(self.a_gpu,
                       np.int32(self.num_rows),
                       np.int32(self.num_cols),
                       output.a_gpu,
                       block=(self.num_cols, self.num_rows, 1))
      
      return output
    else:
      raise MemoryError("Not implemented")
