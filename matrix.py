import numpy as np
import kernel_lib as kl

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
    self.start_idx = 0
    self.shape = [self.num_rows, self.num_cols]
    self.modified_since_last_print = 1
    self.needs_unstriding = 0
    self.stride_mat = None
    # __rmul__ = __mul__

  @staticmethod
  def modified(func):
    def wrapper(self, *args, **kwargs):
      self.modified_since_last_print = 1
      return func(self, *args, **kwargs)
    return wrapper

  @staticmethod
  def strided(func):
    def wrapper(self, *args, **kwargs):
      self.needs_unstriding = 1
      return func(self, *args, **kwargs)
    return wrapper

  @staticmethod
  def printed(func):
    def wrapper(self, *args, **kwargs):
      val = func(self, *args, **kwargs)
      self.modified_since_last_print = 0
      self.needs_unstriding = 0
      return val
    return wrapper

  @printed
  def __str__(self):
    if (self.modified_since_last_print):
      if (self.allocated_on_gpu):
        if (not self.allocated_on_host):
          print(f"Allocating {self.num_elements()} on host to print this matrix!")
          self.alloc_on_host()

        if (self.stride != 1 and self.needs_unstriding):
          # print(f"Allocating {self.num_elements()} on GPU to destride!")
          print("Gathering the elements for this matrix....")
          output_gpu = kl.cuda.mem_alloc(self.num_elements() * self.dtype().nbytes)

          # TODO: Technically the original GPU matrix may need to be deallocated...
          # Otherwise memory leak in GPU memory depends on the # of references left...

          kl.gather(self.a_gpu,
                 np.int32(self.start_idx),
                 np.int32(self.stride),
                 np.int32(self.num_rows),
                 np.int32(self.num_cols),
                 output_gpu,
                 block=(self.num_cols, self.num_rows, 1))
          self.a_gpu = output_gpu

        self.copy_d_to_h()
        self.a_host = self.a_host.reshape(self.num_rows, self.num_cols)
    else:
      print("Returning the cached matrix value!")

    return f"{self.a_host}"

  def __repr__(self):
    return self.__str__()

  def compare(self, other):
    if (isinstance(other, np.ndarray)):
      self.copy_d_to_h()
      return np.allclose(self.a_host, other, rtol=1e-5, atol=1e-8)
    else:
      raise MemoryError("Not implemented!")

  def shapes_match(self, other):
    return self.shape == other.shape

  @modified
  def __add__(self, other):
    if (isinstance(other, Matrix)):
      if (self.allocated_on_gpu and other.allocated_on_gpu):
        # TODO: Matrix + Vector
        # if (self.a_gpu.num_rows == other.a_gpu.num_rows and
        #     self.a_gpu.num_cols == 1 or other.a_gpu.num_cols == 1):
          
        if (self.shapes_match(other)):
          output = Matrix(self.num_rows, self.num_cols, self.dtype, gpu=True)
          output.alloc_on_gpu()

          kl.regular_add(self.a_gpu, other.a_gpu,
                      np.int32(self.num_elements()),
                      output.a_gpu,
                      block=(self.num_elements(), 1, 1))

          return output
        else:
          raise MemoryError("Matrix shapes do not match for add operation...")
      else:
        raise MemoryError("Not implemented")
    else:
      raise MemoryError("Not implemented")

  @modified
  #TODO: Change to matmul rather than mul...
  def __mul__(self, other):
    if (isinstance(other, Matrix)):
      if (self.allocated_on_gpu and other.allocated_on_gpu):
        max_num_rows = max(self.num_rows, other.num_rows)
        max_num_cols = max(self.num_cols, other.num_cols)

        output = Matrix(self.num_rows, other.num_cols, self.dtype, gpu=True)
        #TODO: Fix this cuda memalloc to use the library
        output_gpu = kl.cuda.mem_alloc(self.num_elements() * self.dtype().nbytes)
        output.set_gpu_matrix(output_gpu)

        kl.regular_matmul(self.a_gpu,
                          other.a_gpu,
                          np.int32(self.num_rows),
                          np.int32(other.num_cols),
                          np.int32(self.num_cols),
                          output_gpu,
                          block=(max_num_cols, max_num_rows, 1))
        kl.cuda.Context.synchronize()

        return output

      else:
        raise MemoryError("Not implemented...")
    else:
      raise ValueError("Not implemented multiplies with different data types other than Matrix")

  @modified
  def __truediv__(self, other):
    if (isinstance(other, int) or isinstance(other, float)):
      other = np.float32(other)
      if (self.allocated_on_gpu):
        output = Matrix(self.num_rows, self.num_cols, self.dtype, gpu=True)
        #TODO: Fix this cuda memalloc replace with method
        output_gpu = kl.cuda.mem_alloc(self.num_elements() * self.dtype().nbytes)
        output.set_gpu_matrix(output_gpu)

        kl.scalar_divide(self.a_gpu,
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
        #TODO: Fix this cuda memalloc
        output_gpu = kl.cuda.mem_alloc(self.num_elements() * self.dtype().nbytes)
        output.set_gpu_matrix(output_gpu)

        kl.regular_matmul(self.a_gpu, other.a_gpu,
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
  @strided
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
    self.a_gpu = kl.cuda.mem_alloc(self.num_elements() * self.dtype().nbytes)

  def alloc_on_host(self):
    # TODO: Kind of implies that this is pre-allocated on GPU I think...
    self.allocated_on_host = True
    print(f"Allocating {self.num_elements()} {self.dtype} onto the host...")
    self.a_host = np.empty(self.num_elements(), self.dtype)
    # self.a_host = self.a_host.reshape(self.num_rows, self.num_cols)

  def copy_d_to_h(self):
    if not self.allocated_on_host:
      print(f"Lazily allocating on host!")
      self.alloc_on_host()
    else:
      print("Pre-allocated on host...")

    kl.cuda.memcpy_dtoh(self.a_host, self.a_gpu)
    self.a_host = self.a_host.reshape(self.num_rows, self.num_cols)

  def copy_h_to_d(self):
    # Assumes H already allocated
    if not self.allocated_on_gpu:
      self.alloc_on_gpu()
    kl.cuda.memcpy_htod(self.a_gpu, self.a_host)

  @modified
  def init_ones(self):
    if (self.allocated_on_gpu):
      kl.init_array_w_val(self.a_gpu, np.int32(1), np.int32(self.num_elements()), block=(self.num_elements(),1,1))
    elif (self.allocated_on_host):
      raise MemoryError("Not implemented")

  @modified
  def init_incremental(self):
    if (self.allocated_on_gpu):
      kl.init_array(self.a_gpu, np.int32(self.num_elements()), block=(self.num_elements(),1,1))
    elif (self.allocated_on_host):
      raise MemoryError("Not implemented")

  # extern "C" __global__ void generate_uniform_random(float* numbers, float scale, int seed, int N) {{
  @modified
  def init_uniform_rand(self, scale):
    if (self.allocated_on_gpu):
      seed = 0
      kl.compile_kernels()
      kl.generate_uniform_random(self.a_gpu,
                        np.float32(scale),
                        np.int32(seed),
                        np.int32(self.num_elements()),
                        block=(self.num_elements(), 1, 1))
    else:
      raise MemoryError("Not implemented")

  @modified
  def transpose(self):
    if (self.allocated_on_gpu):
      output = Matrix(self.num_cols, self.num_rows, self.dtype, gpu=True)
      #TODO: Fix this cuda memalloc
      output_gpu = kl.cuda.mem_alloc(self.num_elements() * self.dtype().nbytes)
      output.set_gpu_matrix(output_gpu)

      kl.matrix_transpose(self.a_gpu,
                       np.int32(self.num_rows),
                       np.int32(self.num_cols),
                       output.a_gpu,
                       block=(self.num_cols, self.num_rows, 1))
      
      return output
    else:
      raise MemoryError("Not implemented")
