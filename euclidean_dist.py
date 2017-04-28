import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

mod = SourceModule("""
  __global__ void euclidean_dist(float *dest, float *a, float *b)
  {
    //int idx = threadIdx.x + threadIdx.y*4;
    int idx = threadIdx.x;
    float sum = 0.0;
    float diff = 0.0;
    for(int i=0; i<4; ++i)
    {
        diff = a[idx*4+i]-b[idx*4+i];
        sum += diff*diff;
    }
    dest[idx] = sum;
  }
  """)

a = numpy.random.randn(4,4).astype(numpy.float32)
b = numpy.random.randn(4,4).astype(numpy.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

dest = numpy.zeros((4,1)).astype(numpy.float32)
dest_gpu = cuda.mem_alloc(dest.nbytes)

func = mod.get_function("euclidean_dist")
func(dest_gpu, a_gpu, b_gpu, block=(4,1,1))

cuda.memcpy_dtoh(dest, dest_gpu)
print a
print b
print dest