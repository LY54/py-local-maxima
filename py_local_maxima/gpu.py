import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy
import math
from scipy.ndimage.morphology import grey_dilation


naive_dilation_mod = SourceModule("""
#define FLT_MAX     3.40282347E+38F

__global__ void NaiveDilationKernel(float *src, float *dst, int width, int height, int window_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width) {
        return;
    }
    window_size = window_size / 2;
    unsigned int start_i = max(y - window_size, 0);
    unsigned int end_i = min(height - 1, y + window_size);
    unsigned int start_j = max(x - window_size, 0);
    unsigned int end_j = min(width - 1, x + window_size);
    float value = -FLT_MAX;
    for (int i = start_i; i <= end_i; i++) {
        for (int j = start_j; j <= end_j; j++) {
            value = max(value, src[i * width + j]);
        }
    }
    dst[y * width + x] = value;
} 
""")

naive_dilation = naive_dilation_mod.get_function("NaiveDilationKernel")

src = numpy.random.randn(4, 4).astype(numpy.float32)
dst = numpy.copy(src)

print('Source:')
print(src)
print()

window_size = 3
block = (32,32, 1)
grid = (math.ceil(src.shape[1] / block[0]), math.ceil(src.shape[0] / block[1]), 1)
naive_dilation(cuda.In(src), 
               cuda.Out(dst),
               numpy.int32(src.shape[1]),
               numpy.int32(src.shape[0]),
               numpy.int32(window_size),
               block=block,
               grid=grid)

print('GPU:')
print(dst)
print()

truth = grey_dilation(src, (window_size, window_size))

print('Truth (CPU):')
print(truth)
print()

print('Comparison:')
print(truth == dst)
print()
