import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy
import math


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


def detect_naive(image, neighborhood, threshold=1e-12):
    """TODO

    NOTE: `neighborhood` should become an integer, not a boolean matrix as in
          cpu code. It specifies the side length of a square window and is
          expected to be odd
    """

    # Image must be 32-bit for GPU code to work at all. Force cast our input
    image = image.astype(numpy.float32)

    # Initialize GPU arrays (on device) and some required CUDA inputs
    gpu_image = gpuarray.to_gpu(image)
    gpu_dilated_image = gpuarray.to_gpu(image)
    block = (32, 32, 1)
    grid = (math.ceil(image.shape[0] / block[0]),
            math.ceil(image.shape[1] / block[1]),
            1)

    # Do the max filter (which is a 2D image dilation)
    naive_dilation(gpu_image.gpudata,
                   gpu_dilated_image.gpudata,
                   numpy.int32(image.shape[0]),
                   numpy.int32(image.shape[1]),
                   numpy.int32(neighborhood.shape[0]),
                   block=block,
                   grid=grid)

    # TODO: Perform peak-finding on GPU, not CPU
    detected_peaks = gpu_dilated_image.get() == image
    detected_peaks[image < threshold] = False
    return detected_peaks
