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

eq_and_thresh_mod = SourceModule("""
__global__ void EqualsAndThresholdKernel(float *a, float *b, float threshold, int width, int height, bool *dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width) {
        return;
    }
    int flatIndex = y * width + x;
    dst[flatIndex] = (a[flatIndex] == b[flatIndex]) && (threshold < a[flatIndex]);
}
""")
eq_and_thresh = eq_and_thresh_mod.get_function("EqualsAndThresholdKernel")


def detect_naive(image, neighborhood, threshold=1e-12):
    """Detect peaks using naive image dilation kernel

    Parameters
    ----------
    image : numpy.ndarray (2D)
        The imagery to find the local maxima of. Note this will be reduced
        to 32-bit float precision due to GPU constraints.
    neighborhood : numpy.ndarray (2D)
        A boolean matrix specifying a scanning window for maxima detection.
        The neigborhood size is implicitly defined by the matrix dimensions.
        Note that the values of this matrix are currently ignored, but
        conceivably could be in the future for more complex windowing.
    threshold : float
        The minimum acceptable value of a peak

    Returns
    -------
    numpy.ndarray (2D)
        A boolean matrix specifying maxima locations (True) and background
        locations (False)
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

    # Detect the peaks (locations where original image equals dilated)
    gpu_detected_peaks = gpuarray.empty_like(gpu_image, dtype=numpy.bool)
    eq_and_thresh(gpu_image.gpudata,
                  gpu_dilated_image.gpudata,
                  numpy.float32(threshold),
                  numpy.int32(image.shape[0]),
                  numpy.int32(image.shape[1]),
                  gpu_detected_peaks.gpudata,
                  block=block,
                  grid=grid)
    return gpu_detected_peaks.get()
