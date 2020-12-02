import py_local_maxima
import numpy as np


def read_dog_paws(filepath):
    image = np.loadtxt(data_file)
    image = [p.squeeze() for p in np.vsplit(image, 4)][0]
    return image


def read_heatmap(filepath):
    return np.load(filepath)


# TUNABLE PARAMETERS
data_file = "/users/PES0808/hollarbl/CS 5168/final_project/py-local-maxima/py_local_maxima/data/heatmap.npy"
data_reader = read_heatmap
neighborhood = np.ones((31, 31))
benchmark_iterations = 10

# Core functionality
data = data_reader(data_file)
py_local_maxima.benchmark(data, neighborhood, benchmark_iterations)
py_local_maxima.evaluate(data, neighborhood)
