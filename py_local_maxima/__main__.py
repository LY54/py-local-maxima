import py_local_maxima
import numpy as np


def read_dog_paws(filepath):
    image = np.loadtxt(data_file)
    image = [p.squeeze() for p in np.vsplit(image, 4)][0]
    return image


def read_heatmap(filepath):
    return np.load(filepath)


# TUNABLE PARAMETERS
data_file = "/Users/benhollar/Documents/College/2020 - Fall/CS 5168/Final Project/py-local-maxima/py_local_maxima/data/heatmap.npy"
data_reader = read_heatmap
neighborhood = np.ones((100, 100))
benchmark_iterations = 1

# Core functionality
data = data_reader(data_file)
py_local_maxima.benchmark(data, neighborhood, benchmark_iterations)
py_local_maxima.evaluate(data, neighborhood)
