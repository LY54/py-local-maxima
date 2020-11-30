import numpy as np
import matplotlib.pyplot as plt
import math
import py_local_maxima
from timeit import timeit


_tests = [
    ('CPU Max Filter',
     'py_local_maxima.cpu.detect_maximum_filter(image, neighborhood)'),
    ('CPU skimage Implementation',
     'py_local_maxima.cpu.detect_skimage(image, neighborhood)'),
]
_setup = 'import py_local_maxima'


def _print_benchmark(algorithm_name, timer, neighborhood_shape):
    template = '{0:30}: {1:5f} seconds per run ({2} neighborhood)'
    print(str.format(template, algorithm_name, timer, neighborhood_shape))


def benchmark(image, neighborhood=np.ones((3, 3)), iterations=100):
    """TODO
    """

    for test in _tests:
        t = timeit(test[1],
                   setup=_setup,
                   number=iterations,
                   globals=locals())
        t = t / iterations
        _print_benchmark(test[0], t, neighborhood.shape)


def evaluate(image, neighborhood=np.ones((3, 3))):
    """TODO
    """

    for i, test in enumerate(_tests):
        maxima_mask = eval(test[1])
        maxima_y, maxima_x = np.where(maxima_mask)
        plt.subplot(2, math.ceil(len(_tests) / 2.0), i + 1)
        plt.imshow(image)
        plt.scatter(maxima_x, maxima_y, s=10, c='k', marker='x')
        plt.title(test[0])
        plt.xticks([])
        plt.yticks([])
    plt.show()
