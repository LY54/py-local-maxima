import numpy as np
from timeit import timeit


_tests = [
    ('CPU Max Filter',
     'py_local_maxima.cpu.detect_maximum_filter(image, neighborhood)'),
]
_setup = 'import py_local_maxima'


def _print_benchmark(algorithm_name, timer):
    template = '{0}: {1:5f} seconds per run'
    print(str.format(template, algorithm_name, timer))


def benchmark(image, neighborhood=np.ones((3, 3)), iterations=100):
    """TODO
    """

    for test in _tests:
        t = timeit(test[1],
                   setup=_setup,
                   number=iterations,
                   globals=locals())
        t = t / iterations
        _print_benchmark(test[0], t)
