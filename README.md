# py-local-maxima

This is an experimental repository dedicated to detecting peaks -- local maxima -- of 2D grayscale imagery. Primarily,
it is designed to compare the execution (in terms of speed and accuracy) of some "better-known" CPU algorithms and
their counterparts as written for GPUs in CUDA.

The inspiration for this experiment was the CS 5168 class (Parallel Programming) at the University of Cincinnati, where
students were asked to design a project focused on optimizing some problem using parallel programming techniques.

## Requirements

To do the bare minimum, this code expects the following Python environment:

* Python 3.x.x (developed using Python 3.6.12)
* Packages:
  * `numpy` (used throughout the project)
  * `matplotlib` (used for plotting peak detection output)
  * `scipy` (used in CPU algorithms)
  * `scikit-image` (used in CPU algorithms)

Strictly speaking, the GPU code was written in such a way as to be optional. However, since it is critical to the intent
of the project, users are also expected to:

* Have a computer with a CUDA-capable GPU
* Install the `pycuda` package

## Usage Example

In a Python environment as described above, the code can be run as a module using the following command:

```raw
python -m py_local_maxima
```

Note: the above command may not work unless the line `data_file = ...` in `__main__py` has been changed to an
appropriate path for your system. Example data is included in the repository under the `data/` directory.

Alternatively, feel free to `import py_local_maxima` in a Python script of your own and use bits and pieces of code as
you wish.
