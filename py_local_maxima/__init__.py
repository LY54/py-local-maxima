from .benchmark import benchmark, evaluate
from . import cpu
try:
    from . import gpu
except ImportError:
    print('PyCuda cannot be loaded. GPU code unavailable.')
