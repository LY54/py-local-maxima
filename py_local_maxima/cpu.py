from scipy.ndimage.filters import maximum_filter as _max_filter
from scipy.ndimage.morphology import binary_erosion as _binary_erosion
from skimage.feature import peak_local_max


def detect_skimage(image, neighborhood, threshold=1e-12):
    """Detect peaks using a local maximum filter (via skimage)

    Parameters
    ----------
    image : numpy.ndarray (2D)
        The imagery to find the local maxima of
    neighborhood : numpy.ndarray (2D)
        A boolean matrix specifying a scanning window for maxima detection.
        The neigborhood size is implicitly defined by the matrix dimensions.
    threshold : float
        The minimum acceptable value of a peak

    Returns
    -------
    numpy.ndarray (2D)
        A boolean matrix specifying maxima locations (True) and background
        locations (False)
    """
    return peak_local_max(image,
                          footprint=neighborhood,
                          threshold_abs=threshold,
                          indices=False)


def detect_maximum_filter(image, neighborhood, threshold=1e-12):
    """Detect peaks using a local maximum filter

    Code courtesy https://stackoverflow.com/a/3689710 (adapted slightly).

    Parameters
    ----------
    image : numpy.ndarray (2D)
        The imagery to find the local maxima of
    neighborhood : numpy.ndarray (2D)
        A boolean matrix specifying a scanning window for maxima detection.
        The neigborhood size is implicitly defined by the matrix dimensions.
    threshold : float
        The minimum acceptable value of a peak

    Returns
    -------
    numpy.ndarray (2D)
        A boolean matrix specifying maxima locations (True) and background
        locations (False)
    """

    # Apply the local maximum filter, then remove any background (below
    # threshold) values from our result.
    detected_peaks = _max_filter(image, footprint=neighborhood) == image
    detected_peaks[image < threshold] = False
    return detected_peaks
