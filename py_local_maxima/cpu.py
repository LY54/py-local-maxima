from scipy.ndimage.filters import maximum_filter as _max_filter
from scipy.ndimage.morphology import binary_erosion as _binary_erosion


def detect_maximum_filter(image, neighborhood):
    """Detect peaks using a local maximum filter

    Code courtesy https://stackoverflow.com/a/3689710 (adapted slightly).

    Parameters
    ----------
    image : numpy.ndarray (2D)
        The imagery to find the local maxima of
    neighborhood : numpy.ndarray (2D)
        A boolean matrix specifying a scanning window for maxima detection.
        The neigborhood size is implicitly defined by the matrix dimensions.

    Returns
    -------
    numpy.ndarray (2D)
        A boolean matrix specifying maxima locations (True) and background
        locations (False)
    """

    # Apply the local maximum filter; all pixels of maximal value in their
    # neighborhood are set to 1
    local_max = _max_filter(image, footprint=neighborhood) == image

    # We must erode the background in order to successfully subtract it from
    # local_max, otherwise a line will appear along the background border
    background = (image == 0)
    eroded_background = _binary_erosion(background,
                                        structure=neighborhood,
                                        border_value=1)

    detected_peaks = local_max ^ eroded_background
    return detected_peaks
