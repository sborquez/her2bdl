"""Stain normalization classes.

This code comes from the tiatoolbox

https://github.com/TissueImageAnalytics/tiatoolbox

"""
import numpy as np


def rgb2od(img):
    r"""Convert from RGB to optical density (:math:`OD_{RGB}`) space.
    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}
    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
            RGB image.
    Returns:
        :class:`numpy.ndarray`:
            Optical density (OD) RGB image.
    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)
    """
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def od2rgb(od):
    r"""Convert from optical density (:math:`OD_{RGB}`) to RGB.
    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}
    Args:
        od (:class:`numpy.ndarray`):
            Optical density (OD) RGB image.
    Returns:
        :class:`numpy.ndarray`:
            RGB Image.
    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)
        >>> rgb_img = transforms.od2rgb(od_img)
    """
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)


class RuifrokExtractor:
    """Reuifrok stain extractor.
    Get the stain matrix as defined in:
    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Examples:
        >>> from tiatoolbox.tools.stainextract import RuifrokExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = RuifrokExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)
    """

    def __init__(self):
        #self.__stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])  # HE
        self.__stain_matrix = np.array([[0.650, 0.704, 0.286], [0.268, 0.570, 0.776]])  # HD

    def get_stain_matrix(self, _):
        """Get the pre-defined stain matrix.
        Returns:
            :class:`numpy.ndarray`:
                Pre-defined  stain matrix.
        """
        return self.__stain_matrix.copy()


class StainNormalizer:
    """Stain normalization base class.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Attributes:
        extractor (CustomExtractor, RuifrokExtractor):
            Method specific stain extractor.
        stain_matrix_target (:class:`numpy.ndarray`):
            Stain matrix of target.
        target_concentrations (:class:`numpy.ndarray`):
            Stain concentration matrix of target.
        maxC_target (:class:`numpy.ndarray`):
            99th percentile of each stain.
        stain_matrix_target_RGB (:class:`numpy.ndarray`):
            Target stain matrix in RGB.
    """

    def __init__(self):
        self.extractor = RuifrokExtractor()
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None
        self.stain_matrix_target_RGB = None

    @staticmethod
    def get_concentrations(img, stain_matrix):
        """Estimate concentration matrix given an image and stain matrix.
        Args:
            img (:class:`numpy.ndarray`):
                Input image.
            stain_matrix (:class:`numpy.ndarray`):
                Stain matrix for haematoxylin and eosin stains.
        Returns:
            numpy.ndarray:
                Stain concentrations of input image.
        """
        od = rgb2od(img).reshape((-1, 3))
        x, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=-1)
        return x.T

    def fit(self, target):
        """Fit to a target image.
        Args:
            target (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
              Target/reference image.
        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = self.get_concentrations(
            target, self.stain_matrix_target
        )
        self.maxC_target = np.percentile(
            self.target_concentrations, 99, axis=0
        ).reshape((1, 2))
        # useful to visualize.
        self.stain_matrix_target_RGB = od2rgb(self.stain_matrix_target)

    def transform(self, img):
        """Transform an image.
        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                RGB input source image.
        Returns:
            :class:`numpy.ndarray`:
                RGB stain normalized image.
        """
        stain_matrix_source = self.extractor.get_stain_matrix(img)
        source_concentrations = self.get_concentrations(img, stain_matrix_source)
        max_c_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= self.maxC_target / max_c_source
        trans = 255 * np.exp(
            -1 * np.dot(source_concentrations, self.stain_matrix_target)
        )

        # ensure between 0 and 255
        trans[trans > 255] = 255
        trans[trans < 0] = 0

        return trans.reshape(img.shape).astype(np.uint8)
