import os
import numpy as np

from skimage.measure import block_reduce


class BaseTracer(object):

    def __init__(self, method='Gradient', results_path=None):
        self.method = method
        self._get_or_create_results_dir(results_path, method)

    def merge_channels_of_traced_image(self, grdImg, origShape):
        """
        Merge gradImage RGB channels to one image
        """
        # Make image of correct shape
        xDim, yDim, chnls = origShape

        # Create empty array on the size of a single channel gradImage
        mrgdImg = np.zeros(shape=(2 * (xDim - 1), 2 * (yDim - 1)))

        # loop over each dimension, populating the gradient image
        x_offset = 2 * (yDim - 1)
        for i in range(0, 2 * (xDim - 1)):
            for j in range(0, 2 * (yDim - 1)):
                mrgdImg[i,j] = (
                    grdImg[i, j]
                    + grdImg[i, j + x_offset]
                    + grdImg[i, j + 2 * x_offset]
                )/3

        # Reduce gradient array to original image shape. Max pool gradient array using 2x2 kernel
        return block_reduce(mrgdImg, (2, 2), np.max)
        
    def _get_or_create_results_dir(self, results_path, method):
        results_path = results_path or 'app/results/misc'

        # Create results directory if it doesn't yet exist
        if "." in results_path:
            results_path, _ = results_path.split(".")

        self.results_path = "/".join([results_path, method])
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)
