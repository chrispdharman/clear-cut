import os
import numpy as np
from skimage.measure import block_reduce

from clear_cut.utils.graph_tools import GraphTools


class BaseTracer(object):

    _graph_tools = None

    def __init__(self, method='Gradient', results_path=None, debug=False, serverless=True):
        self.debug = debug
        self.method = method
        self.serverless = serverless
        self.results_path = results_path

        if not self.serverless:
            self._get_or_create_results_dir()
    
    @property
    def graph_tools(self):
        if not self._graph_tools:
            if self.serverless:
                self._graph_tools = GraphTools(serverless=True, debug=self.debug)
                return self._graph_tools

            self._graph_tools = GraphTools(debug=self.debug)
        
        return self._graph_tools

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
                )

        # Reduce gradient array to original image shape. Max pool gradient array using 2x2 kernel
        return block_reduce(mrgdImg, (2, 2), np.max)
        
    def _get_or_create_results_dir(self):
        results_path = self.results_path

        # Create results directory if it doesn't yet exist
        if '.' in results_path:
            results_path, _ = results_path.split('.')

        if not os.path.isdir(results_path):
            os.makedirs(results_path)
