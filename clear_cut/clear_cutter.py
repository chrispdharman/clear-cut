import pydoc
import json
import os
import time
import numpy as np
from collections import defaultdict
from random import randint

from clear_cut.utils.edge_utility import ImageUtils


class ClearCut(ImageUtils):

    _tracer = None

    def __init__(self, debug=False, serverless=True, **kwargs):
        """
        If serverless, we must store results in S3 buckets
        """
        self.debug = debug
        self.serverless = serverless

        self._default_image_selection(**kwargs)
    
    @property
    def tracer(self, method='gradient'):
        if not self._tracer:
            Tracer = pydoc.locate(
                f'clear_cut.utils.tracers.{method}.{str.capitalize(method)}Tracer'
            )
            self._tracer = Tracer(
                results_path=self.results_path,
                debug=self.debug,
                serverless=self.serverless,
            )
        
        return self._tracer

    def run(self):
        # Determine segmentation edges of the image (default method = gradient)
        edgy_image = self.tracer.trace_objects_in_image(image=self.image)

        # Reduce noise (edge pixels that cannot possibly contain an edge)
        edgy_image = self.edge_killer(edgy_image, pixel_tolerance=self.pixel_tolerance)

        self.graph_tools.save_image(
            edgy_image,
            filepath=f'{self.tracer.results_path}/0007_noise_reduced_image.png',
        )

        # Mask over the original image
        wipe_mask = edgy_image < 0.01
        bold_mask = edgy_image > 0.01
        self.image[wipe_mask] = 255
        self.image[bold_mask] = 0
        self.graph_tools.save_image(
            self.image,
            filepath=f'{self.tracer.results_path}/0008_edge_masked_image.png',
        )

    def _default_image_selection(self, **kwargs):
        # Assign overwritable ClearCut parameters. Otherwise, use defaults
        self.image_filepath = kwargs.get('image_filepath')
        self.image_size_threshold = kwargs.get('image_size_threshold', 600)
        self.pixel_tolerance = kwargs.get('pixel_tolerance', 10)

        # Read in image
        image_filepath = self._determine_image_filepath()
        self.image = np.array(
            self.graph_tools.upright_image(image_filepath=image_filepath)
        )

        # Determine results path
        self.results_path = kwargs.get('results_path') or os.getcwd()
        if not self.results_path.endswith('/'):
            self.results_path = f'{self.results_path}/'

        filename, _ = self.image_filename.split('.')
        self.results_path = f'{self.results_path}results/{filename}'

        self._reduce_image_size()

    def _determine_image_filepath(self, ):
        if self.image_filename is not None:
            # Gives user full control over specifying the correct image file location
            return self.image_filename

        # Fallback to our default Bob Ross image
        self.image_filename = 'Bob.jpeg'
        images_path = f'/opt/python/' if self.serverless else f'{os.getcwd()}/venv/'

        return f'{images_path}lib/python3.7/site-packages/clear_cut/images/{self.image_filename}'

    def _reduce_image_size(self):
        # Build pooling dictionary
        pooling_history = defaultdict(lambda: defaultdict(tuple))
        pooling_history['iteration:0']['image_shape'] = self.image.shape

        # Check if the image is too small to be pooled, then pool the image
        while self.graph_tools.image_mean(self.image.shape) > self.image_size_threshold:
            image, kernel = self.graph_tools.reduce_image(image=self.image)
            
            # Update dictionary
            iter_no = 'iteration:{}'.format(len(pooling_history.keys()))
            pooling_history[iter_no] = {
                'image_shape': image.shape,
                'kernal_size': kernel,
            }
            
            # Must assign within the loop to dynamicaly update the while condition
            self.image = image
        
        # note that the final k is stored in "k"
        if self.debug:
            print('pooling_history={}'.format(
                json.dumps(pooling_history, indent=4)
            ))

            self.graph_tools.save_image(
                self.image,
                filepath=f'{self.tracer.results_path}/0001_size_reduced_image.png',
            )

            self.graph_tools.save_image(
                self.image,
                filepath=f'{self.tracer.results_path}/0002_size_reduced_image_channel_collage.png',
                split_rgb_channels=True,
            )
