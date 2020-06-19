import pydoc
import json
import time
import numpy as np
from collections import defaultdict
from random import randint

from clear_cut.utils.edge_utility import ImageUtils


class ClearCut(ImageUtils):

    _tracer = None

    def __init__(self, base_path='', debug=False, serverless=True):
        """
        If serverless, we must store results in S3 buckets
        """
        self.base_path = '/opt/python/lib/python3.7/site-packages/' if serverless else base_path
        self.debug = debug
        self.serverless = serverless
        
        self.base_dir = f'{self.base_path}clear_cut/images'
        self.default_image_selection()
    
    @property
    def tracer(self, method='gradient'):
        if not self._tracer:
            Tracer = pydoc.locate(
                f'clear_cut.utils.tracers.{method}.{str.capitalize(method)}Tracer'
            )
            self._tracer = Tracer(
                results_path=self.results_filepath,
                debug=self.debug,
                serverless=self.serverless,
            )
        
        return self._tracer

    def default_image_selection(self):
        self.image_filename = 'Bob.jpeg'

        self.image_filepath = '/'.join([self.base_dir, self.image_filename])
        self.image_size_threshold = 600
        self.pixel_tolerance = 10

        self.image_raw = self.graph_tools.upright_image(image_filepath=self.image_filepath)
        self.image = np.array(self.image_raw)

        filename, _ = self.image_filename.split('.')
        self.results_filepath = f'{self.base_path}results/{filename}'
        self.reduce_image_size()

    def run(self):
        # Determine segmentation edges of the image (default method = gradient)
        edgy_image = self.tracer.trace_objects_in_image(image=self.image)

        # Reduce noise (edge pixels that cannot possibly contain an edge)
        edgy_image = self.edge_killer(edgy_image, pixel_tolerance=self.pixel_tolerance)

        self.graph_tools.save_image(
            edgy_image,
            filepath='{}/0007_noise_reduced_image.png'.format(self.tracer.results_path),
        )

        # Mask over the original image
        wipe_mask = edgy_image < 0.01
        bold_mask = edgy_image > 0.01
        self.image[wipe_mask] = 255
        self.image[bold_mask] = 0
        self.graph_tools.save_image(
            self.image,
            filepath='{}/0008_edge_masked_image.png'.format(self.tracer.results_path),
        )

    def reduce_image_size(self):
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
                filepath='{}/0001_size_reduced_image.png'.format(self.tracer.results_path),
            )

            self.graph_tools.save_image(
                self.image,
                filepath='{}/0002_size_reduced_image_channel_collage.png'.format(self.tracer.results_path),
                split_rgb_channels=True,
            )
