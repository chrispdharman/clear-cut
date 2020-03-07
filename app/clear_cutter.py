import pydoc
import json
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import defaultdict
from PIL import Image, ExifTags
from random import randint
from skimage.measure import block_reduce

from utils.edge_utility import ImageUtils

DEBUG = True


class ClearCut(ImageUtils):

    _tracer = None

    def __init__(self):
        self.base_dir = 'app/images'
        self.default_image_selection()
    
    @property
    def tracer(self, method='gradient'):
        if not self._tracer:
            Tracer = pydoc.locate('utils.tracers.{}.{}Tracer'.format(
                method, str.capitalize(method)
            ))
            self._tracer = Tracer()
        
        return self._tracer

    def default_image_selection(self):
        self.image_filename = 'Bob.jpeg'
        self.image_filename = 'colorful1.jpeg'
        self.image_filename = 'john1.jpg'
        #self.image_filename = 'minimal1.jpg'
        #self.image_filename = 'heathers_cats.jpg'
        self.image_filepath = '/'.join([self.base_dir, self.image_filename])
        self.image_size_threshold = 400
        self.pixel_tolerance = 4
        self.image_raw = self.__upright_image()
        self.image = np.array(self.image_raw)
        self.results_filepath = '/'.join(
            ['results', self.__get_file_name(self.image_filename)]
        )
        self.__reduce_image_size()

    def run(self):
        # Determine segmentation edges of the image (default method = gradient)
        edgy_images = self.tracer.trace_objects_in_image(image=self.image)

        # Reduce noise (edge pixels that cannot possibly contain an edge)
        edgy_images = self.edge_killer(edgy_images, pixel_tolerance=self.pixel_tolerance)

        # Display merged rgb gradient image with cutoff applied
        plt.figure()
        plt.imshow(edgy_images)
        plt.savefig('{}/noise_reduced_image.png'.format(self.tracer.results_path))

    def __get_file_name(self, filename):
        '''
        Get file name, removing file extension
        '''
        name, _ = filename.split('.')
        return name

    def __reduce_image_size(self):
        # Build pooling dictionary
        k = 0
        pooling_history = defaultdict(lambda: defaultdict(tuple))
        pooling_history[str(k)]['image_shape'] = self.image.shape

        # Check if the image is too small to be pooled, then pool the image
        while self.image_mean(self.image.shape) > self.image_size_threshold:
            k += 1

            # Calculate the smallest kernel size that fits into the image
            krn_h, krn_w, image = self.calculate_kernel_size(self.image)                

            # Reduce image size, given calculated kernel (max pooling)
            self.image = block_reduce(image, (krn_h, krn_w, 1), np.max)

            # Update dictionary
            pooling_history[str(k)]['image_shape'] = self.image.shape
            pooling_history[str(k)]['kernal_shape'] = krn_h, krn_w
        
        # note that the final k is stored in "k"
        if DEBUG:
            print('pooling_history={}'.format(
                json.dumps(pooling_history, indent=4)
            ))

            # View raw image
            plt.figure()
            plt.imshow(image)
            plt.savefig('{}/size_reduced_image.png'.format(self.tracer.results_path))

            # View rgb channels
            plt.figure()
            plt.imshow(
                np.rot90(
                    np.concatenate(
                        (
                            np.concatenate(
                                (
                                    self.rot90_CW(image[:, :, 0]),
                                    self.rot90_CW(image[:, :, 1]),
                                )
                            ),
                            self.rot90_CW(image[:, :, 2]),
                        )
                    )
                )
            )
            plt.savefig('{}/size_reduced_image_channel_collage.png'.format(self.tracer.results_path))

    def __upright_image(self):
        '''
        Check for image orientation in exif data. See reference
        https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
        '''
        image = Image.open(self.image_filepath)
        if image._getexif() is not None:
            exif = dict(
                (ExifTags.TAGS[k], v)
                for k, v in image._getexif().items()
                if k in ExifTags.TAGS
            )
            if exif['Orientation'] == 3:
                return np.rot90(np.rot90(image))

            if exif['Orientation'] == 6:
                return np.rot90(image)

            if exif['Orientation'] == 8:
                return np.rot90(image, k=-1)

        return image


clear_cut = ClearCut()
clear_cut.run()