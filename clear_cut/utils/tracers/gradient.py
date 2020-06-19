import os
import numpy as np
from itertools import product

from clear_cut.utils.tracers.base import BaseTracer


class GradientTracer(BaseTracer):
    
    # model_no is just a unique timestamp, i.e. model_no = time.time()
    def trace_objects_in_image(self, image=None, results_path=None, model_no=None):
        '''
        Object tracing one-layer gradient method
        GradImage: create numpy 2D array of size (2n-1) of the original
        '''
        dimY, dimX, channels = image.shape

        # Append an image (in x-direction) for each of the separate channels
        grad_image = np.zeros(
            shape=(
                2 * (dimX - 1) * channels,
                2 * (dimY - 1)
            )
        )

        # Define iteratables
        c_range = range(0, channels)
        x_range = range(0, 2 * (dimX - 1))
        y_range = range(0, 2 * (dimY - 1))

        for k in c_range:
            x_offset = 2 * k * (dimX - 1)

            for i, j in product(x_range, y_range):
                self.calculate_gradient_images_coordinates(
                    image,
                    grad_image,
                    coordinate=(i, j, k),
                    x_offset=x_offset,
                )

        edge_array = self.draw_edge_image(grad_image, image_shape=image.shape)

        # return an array of 0s (non-edges) and 1s (edges), same shape as passed in image
        self._print_if_debugging(f'Is {image.shape} == {edge_array.shape}?')
        return edge_array

    def calculate_gradient_images_coordinates(self, image, grad_image, coordinate=None, x_offset=None):
        """
        :params image: original image (numpy array of size M x N)
        :params image: gradient image (numpy array of size (2M-1) x (2N-1))
        :params coordinates: specific pixel of the original image
        :params x_offset: deals with the initial point of each r, g, or b image in the "grid"
        """
        i, j, k = coordinate
        if i % 2:
            # across odd numbered rows and ...
            # ... adjacent pixels (top to bottom gradient)
            # ... diagonal pixels (top-left to bottom-right gradient)
            grad_image[i + x_offset, j] = (
                image[int(j / 2) + (j % 2), int((i + 1) / 2)]
                - image[int(j / 2), int((i - 1) / 2)]
            )[k]
        else:
            # across even numbered rows and ...
            # ... adjacent pixels (left to right gradient)
            # ... diagonal pixels (top-right to bottom-left gradient)
            grad_image[i + x_offset, j] = (
                image[int(j / 2) + 1, int(i / 2)]
                - image[int(j / 2), int((i / 2) + (j % 2))]
            )[k]

    def draw_edge_image(self, grad_image, image_shape=None, image_cut=0.08):
        # Too small (shapes distinct but too much noise): 0.02
        # Maybe right? 0.07 (Bob.jpeg)
        # Too large (shaped not distinct enough): 0.10
        
        edge_array = self.tidy_edge_image_edges(
            self.merge_channels_of_traced_image(
                self.trimmed_image(grad_image, image_cut=image_cut),
                image_shape
            ),
            image_shape=image_shape
        )

        if self.serverless:
            # Will need to set up to hit S3 here
            return edge_array
        
        ## Otherwise save images in local results directory

        # Display separate rgb gradient images without cutoff applied
        Image.fromarray(
            np.absolute(grad_image.T)
        ).save(
            '{}/0003_gradient_image_raw.png'.format(self.results_path)
        )

        # Display separate rgb gradient images with cutoff applied
        Image.fromarray(
            np.multiply((np.absolute(grad_image.T) < (1-image_cut)*255),(np.absolute(grad_image.T) > image_cut*255))
        ).save(
            '{}/0004_gradient_image_cut.png'.format(self.results_path)
        )

        # Display merged rgb gradient image without cutoff applied
        Image.fromarray(
            self.merge_channels_of_traced_image(grad_image.T, image_shape)
        ).save(
            '{}/0005_merged_image_raw.png'.format(self.results_path)
        )
        
        # Display merged rgb gradient image with cutoff applied
        Image.fromarray(
            edge_array
        ).save(
            '{}/0006_merged_image_cut.png'.format(self.results_path)
        )

        return edge_array

    def tidy_edge_image_edges(self, edge_array, image_shape=None):
        # Append 0s (non-edge pixels) to any missing columns/rows.
        # This is akin to filling the colour of the edges with the same colour as their adjacent pixels
        x_miss = image_shape[0] - edge_array.shape[0]
        if x_miss == 0:
            self._print_if_debugging('Same number of rows. Good!')
        elif x_miss > 0:
            self._print_if_debugging('Lost rows in compressing gradient. It can happen! Attempting to automatically dealing with it.')
            edge_array = np.concatenate((edge_array, np.zeros((1, edge_array.shape[1]))), axis = 0)
        else:
            raise Exception('Gained rows in compressing gradient. Doesn\'t make sense!')

        y_miss = image_shape[1] - edge_array.shape[1]
        if y_miss == 0:
            self._print_if_debugging('Same number of columns. Good!')
        elif y_miss > 0:
            self._print_if_debugging('Lost columns in compressing gradient. It can happen! Attempting to automatically dealing with it.')
            edge_array = np.concatenate((edge_array, np.zeros((edge_array.shape[0], 1))), axis=1)
        else:
            raise Exception('Gained columns in compressing gradient. Doesn\'t make sense!')

        return edge_array
    
    def trimmed_image(self, grad_image, image_cut=None):
        """
        Removes all edge pixels that have "too flat" a gradient
        """
        return np.multiply(
            (np.absolute(grad_image.T) < (1 - image_cut) * 255),
            (np.absolute(grad_image.T) > image_cut * 255)
        )
    
    def _print_if_debugging(self, message):
        if self.debug:
            print(message)
