import os
import numpy as np

import matplotlib.pyplot as plt

from utils.tracers.base import BaseTracer


class GradientTracer(BaseTracer):
    
    # model_no is just a unique timestamp, i.e. model_no = time.time()
    def trace_objects_in_image(self, image=None, results_path=None, model_no=None):
        '''
        Object tracing one-layer gradient method
        '''
        # gradImage: create numpy 2D array of size (2n-1) of the original
        dimY, dimX, chanls = image.shape

        # append an image (in x-direction) for each of the separate channels
        grad_image = np.zeros(shape=(2*chanls*(dimX-1),2*(dimY-1)))

        # loop over each dimension, populating the gradient image
        for k in range(0, chanls):
            # This offset deals with the initial point of each r, g, or b image in the "grid"
            x_offset = 2 * k * (dimX-1)

            for i in range(0, 2 * (dimX - 1)):
                for j in range(0, 2 * (dimY - 1)):
                    #print("i=",i,", j=",j)
                    if i % 2 == 1:
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

        edge_array = self.draw_edge_image(grad_image, image_shape=image.shape, visualise=True)

        # return an array of 0s (non-edges) and 1s (edges), same shape as passed in image
        print("Is ", image.shape," = ", edge_array.shape, "?")
        return edge_array

    def draw_edge_image(self, grad_image, image_shape=None, image_cut=0.08, visualise=False):
        # Too small (shapes distinct but too much noise): 0.02
        # Maybe right? 0.07 (Bob.jpeg)
        # Too large (shaped not distinct enough): 0.10
        
        edge_array = self.tidy_edge_image_edges(
            self.merge_channels_of_traced_image(
                np.multiply(
                    (np.absolute(grad_image.T) < (1 - image_cut) * 255),
                    (np.absolute(grad_image.T) > image_cut * 255)),
                image_shape
            ),
            image_shape=image_shape
        )

        if visualise:
            # Display separate rgb gradient images without cutoff applied
            plt.figure()
            plt.imshow(np.absolute(grad_image.T), interpolation="nearest")
            plt.savefig('{}/gradient_image_raw.png'.format(self.results_path))

            # Display separate rgb gradient images with cutoff applied
            plt.figure()
            plt.imshow(np.multiply((np.absolute(grad_image.T) < (1-image_cut)*255),(np.absolute(grad_image.T) > image_cut*255)))
            plt.savefig('{}/gradient_image_cut.png'.format(self.results_path))

            # Display merged rgb gradient image without cutoff applied
            mrgIm1 = self.merge_channels_of_traced_image(grad_image.T, image_shape)
            plt.figure()
            plt.imshow(mrgIm1)
            plt.savefig('{}/merged_image_raw.png'.format(self.results_path))
            
            # Display merged rgb gradient image with cutoff applied
            plt.figure()
            plt.imshow(edge_array)
            plt.savefig('{}/merged_image_cut.png'.format(self.results_path))

        return edge_array

    def tidy_edge_image_edges(self, edge_array, image_shape=None):
        # Append 0s (non-edge pixels) to any missing columns/rows.
        # This is akin to filling the colour of the edges with the same colour as their adjacent pixels
        x_miss = image_shape[0] - edge_array.shape[0]
        if x_miss == 0:
            print("Same number of rows. Good!")
        elif x_miss > 0:
            print("Lost rows in compressing gradient. It can happen! Attempting to automatically dealing with it.")
            edge_array = np.concatenate((edge_array, np.zeros((1, edge_array.shape[1]))), axis = 0)
        else:
            raise Exception("Gained rows in compressing gradient. Doesn't make sense!")

        y_miss = image_shape[1] - edge_array.shape[1]
        if y_miss == 0:
            print("Same number of columns. Good!")
        elif y_miss > 0:
            print("Lost columns in compressing gradient. It can happen! Attempting to automatically dealing with it.")
            edge_array = np.concatenate((edge_array, np.zeros((edge_array.shape[0], 1))), axis=1)
        else:
            raise Exception("Gained columns in compressing gradient. Doesn't make sense!")

        return edge_array
