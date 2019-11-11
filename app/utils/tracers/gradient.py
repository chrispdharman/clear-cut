import os
import numpy as np

import matplotlib.pyplot as plt

from base import BaseTracer 


class GradientTracer(BaseTracer):

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
            x_offset = 2 * k * (dimX-1)
            for i in range(0, 2 * (dimX - 1)):
                for j in range(0, 2 * (dimY - 1)):
                    #print("i=",i,", j=",j)
                    if i % 2 == 1:
                        # across odd numbered rows
                        # grad_image[i, j] = origImage[int(i / 2), int(j / 2)] - origImage[int(i - 1 / 2), int(j / 2)]
                        if j % 2 == 0:
                            # across adjacent pixels (top to bottom gradient)
                            # grad_image[i, j] = 0.9
                            grad_image[i + x_offset, j] = (image[int(j / 2), int((i + 1) / 2)] - image[
                                int(j / 2), int((i - 1) / 2)])[k]
                            # print("(j,i)=("+str(j)+","+str(i)+")"+"\t grad: ("+str(int(j/2))+","+str(int((i+1)/2))+")-("+str(int(j/2))+","+str(int((i-1)/2))+")")
                        else:
                            # across diagonal pixels (top-left to bottom-right gradient)
                            # grad_image[i, j] = 1.0
                            grad_image[i+x_offset, j] = (image[int(j / 2) + 1, int((i + 1) / 2)] - image[
                                int(j / 2), int((i - 1) / 2)])[k]
                            # print("(j,i)=(" + str(j) + "," + str(i) + ")" + "\t grad: (" + str(int(j / 2)+1) + "," + str(int((i + 1) / 2)) + ")-(" + str(int(j / 2)) + "," + str(int((i - 1) / 2)) + ")")
                    else:
                        # across even numbered rows
                        if j % 2 == 0:
                            # across adjacent pixels (left to right gradient)
                            # grad_image[i, j] = 0.1
                            grad_image[i+x_offset, j] = (image[int(j / 2) + 1, int(i / 2)] - image[int(j / 2), int(i / 2)])[k]
                            # print("(j,i)=(" + str(j) + "," + str(i) + ")" + "\t grad: (" + str(int(j / 2) + 1) + "," + str(int(i / 2)) + ")-(" + str(int(j / 2)) + "," + str(int(i / 2)) + ")")
                        else:
                            # across diagonal pixels (top-right to bottom-left gradient)
                            # grad_image[i, j] = 0.0
                            grad_image[i+x_offset, j] = (image[int(j / 2) + 1, int(i / 2)] - image[
                                int(j / 2), int((i / 2) + 1)])[k]
                            # print("(j,i)=(" + str(j) + "," + str(i) + ")" + "\t grad: (" + str(int(j / 2) + 1) + "," + str(int(i / 2)) + ")-(" + str(int(j / 2)) + "," + str(int((i / 2)+1)) + ")")

        # Too small (shapes distinct but too much noise): 0.02
        # Maybe right? 0.07 (Bob.jpeg)
        # Too large (shaped not distinct enough): 0.10
        image_cut = 0.08
        #imCut = 0.06
        # display gradient image
        plt.figure()
        plt.imshow(np.absolute(grad_image.T), interpolation="nearest")
        plt.figure()
        plt.imshow(np.multiply((np.absolute(grad_image.T) < (1-image_cut)*255),(np.absolute(grad_image.T) > image_cut*255)))

        # merge channels
        mrgIm1 = self.merge_channels_of_traced_image(grad_image.T, image.shape)
        plt.figure()
        plt.imshow(mrgIm1)

        edge_array = self.merge_channels_of_traced_image(
            np.multiply((np.absolute(grad_image.T) < (1 - image_cut) * 255), (np.absolute(grad_image.T) > image_cut * 255)),
            image.shape)
        plt.figure()
        plt.imshow(edge_array)

        # append 0s (non-edge pixels) to any missing columns/rows
        x_miss = image.shape[0] - edge_array.shape[0]
        if x_miss == 0:
            print("Same number of rows. Good!")
        elif x_miss > 0:
            print("Lost rows in compressing gradient. It can happen! Attempting to automatically dealing with it.")
            edge_array = np.concatenate((edge_array, np.zeros((1, edge_array.shape[1]))), axis = 0)
        else:
            print("Gained rows in compressing gradient. Doesn't make sense!")
            exit()

        y_miss = image.shape[1] - edge_array.shape[1]
        if y_miss == 0:
            print("Same number of columns. Good!")
        elif y_miss > 0:
            print("Lost columns in compressing gradient. It can happen! Attempting to automatically dealing with it.")
            edge_array = np.concatenate((edge_array, np.zeros((edge_array.shape[0], 1))), axis=1)
        else:
            print("Gained columns in compressing gradient. Doesn't make sense!")
            exit()

        # return an array of 0s (non-edges) and 1s (edges), same shape as passed in image
        print("Is ",image.shape," = ",edge_array.shape,"?")
        return edge_array
