import os
import sys
import csv
import time
import math
import numpy as np
from random import randint

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm


class ImageUtils(object):

    def __init__(self):
        # increase csv file size limit
        csv.field_size_limit(sys.maxsize)

    def reduce_iter(self, i):
        return i - (i > 5) * 8

    # def rotate image 90 deg CW shortcut
    def rot90_CW(self, image):
        return np.rot90(image, -1)

    # function to calculate the smallest kernel size for th given image
    def calculate_kernel_size(self, img):
        # determine image size
        #print("Image size: ", img.shape)
        img_h, img_w, *_ = img.shape
        newImg = img

        # determine lowest denominator in image height
        k_h = 2
        while( img_h % k_h != 0 ):
            k_h += 1
            if (k_h > img_h/2):
                print("Error: the image height is a prime number. Cannot determine pooling kernel size.")

                # function to remove one pixel layer off the image "height"
                newImg = self.image_crop(img, edge="h")
                k_h = 2
                break

        # determine lowest denominator in image width
        k_w = 2
        while ( (img_w % k_w) != 0 ):
            k_w += 1
            if (k_w > img_w/2):
                print("Error: the image width is a prime number. Cannot determine pooling kernel size.")

                # function to remove one pixel layer off the image "width"
                newImg = self.image_crop(img, edge = "w")
                k_w = 2
                break

        #print("Newish shape=", newImg.shape)
        return k_h, k_w, newImg

    # determine average image size
    def image_mean(self, image_shape):
        return (image_shape[0] + image_shape[1]) / 2

    # cut off pixel pixel of the image and return it
    def image_crop(self, im, edge = "both"):
        if edge=="h":
            new_image = im[:im.shape[0]-1, :, :]
        elif edge=="w":
            new_image = im[:, :im.shape[1]-1, :]
        else:
            new_image = im[:im.shape[0]-1, :im.shape[1]-1, :]
        return new_image

    def edge_pixel_positions(self, edge_img):
        """
        Determine coordinates of edge pixels.
        :return: (N x 2) numpy array, where N is the number of edge pixels.
        """
        edge_coordinates = []
        shape = edge_img.shape
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                if edge_img[i,j] > 0.:
                    #print("edge_img[i,j]=",edge_img[i,j])
                    edge_coordinates.append([i,j])
        
        return np.array(edge_coordinates)

    def edge_kill(self, edg_img, coord, radius, wipe=False):
        """
        Determine whether the "shells border" are all non-edge pixels or not
        """
        # This may have already been wiped
        if not edg_img[coord[0], coord[1]]:
            if wipe:
                return edg_img

            return True

        # Initial counter and pre-define useful values
        border_size = 2 * radius + 1

        # Run over the square of pixels surrounding "radius" pixels around coord
        for i in range(0, border_size**2):
            dx = (i % border_size) - radius
            dy = (i // border_size) - radius

            x = coord[0] + dx
            y = coord[1] + dy

            if wipe:
                # Wipe whole shell of edge pixels
                try:
                    edg_img[x, y] = 0
                except IndexError:
                    # The central edge pixel is too close to the image perimeter, so ignore it
                    pass

                continue

            # Skip the inner shell pixels, just run around the border
            if abs(dx) != radius and abs(dy) != radius:
                continue
            
            try:
                if edg_img[x, y] > 0.:
                    # Found an edge pixel on the border, thus we cannot wipe this shell
                    return False
                
            except IndexError:
                # The central edge pixel is too close to the image perimeter, so ignore it
                continue

        if wipe:
            return edg_img

        # If we got here, all border pixels must be non-edge pixels
        return True

    def edge_killer(self, edge_image, pixel_tolerance=1):
        """
        Check and wipe out any edge pixels found within a pixel_tolerance radius.
        We refer to the radius of surround pixels as "the shell".
        """
        edge_coordinates = self.edge_pixel_positions(edge_image)

        for edge_coordinate in edge_coordinates:
            # Iterate over each layer of the shell (r --> r - 1 decrements)
            for sub_radius in reversed(range(1, pixel_tolerance + 1)):
                noisy_shell_found = self.edge_kill(edge_image, edge_coordinate, radius=sub_radius)
                if noisy_shell_found:
                    edge_image = self.edge_kill(edge_image, edge_coordinate, radius=sub_radius-1, wipe=True)
                    break

        return edge_image

    def __within_radius(self, x, y, chosen_one, R=10):
        return math.sqrt((x - chosen_one[0]) ** 2 + (y - chosen_one[1]) ** 2) <= R
