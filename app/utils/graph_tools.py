import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image, ExifTags
from skimage.measure import block_reduce


class GraphTools(object):

    def calculate_kernel_size(self, image):
        # Determine kernel size from image
        image_h, image_w, *_ = image.shape

        kernel_height, image = self.find_lowest_denominator(image, image_length=image_h, edge='height')
        kernel_width, image = self.find_lowest_denominator(image, image_length=image_w, edge='width')

        kernel_size = (kernel_height, kernel_width)
        return image, kernel_size
    
    def crop_image(self, image, edge='both'):
        # Cut off a single pixel layer from the image
        if edge == 'height':
            return image[:image.shape[0]-1, :, :]
        elif edge == 'width':
            return image[:, :image.shape[1]-1, :]

        # Cut both edges
        return image[:image.shape[0]-1, :image.shape[1]-1, :]
    
    def edge_pixel_positions(self, edge_image):
        """
        Determine coordinates of edge pixels.
        :return: (N x 2) numpy array, where N is the number of edge pixels.
        """
        return np.argwhere(edge_image > 0.)
    
    def find_lowest_denominator(self, image, image_length=None, edge='height'):
        for factor in range(2, image_length // 2):
            if not image_length % factor:
                # Found the smallest denominator (stored in k_h)
                return factor, image
        
        # Remove one pixel layer off the image edge
        image = self.crop_image(image, edge=edge)
        return 2, image

    def image_mean(self, image_shape):
        # Determine mean image size
        return (image_shape[0] + image_shape[1]) / 2
    
    def reduce_image(self, image=None):
        # Calculate the smallest kernel size that fits into the image
        image, kernel = self.calculate_kernel_size(image)                

        # Reduce image size, given calculated kernel (max pooling)
        image = block_reduce(image, (kernel[0], kernel[1], 1), np.max)

        return image, kernel

    def save_image(self, image, filepath=None, split_rgb_channels=False):
        if split_rgb_channels:
            image = self.split_rgb_channels(image)

        plt.figure()
        plt.imshow(image)
        plt.savefig(filepath)

    @staticmethod
    def split_rgb_channels(image):
        return np.rot90(
            np.concatenate(
                (
                    np.concatenate(
                        (
                            np.rot90(image[:, :, 0], -1),
                            np.rot90(image[:, :, 1], -1),
                        )
                    ),
                    np.rot90(image[:, :, 2], -1),
                )
            )
        )
    
    @staticmethod
    def upright_image(image_filepath=None):
        '''
        Check for image orientation in exif data. See reference
        https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
        '''
        image = Image.open(image_filepath)
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
