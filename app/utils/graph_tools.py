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

        # determine lowest denominator in image height
        k_h = 2
        while image_h % k_h:
            # Notice: it starts from 3
            k_h += 1

            if k_h > image_h/2:
                print("Error: the image height is a prime number. Cannot determine pooling kernel size.")

                # function to remove one pixel layer off the image "height"
                image = self.crop_image(image, edge='height')
                k_h = 2
                break

        # determine lowest denominator in image width
        k_w = 2
        while image_w % k_w:
            # Notice: it starts from 3
            k_w += 1

            if k_w > image_w/2:
                print("Error: the image width is a prime number. Cannot determine pooling kernel size.")

                # function to remove one pixel layer off the image "width"
                image = self.crop_image(image, edge='width')
                k_w = 2
                break

        return k_h, k_w, image
    
    def crop_image(self, image, edge='both'):
        # Cut off a single pixel layer from the image
        if edge == 'height':
            return image[:image.shape[0]-1, :, :]
        elif edge == 'width':
            return image[:, :image.shape[1]-1, :]

        return image[:image.shape[0]-1, :image.shape[1]-1, :]

    def image_mean(self, image_shape):
        # Determine mean image size
        return (image_shape[0] + image_shape[1]) / 2
    
    def reduce_image(self, image=None):
        # Calculate the smallest kernel size that fits into the image
        krn_h, krn_w, image = self.calculate_kernel_size(image)                

        # Reduce image size, given calculated kernel (max pooling)
        image = block_reduce(image, (krn_h, krn_w, 1), np.max)

        return krn_h, krn_w, image

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
