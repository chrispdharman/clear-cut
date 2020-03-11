import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image, ExifTags
from skimage.measure import block_reduce


class GraphTools(object):

    def calculate_kernel_size(self, img):
        # Determine kernel size from image
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

        return k_h, k_w, newImg
    
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
