import pydoc
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image, ExifTags
from random import randint
from skimage.measure import block_reduce
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

from utils.edge_utility import ImageUtils


class ClearCut(ImageUtils):

    def __init__(self):
        self.base_dir = "app/images"
        self.default_image_selection()

        self._tracer = None
    
    @property
    def tracer(self, method="gradient"):
        if not self._tracer:
            Tracer = pydoc.locate('utils.tracers.{}.{}Tracer'.format(
                method, str.capitalize(method)
            ))
            self._tracer = Tracer()
        return self._tracer

    def default_image_selection(self):
        self.image_filename = "Bob.jpeg"
        self.image_filename = "colorful1.jpeg"
        #self.image_filename = "john1.jpg"
        #self.image_filename = "minimal1.jpg"
        #self.image_filename = "heathers_cats.jpg"
        self.image_filepath = '/'.join([self.base_dir, self.image_filename])
        self.image_raw = self.__upright_image()
        self.image = np.array(self.image_raw)
        print("Image size: ", self.image.shape)

        self.results_filepath = '/'.join(
            ["results", self.__get_file_name(self.image_filename)]
        )
        self.__reduce_image_size()

    def run(self):
        # Determine segmentation edges of the image (default method = gradient)
        edgy_images = self.tracer.trace_objects_in_image(image=self.image)

        # Remove noise (edge pixels that cannot possibly contain an edge)
        # TODO: may need to change order with edgeFiller?
        edgy_images = self.edge_killer(edgy_images, pixel_tolerance=4)

        # Use direction bias to fill in between edge pixels (to make image clearer)
        edgy_images = self.edge_filler(edgy_images, edge_bias=10)
        #plt.figure()
        #plt.imshow(edgy_images > 0.)
        #plt.show()

        # mask original image with edge array (original edges are red, filled edge are blue)
        #image[:, :, 2][new_edgy_images > 0.3] = 255
        #image[:, :, 0][edgy_images > 0.3] = 255
        #image[:, :, 2][edgy_images > 0.3] = 0
        '''plt.figure()
        plt.imshow(image)
        plt.show()'''

        # determine closed edge paths
        objNo = 0
        while objNo < 10:
            # run one iteration of random path edge race
            objBool, objEdgeArray = randomPathEdgeRace(edgy_images)

            if objBool:
                print("An object was found :)")
                objNo = objNo + 1

                # show path drawn
                edge_show(edgy_images, objEdgeArray)
            #else:
            #    print("No object was found :(")

        exit()

        # using pooling history, reconstruct edgy array to the same size as the original image
        pdict['im_edge_array' + str(k)] = edgy_images
        while k > 0:
            # get image and kern size values
            im_h = pdict['im_h'+str(k-1)]
            im_w = pdict['im_w' + str(k-1)]
            krn_h = pdict['im_kern_h' + str(k)]
            krn_w = pdict['im_kern_w' + str(k)]
            edg_size = pdict['im_edge_array' + str(k)].shape
            print("edg_size=",edg_size)

            # generate scaled up edge array
            new_edge_array = pdict['im_edge_array' + str(k)].resize((im_h, im_w))
            '''new_edge_array = np.zeros((im_h, im_w))
            for i in range(0, edg_size[0]):
                for j in range(0, edg_size[1]):
                    #print("i=", i, ", j=", j, "  --- ", int(krn_h * i), " & ", int(krn_w * j))
                    if pdict['im_edge_array' + str(k)][i,j] > 0.:
                        new_edge_array[int(krn_h * i), int(krn_w * j)] = 1.

            # update k before loop ends
            k = k - 1
            pdict['im_edge_array' + str(k)] = new_edge_array'''

        print(pdict['im_edge_array3'][4:11,0])
        print(pdict['im_edge_array2'][8:22, 0])
        print(pdict['im_edge_array1'][16:44, 0])
        print(pdict['im_edge_array0'][32:88, 0])
        '''plt.figure()
        plt.imshow(pdict['im_edge_array0'])

        # mask original image with edge array
        imageRaw[:,:,0][pdict['im_edge_array0'] > 0.3] = 255
        plt.figure()
        plt.imshow(imageRaw)
        plt.show()'''

        # view a random image of the data
        plt.clf()
        plt.figure()
        plt.imshow(train_data[randint(0, train_data.shape[0])].reshape(28, 28))
        plt.show()

    def __get_file_name(self, filename):
        '''
        Get file name, removing file extension
        '''
        name, _ = filename.split('.')
        return name

    def __get_trained_model(self):
        '''
        Load mnist data
        '''
        # mnist = tf.contrib.learn.datasets.load_dataset("mnist", one_hot=True)
        #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        #train_data = mnist.train.images
        #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        #test_data = mnist.test.images
        #test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    def __reduce_image_size(self):

        # create dictionary to store the history of pooled images
        pdict = {
            'image': {
                'height': [self.image.shape[0]],
                'width': [self.image.shape[1]],
            },
            'kernel': {
                'height': ['N/A'],
                'width': ['N/A'],
            },
        }

        # check if the image is too small to be pooled, then pool the image
        #while img_mean(image.shape) > 500:
        k = 0
        while self.img_mean(self.image.shape) > 300:
            # calculate the smallest kernel size that fits into the image
            krn_h, krn_w, image = self.calculate_kernel_size(self.image)
            #print("krn_h=", krn_h, ", krn_w=", krn_w)

            # reduce image size, given calculated kernel (max pooling)
            self.image = block_reduce(image, (krn_h, krn_w, 1), np.max)
            print("New shape=",self.image.shape)

            # update dictionary
            pdict['im_h' + str(k)], pdict['im_w' + str(k)], _ = self.image.shape
            pdict['im_kern_h' + str(k)], pdict['im_kern_w' + str(k)] = krn_h, krn_w
            
            k += 1
        # note that the final k is stored in "k"

        # View raw image
        '''plt.figure()
        plt.imshow(image)

        # View rgb channels
        plt.figure()
        plt.imshow(np.rot90(
            np.concatenate((np.concatenate((rotIm(image[:, :, 0]), rotIm(image[:, :, 1]))), rotIm(image[:, :, 2])))))'''

        # view a specific image of the data
        #plt.figure()
        #plt.imshow(train_data[3].reshape(28, 28))
        #plt.show()
        #plt.clf()


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
                image = np.rot90(np.rot90(image))
        return image


clear_cut = ClearCut()
clear_cut.run()