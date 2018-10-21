### IMPORT STANDARD PYTHON LIBRARIES/FUNCTIONS
#import tensorflow as tf
import numpy as np
from random import randint
import matplotlib
from PIL import Image, ExifTags
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from tensorflow.examples.tutorials.mnist import input_data

### IMPORT CUSTOM LIBRARIES/FUNCTIONS
from edgeUtility import *

# main routine
def main():
    # load data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist", one_hot=True)
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #train_data = mnist.train.images
    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    #test_data = mnist.test.images
    #test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # import single image
    #imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/Bob.jpeg"
    #imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/colorful1.jpeg"
    #imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/john1.jpg"
    #imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/minimal1.jpg"
    imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/heathers_cats.jpg"
    imageRaw = Image.open(imagePath)
    image = np.array(imageRaw)
    print("Image size: ", image.shape)

    # check whether or not the raw image contains exif data
    try:
        # check and rotate image to correct orientation
        # taken from https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
        exif = dict((ExifTags.TAGS[k], v) for k, v in imageRaw._getexif().items() if k in ExifTags.TAGS)
        if exif['Orientation']==3:
            image=np.rot90(np.rot90(image))
    except AttributeError as err:
        print("AttributeError: ", err)
    imageRaw = image

    # create dictionary to store the history of pooled images
    pdict = {}
    k = 0
    pdict['im_h'+str(k)], pdict['im_w'+str(k)], _ = image.shape
    pdict['im_kern_h' + str(k)], pdict['im_kern_w' + str(k)] = ['N/A', 'N/A']
    # check if the image is too small to be pooled, then pool the image
    while img_mean(image.shape) > 500:
        k = k + 1
        # calculate the smallest kernel size that fits into the image
        krn_h, krn_w, image = calcKernelSize(image)
        #print("krn_h=", krn_h, ", krn_w=", krn_w)

        # reduce image size, given calculated kernel (max pooling)
        image = block_reduce(image, (krn_h, krn_w, 1), np.max)
        #print("New shape=",image.shape)

        # update dictionary
        pdict['im_h' + str(k)], pdict['im_w' + str(k)], _ = image.shape
        pdict['im_kern_h' + str(k)], pdict['im_kern_w' + str(k)] = krn_h, krn_w
    # note that the final k is stored in "k"

    # print dictionary
    for key, value in pdict.items():
        #print(key,": \t",value)
        continue

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


    # MAKE AS FUNCTION PASSING IN AND RETURNING pdict
    # execute clearCut method and store in edge array for masking
    #edgy_images = traceObjectsInImage(image, method = "texture") # later think about implementing different methods as an argument
    edgy_images = traceObjectsInImage(image, method= "gradient")

    # remove edge pixels that cannot possibly contain an edge (may need to change order with edgeFiller?)
    edgy_images = edgeKiller(edgy_images, objectTolerance = 4)

    # use direction bias to fill in between edge pixels (possible edges)
    edgy_images = edgeFiller(edgy_images, edge_bias = 10)
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

main()
exit()