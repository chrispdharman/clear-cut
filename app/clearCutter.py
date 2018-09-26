#import tensorflow as tf
import numpy as np
from skimage.measure import block_reduce
from random import randint
import matplotlib
from PIL import Image, ExifTags
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from tensorflow.examples.tutorials.mnist import input_data

# object tracing method
def traceObjectsInImage(origImage):
    # gradImage: create numpy 2D array of size (2n-1) of the original
    dimY, dimX, chanls = origImage.shape

    # append an image (in x-direction) for each of the separate channels
    gradImage = np.zeros(shape=(2*chanls*(dimX-1),2*(dimY-1)))

    # loop over each dimension, populating the gradient image
    for k in range(0,chanls):
        x_offset = k*2*(dimX-1)
        for i in range(0, 2 * (dimX - 1)):
            for j in range(0, 2 * (dimY - 1)):
                #print("i=",i,", j=",j)
                if i % 2 == 1:
                    # across odd numbered rows
                    # gradImage[i, j] = origImage[int(i / 2), int(j / 2)] - origImage[int(i - 1 / 2), int(j / 2)]
                    if j % 2 == 0:
                        # across adjacent pixels (top to bottom gradient)
                        # gradImage[i, j] = 0.9
                        gradImage[i+x_offset, j] = (origImage[int(j / 2), int((i + 1) / 2)] - origImage[
                            int(j / 2), int((i - 1) / 2)])[k]
                        # print("(j,i)=("+str(j)+","+str(i)+")"+"\t grad: ("+str(int(j/2))+","+str(int((i+1)/2))+")-("+str(int(j/2))+","+str(int((i-1)/2))+")")
                    else:
                        # across diagonal pixels (top-left to bottom-right gradient)
                        # gradImage[i, j] = 1.0
                        gradImage[i+x_offset, j] = (origImage[int(j / 2) + 1, int((i + 1) / 2)] - origImage[
                            int(j / 2), int((i - 1) / 2)])[k]
                        # print("(j,i)=(" + str(j) + "," + str(i) + ")" + "\t grad: (" + str(int(j / 2)+1) + "," + str(int((i + 1) / 2)) + ")-(" + str(int(j / 2)) + "," + str(int((i - 1) / 2)) + ")")
                else:
                    # across even numbered rows
                    if j % 2 == 0:
                        # across adjacent pixels (left to right gradient)
                        # gradImage[i, j] = 0.1
                        gradImage[i+x_offset, j] = (origImage[int(j / 2) + 1, int(i / 2)] - origImage[int(j / 2), int(i / 2)])[k]
                        # print("(j,i)=(" + str(j) + "," + str(i) + ")" + "\t grad: (" + str(int(j / 2) + 1) + "," + str(int(i / 2)) + ")-(" + str(int(j / 2)) + "," + str(int(i / 2)) + ")")
                    else:
                        # across diagonal pixels (top-right to bottom-left gradient)
                        # gradImage[i, j] = 0.0
                        gradImage[i+x_offset, j] = (origImage[int(j / 2) + 1, int(i / 2)] - origImage[
                            int(j / 2), int((i / 2) + 1)])[k]
                        # print("(j,i)=(" + str(j) + "," + str(i) + ")" + "\t grad: (" + str(int(j / 2) + 1) + "," + str(int(i / 2)) + ")-(" + str(int(j / 2)) + "," + str(int((i / 2)+1)) + ")")

    # Too small (shapes distinct but too much noise): 0.02
    # Maybe right? 0.07 (Bob.jpeg)
    # Too large (shaped not distinct enough): 0.10
    imCut = 0.08
    #imCut = 0.06
    # display gradient image
    plt.figure()
    plt.imshow(np.absolute(gradImage.T), interpolation="nearest")
    plt.figure()
    plt.imshow(np.multiply((np.absolute(gradImage.T) < (1-imCut)*255),(np.absolute(gradImage.T) > imCut*255)))

    # merge channels
    mrgIm1 = mergeChannelsTracedImage(gradImage.T, origImage.shape)
    plt.figure()
    plt.imshow(mrgIm1)

    mrgIm2 = mergeChannelsTracedImage(
        np.multiply((np.absolute(gradImage.T) < (1 - imCut) * 255), (np.absolute(gradImage.T) > imCut * 255)),
        origImage.shape)
    plt.figure()
    plt.imshow(mrgIm2)

    # reduce gradient array to original image shape. Max pool gradient array using 2x2 kernel
    edge_array = block_reduce(mrgIm2, (2, 2), np.max)

    # append 0s (non-edge pixels) to any missing columns/rows
    x_miss = origImage.shape[0] - edge_array.shape[0]
    if x_miss == 0:
        print("Same number of rows. Good!")
    elif x_miss > 0:
        print("Lost rows in compressing gradient. It can happen! Attempting to automatically dealing with it.")
        edge_array = np.concatenate((edge_array, np.zeros((1, edge_array.shape[1]))), axis = 0)
    else:
        print("Gained rows in compressing gradient. Doesn't make sense!")
        exit()

    y_miss = origImage.shape[1] - edge_array.shape[1]
    if y_miss == 0:
        print("Same number of columns. Good!")
    elif y_miss > 0:
        print("Lost columns in compressing gradient. It can happen! Attempting to automatically dealing with it.")
        edge_array = np.concatenate((edge_array, np.zeros((edge_array.shape[0], 1))), axis=1)
    else:
        print("Gained columns in compressing gradient. Doesn't make sense!")
        exit()

    # return an array of 0s (non-edges) and 1s (edges), same shape as passed in image
    print("Is ",origImage.shape," = ",edge_array.shape,"?")
    return edge_array

# Merge gradImage RGB channels to one image
def mergeChannelsTracedImage(grdImg, origShape):
    # make image of correct shape
    xDim, yDim, chnls = origShape
    #print("xDim= ",xDim,", yDim=",yDim)
    #print("grdImg.shape=",grdImg.shape)

    # create empty array on the size of a single channel gradImage
    mrgdImg = np.zeros(shape=(2 * (xDim - 1), 2 * (yDim - 1)))
    #print("mrgdImg.shape=",mrgdImg.shape)

    # loop over each dimension, populating the gradient image
    x_offset = 2*(yDim-1)
    #x_offset = 2 * (yDim - 1)
    for i in range(0, 2 * (xDim - 1)):
        #print("i=", i)
        for j in range(0, 2 * (yDim - 1)):
            #print("i=", i, ", j=", j)
            mrgdImg[i,j] = (grdImg[i,j] + grdImg[i,j+ x_offset] + grdImg[i,j + 2*x_offset])/3

    return mrgdImg

# def rotate image 90 deg CW shortcut
def rotIm(img):
    return np.rot90(img, 1, (1,0))

# function to calculate the smallest kernel size for th given image
def calcKernelSize(img):
    # determine image size
    print("Image size: ", img.shape)
    img_h = img.shape[0]
    img_w = img.shape[1]
    newImg = img

    # determine lowest denominator in image height
    k_h = 2
    while( (img_h % k_h == 0) is False ):
        k_h = k_h + 1
        if (k_h > img_h/2):
            print("Error: the image height is a prime number. Cannot determine pooling kernel size.")

            # function to remove one pixel layer off the image "height"
            newImg = img_crop(img, edge="h")
            k_h = 2
            break

    # determine lowest denominator in image height
    k_w = 2
    while ( (img_w % k_w) == 0 is False ):
        k_w = k_w + 1
        if (k_w > img_w/2):
            print("Error: the image width is a prime number. Cannot determine pooling kernel size.")

            # function to remove one pixel layer off the image "width"
            newImg = img_crop(img, edge = "w")
            k_w = 2
            break

    print("Newish shape=", newImg.shape)
    return k_h, k_w, newImg

# determine average image size
def img_mean(imgshp):
    return (imgshp[0]+imgshp[1])/2

# cut off pixel pixel of the image and return it
def img_crop(im, edge = "both"):
    if edge=="h":
        new_image = im[:im.shape[0]-1, :, :]
    elif edge=="w":
        new_image = im[:, :im.shape[1]-1, :]
    else:
        new_image = im[:im.shape[0]-1, :im.shape[1]-1, :]
    return new_image

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
    pdict['im_h'+str(k)] = image.shape[0]
    pdict['im_w'+str(k)] = image.shape[1]
    pdict['im_kern_h' + str(k)] = 'N/A'
    pdict['im_kern_w' + str(k)] = 'N/A'
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
        pdict['im_h' + str(k)] = image.shape[0]
        pdict['im_w' + str(k)] = image.shape[1]
        pdict['im_kern_h' + str(k)] = krn_h
        pdict['im_kern_w' + str(k)] = krn_w
    # note that the final k is stored in "k"

    # print dictionary
    for key, value in pdict.items():
        #print(key,": \t",value)
        continue

    # View raw image
    plt.figure()
    plt.imshow(image)

    # View rgb channels
    plt.figure()
    plt.imshow(np.rot90(
        np.concatenate((np.concatenate((rotIm(image[:, :, 0]), rotIm(image[:, :, 1]))), rotIm(image[:, :, 2])))))

    # view a specific image of the data
    #plt.figure()
    #plt.imshow(train_data[3].reshape(28, 28))
    #plt.show()
    #plt.clf()


    # MAKE AS FUNCTION PASSING IN AND RETURNING pdict
    # execute clearCut method and store in edge array for masking
    edgy_images = traceObjectsInImage(image) # later think about implementing different methods as an argument
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
        new_edge_array = np.zeros((im_h, im_w))
        for i in range(0, edg_size[0]):
            for j in range(0, edg_size[1]):
                #print("i=", i, ", j=", j, "  --- ", int(krn_h * i), " & ", int(krn_w * j))
                if pdict['im_edge_array' + str(k)][i,j] > 0.:
                    new_edge_array[int(krn_h * (i + 0.5)), int(krn_w * (j + 0.5))] = 1.

        # update k before loop ends
        k = k - 1
        pdict['im_edge_array' + str(k)] = new_edge_array

    plt.figure()
    plt.imshow(pdict['im_edge_array0'])

    # mask original image with edge array
    imageRaw[:,:,0][pdict['im_edge_array0'] > 0.] = 255
    plt.figure()
    plt.imshow(imageRaw)
    plt.show()
    exit()

    # view a random image of the data
    plt.clf()
    plt.figure()
    plt.imshow(train_data[randint(0, train_data.shape[0])].reshape(28, 28))
    plt.show()

main()
exit()