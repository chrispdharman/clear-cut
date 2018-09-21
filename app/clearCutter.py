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
def calcKernelSize(imgShape):
    img_h = imgShape[0]
    img_w = imgShape[1]

    # determine lowest denominator in image height
    k_h = 2
    while( img_h % k_h == 0 is False ):
        k_h = k_h + 1
        if (k_h > img_h):
            print("Error: the image height is a prime number. Cannot determine pooling kernel size.")
            exit()

    # determine lowest denominator in image height
    k_w = 2
    while (img_w % k_w == 0 is False):
        k_w = k_w + 1
        if (k_w > img_w):
            print("Error: the image width is a prime number. Cannot determine pooling kernel size.")
            exit()

    return k_h, k_w

# determine average image size
def imgMean(imgShp):
    return (imgShp[0]+imgShp[1])/2

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
    imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/Bob.jpeg"
    imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/minimal1.jpg"
    #imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/colorful1.jpeg"
    #imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/john1.jpg"
    #imagePath = "/Users/ch392/Documents/dataScience/personalStudy/clearCut/app/images/heathers_cats.jpg"
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

    # check if the image is too small to be pooled, then pool the image
    while (imgMean(image.shape) > 500):
        # calculate the smallest kernel size that fits into the image
        krn_h, krn_w = calcKernelSize(image.shape)
        print("krn_h=", krn_h, ", krn_w=", krn_w)

        # reduce image size, given calculated kernel (max pooling)
        image = block_reduce(image, (krn_h, krn_w, 1), np.max)
        print("New shape=",image.shape)

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

    # execute clearCut method
    traceObjectsInImage(image) # later think about implementing different methods as an argument
    plt.show()
    exit()

    # view a random image of the data
    plt.clf()
    plt.figure()
    plt.imshow(train_data[randint(0, train_data.shape[0])].reshape(28, 28))
    plt.show()

main()
exit()