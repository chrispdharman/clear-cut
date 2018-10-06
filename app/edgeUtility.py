import numpy as np
from random import randint
from skimage.measure import block_reduce
import matplotlib.pyplot as plt


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
    '''plt.figure()
    plt.imshow(np.absolute(gradImage.T), interpolation="nearest")
    plt.figure()
    plt.imshow(np.multiply((np.absolute(gradImage.T) < (1-imCut)*255),(np.absolute(gradImage.T) > imCut*255)))'''

    # merge channels
    mrgIm1 = mergeChannelsTracedImage(gradImage.T, origImage.shape)
    #plt.figure()
    #plt.imshow(mrgIm1)

    mrgIm2 = mergeChannelsTracedImage(
        np.multiply((np.absolute(gradImage.T) < (1 - imCut) * 255), (np.absolute(gradImage.T) > imCut * 255)),
        origImage.shape)
    #plt.figure()
    #plt.imshow(mrgIm2)

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
    #print("Image size: ", img.shape)
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

    # determine lowest denominator in image width
    k_w = 2
    while ( (img_w % k_w) == 0 is False ):
        k_w = k_w + 1
        if (k_w > img_w/2):
            print("Error: the image width is a prime number. Cannot determine pooling kernel size.")

            # function to remove one pixel layer off the image "width"
            newImg = img_crop(img, edge = "w")
            k_w = 2
            break

    #print("Newish shape=", newImg.shape)
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

# determine positions of edge vectors, return (? x 2) array
def edgePxlPos(edge_img):
    pos_vec = []
    shape = edge_img.shape
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            if edge_img[i,j] > 0.:
                pos_vec.append([i,j])
    pos_vec = np.array(pos_vec)
    #print("Array: ",pos_vec)
    #print("Shape: ",pos_vec.shape)
    return pos_vec

# return pxl at perimeter if you go out of bounds
def protPxl(pxl, max_cap):
    # return bounded value
    if pxl < 0:
        return 0
    elif pxl > max_cap:
        return max_cap - 1
    else:
        return pxl

# determine positions of edge vectors, return (? x 2) array.
def edgeKiller(edge_img, edge_pos, min_rad = 0):
    # iterate through edge pixels
    for k in range(0, edge_pos.shape[0]):
        coord = edge_pos[k]
        print("@",coord)
        print("(coord[0]-8*min_rad):(coord[0]+8*min_rad+1)=", protPxl((coord[0] - 8 * min_rad - 1), edge_img.shape[0]),
              ":", protPxl((coord[0] + 8 * min_rad + 2), edge_img.shape[0]))
        print("(coord[1]-8*min_rad):(coord[1]+8*min_rad+1)=", protPxl((coord[1] - 8 * min_rad - 1), edge_img.shape[1]),
              ":", protPxl((coord[1] + 8 * min_rad + 2), edge_img.shape[1]))

        # loop around the neighbouring pixels (excluding the pixel itself)
        count = 0
        for i in range(0, 9):
            # set increment in x, y, multiplier, and initial edge value
            dx = i % 3 - 1
            dy = i // 3 - 1
            mult = 1
            if not i == 4:
                # reiterate with an increasing multiplier until a non-edge pixel is found
                try:
                    while edge_img[coord[0] + mult * dx, coord[1] + mult * dy] > 0.:
                        #print("\t ...multiplying edge_value, step in same direction")
                        mult = mult + 1
                        count = count + 1
                except(IndexError):
                    #print("Edge reached perimeter of the image. No need to fill this edge.")
                    continue
        # change edge pixel to non-edge pixel if it is not surrounded by any other edge pixels
        #print("count=",count," min_rad=",min_rad)
        if count < (1 + 8 * min_rad):
            #print("\t ...removing edge pixel")
            grph_rad = 1 + 8 * min_rad

            plt.figure()
            plt.imshow(edge_img[protPxl((coord[0] - grph_rad), edge_img.shape[0]):protPxl((coord[0] + grph_rad + 1),
                                                                                          edge_img.shape[0]),
                       protPxl((coord[1] - grph_rad), edge_img.shape[1]):protPxl((coord[1] + grph_rad + 1),
                                                                                 edge_img.shape[1])])
            plt.show()

            edge_img[coord[0], coord[1]] = 0

    return edge_img

# determine positions of edge vectors, return (? x 2) array.
# edge_bias details how many edge pixels must be adjacent in ...
# ... the same direction before considering it an extendable edge
def edgeFiller(edge_img, edge_pos, edge_bias = 10):
    # iterate through edge pixels
    for k in range(0, edge_pos.shape[0]):
        coord = edge_pos[k]
        #print("@",coord)

        # loop around the neighbouring pixels (excluding the pixel itself)
        for i in range(0, 9):
            # set increment in x, y, multiplier, and initial edge value
            dx = i % 3 - 1
            dy = i // 3 - 1
            mult = 1
            edge_value = 0.03
            if not i == 4:
                # reiterate with an increasing multiplier until a non-edge pixel is found
                try:
                    while edge_img[coord[0] + mult * dx, coord[1] + mult * dy] > 0.:
                        #print("\t ...multiplying edge_value, step in same direction")
                        mult = mult + 1
                    # write the value of (mult * edge_value) to non-edge pixel, subject to the edge_bias parameter
                    if (mult > edge_bias):
                        #print("\t ...writing out (edge_value x ", mult, ")")
                        edge_img[coord[0] + i % 3 - 1, coord[1] + i // 3 - 1] = mult * edge_value
                except(IndexError):
                    #print("Edge reached perimeter of the image. No need to fill this edge.")
                    continue
    return edge_img

def randomPathEdgeRace(img, edgy_img):
    # update the edge pixel position list
    posList = edgePxlPos(edgy_img)

    # pick random edge pixel to start from
    initEdgePxl = posList[randint(0,posList.shape[0])]
    pxl_lst = [0, 0, 0, 0]
    print("Initial edge pixel=",initEdgePxl,"--> Value of ", edgy_img[initEdgePxl[0],initEdgePxl[1]])

    # determine the pixels around this initial pixel in each direction
    horizontal_lst = pxlLen(edgy_img, initEdgePxl, [0, 1],
                          edge=pxlLen(edgy_img, initEdgePxl, [0, -1],  edge=np.expand_dims(initEdgePxl, axis=0)))
    vertical_lst = pxlLen(edgy_img, initEdgePxl, [1, 0],
                          edge=pxlLen(edgy_img, initEdgePxl, [-1, 0],  edge=np.expand_dims(initEdgePxl, axis=0)))
    pos_grad_lst = pxlLen(edgy_img, initEdgePxl, [-1, 1],
                          edge=pxlLen(edgy_img, initEdgePxl, [1, -1],  edge=np.expand_dims(initEdgePxl, axis=0)))
    neg_grad_lst = pxlLen(edgy_img, initEdgePxl, [1, 1],
                          edge=pxlLen(edgy_img, initEdgePxl, [-1, -1], edge=np.expand_dims(initEdgePxl, axis=0)))

    print("Horizontal length = \t", len(horizontal_lst))
    print("Vertical length = \t", len(vertical_lst))
    print("+ Gradient length = \t", len(pos_grad_lst))
    print("- Gradient length = \t", len(neg_grad_lst))
    pxl_lst = [horizontal_lst, vertical_lst, pos_grad_lst, neg_grad_lst]
    pxl_radii = [len(horizontal_lst), len(vertical_lst),len(pos_grad_lst), len(neg_grad_lst)]

    rad = np.max(pxl_radii)
    start_line = (pxl_lst[np.argmin(pxl_radii)])
    #print("Start line shape = ", start_line.shape)

    # view the initial pixel region with the start line and initial pixel "shown"
    for i in range(0, len(start_line)):
        edgy_img[start_line[i,0], start_line[i,1]] = 2.0
    edgy_img[initEdgePxl[0],initEdgePxl[1]]= 2.5
    plt.figure()
    plt.imshow(edgy_img)
    plt.figure()
    plt.imshow(edgy_img[(initEdgePxl[0] - rad - 1):(initEdgePxl[0] + rad + 1),
               (initEdgePxl[1] - rad -1):(initEdgePxl[1] + rad + 1)])
    plt.show()

    # pick (but remember) a direction orthogonal to the smallest thickness to tend the path toward)


    exit()

    # ensure we return the right array
    if enclosedBool:
        objPerimeter = edgePath
    else:
        objPerimeter = ""

    return enclosedBool, objPerimeter

# determine the edge pixels before the first non-edge pixel in a pre-determined direction
def pxlLen(edgy_img, init_pxl, pxl_dir, edge):
    # initialise increments
    init_x, init_y = init_pxl
    dx, dy = pxl_dir
    dx_0, dy_0 = pxl_dir

    #print("Ingoing edge: ", edge)

    # store pixel positions in a list
    while edgy_img[init_x + dx, init_y + dy] > 0.:
        edge = np.append(edge, np.array([[init_x + dx, init_y + dy]]), axis = 0)
        # increment value of edge length by 1
        dx = dx + dx_0
        dy = dy + dy_0
        #print("@(",init_x + dx,",",init_y + dy,")=",edgy_img[init_x + dx, init_y + dy])

    #print("Outgoing edge: ", edge)
    return edge