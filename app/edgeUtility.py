import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from random import randint
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import time


# quick access to debug output
ShowPath = False
TimeNucleation = False
DebugStepPath = False

# object tracing method handler
def traceObjectsInImage(origImage, method = "gradient"):
    if method == "gradient":
        # good for handling sharp edges
        return traceObjectsInImage_gradient(origImage)
    elif method == "texture":
        # good for handling blurred edges
        return traceObjectsInImage_texture(origImage)
    else:
        print("No edge detection method specified. Stopping code execution.")
        exit()

# count if there are any coordinates surrounding the chosen_one
def cluster_counter(chosen_one, pxl_list, R = 10):
    coord_list = []
    for coord in range(0, len(pxl_list)):
        x, y = pxl_list[coord]
        if math.sqrt( (x - chosen_one[0])**2 + (y - chosen_one[1])**2 ) <= R:
            coord_list.append(pxl_list[coord])
    return coord_list

# boolean to check if this is a new direction or not
def new_direction(chsn_one, prev_directions):
    for prev_one in prev_directions:
        if chsn_one[0] == prev_one[0] and chsn_one[1] == prev_one[1]:
            return False
    return True

def reduce_iter(i):
    while i>5:
        i = i -8
    return i

def enclosed_points(remaining_pxls, dead_path, alive_direction, rad):
    # determine the alive points within the enclosed path: dead_path
    #print("(np.array(dead_path).T)[0]=", (np.array(dead_path).T)[0])
    y_lst = (np.array(dead_path).T)[0]
    min_y = np.min(y_lst)
    max_y = np.max(y_lst)
    #print("min_y=", min_y, " max_y=", max_y)

    # determine and remove all remaining pixels within the random path
    enc_list = []
    #print("dead_path=",dead_path)
    for y0 in range(min_y, max_y + rad, rad):
        # determine the domain of x within the random path at this value of y
        x_lst = []
        for pt in dead_path:
            #print("pt=",pt, " pt[0]=", pt[0], " and x0=", y0)
            if pt[0] == y0:
                x_lst.append(pt[1])
        #print(">x_lst=",x_lst)
        x_lst = np.array(x_lst)
        min_x = np.min(x_lst)
        max_x = np.max(x_lst)
        #print(">min_x=", min_x, " max_x=", max_x)
        for x0 in range(min_x, max_x + rad, rad):
            for cood in alive_direction:
                if cood[0]==y0 and cood[1]==x0:
                    #print(">>cood=", cood, " y0=", y0, " x0=", x0)
                    enc_list += cluster_counter(cood, remaining_pxls, R=rad)
    #print("enc_list=",enc_list)
    return enc_list

# cluster bubble nucleate building block code
# randomly select a pixel coordinate in the existing list
def clstr_nucleate(point, rad, lbl_no, remaining_pxls, cluster_list, border=0, iter_max = 9, init = False, end_counter = 1):
    timeIt = TimeNucleation
    if timeIt:
        t_0 = time.time()
    print("\t Nucleating...")
    # keep finding points in a clustered region
    iter = 0
    fully_nucleated = False
    alive_direction = []
    dead_direction = []
    while not fully_nucleated:
        # update counter, classification label and create new list
        start_counter = end_counter
        iter = iter + 1

        if timeIt:
            t_prev = time.time()

        # determine the next direction for this iter
        if iter > 1:
            mult = rad * (1 + (iter - 2) // 8)
            if (iter - 2) % 8 < 4:
                dx, dy = [(2 * ((reduce_iter(iter) - 2) % 2) - 1) * mult,
                          (2 * ((reduce_iter(iter) - 2) // 2) - 1) * mult]
            else:
                dx, dy = [(int(round( -1 * math.cos((iter-6)*math.pi/2) ))) * mult,
                          (int(round( math.sin((iter-6)*math.pi/2) ))) * mult]
        else:
            dx, dy = [0, 0]

        if timeIt:
            print("\t \t 1. Determine direction: ",time.time()-t_prev, "seconds")
            t_prev = time.time()

        chsn_one = [point[0] + dy, point[1] + dx]
        #print("iter=",iter,"\t chsn_one=", chsn_one)
        #print("dx=", dx, " dy=", dy)

        ## count number of pixels in radius "R" pxls around it, add these to the new cluster list.
        # make sure you don't re-evaluate a previous circle
        new_dir_bool = new_direction(chsn_one, dead_direction + alive_direction)
        if new_dir_bool:
            inc_list = cluster_counter(chsn_one, remaining_pxls, R = rad)
        else:
            # already covered this direction
            inc_list = []
        end_counter = start_counter + len(inc_list)
        #print("end_counter = ", end_counter)
        if timeIt:
            print("\t \t 2. Counting pixels: ",time.time()-t_prev, "seconds")
            t_prev = time.time()

        ## If the first evaluation does not have a change in counter value, append to the cluster_list["label_0"] list
        if iter == 1 and init:
            if end_counter == start_counter:
                cluster_list["label_0"] += inc_list
                fully_nucleated = True
            else:
                cluster_list["label_" + str(lbl_no)] = []
                # go to next iter
        '''elif new_dir_bool:
            if end_counter == start_counter:
                ## If the number has not changed, try another direction until all directions are exhausted
                print("Exhausted direction")
            else:
                ## if there are more pixels than before, keep going in that direction
                print("Continue in this direction")'''

        if end_counter == start_counter:
            dead_direction.append([chsn_one[0]+border, chsn_one[1]+border])
        else:
            alive_direction.append([chsn_one[0]+border, chsn_one[1]+border])

        # append inc_list to current cluster list
        #print("Before: ",cluster_list["label_" + str(lbl_no)])
        #print("--Add inc_list=",inc_list)
        cluster_list["label_" + str(lbl_no)] += inc_list
        #print("After: ", cluster_list["label_" + str(lbl_no)])

        if timeIt:
            print("\t \t 3. Updating lists/dictionaries: ",time.time()-t_prev, "seconds")
            t_prev = time.time()

        # this might present a problem for post-init determination
        '''if not inc_list == []:
            #for coord in range(0, len(inc_list)):
            #    remaining_pxls.remove(inc_list[coord])
            # See https://stackoverflow.com/questions/21510140/best-way-to-remove-elements-from-a-list
            remaining_pxls = [c for c in remaining_pxls if c not in inc_list]

        if timeIt:
            print("\t \t 4. Removing found pixels: ",time.time()-t_prev, "seconds")
            t_prev = time.time()'''

        if iter == iter_max:
            fully_nucleated = True

    #return remaining_pxls, cluster_list, alive_direction, dead_direction
    return cluster_list, alive_direction, dead_direction

# object tracing texture method
def traceObjectsInImage_texture(origImage):
    # gradImage: create numpy 2D array of size (2n-1) of the original
    dimY, dimX, chanls = origImage.shape

    # append an image (in x-direction) for each of the separate channels
    textureImage = np.zeros(shape=(dimY, dimX, 2), dtype = float)
    #textureImage = np.zeros(shape=(dimY, dimX, 3), dtype=float)

    # plot the red pixel pixel value versus the (r-g) % difference and (r-b) % difference
    # plot3D pre-setup
    #fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # loop over each dimension, populating the textureImage with various labels
    remaining_pxls = []
    for j in range(0, dimY):
        #print("j/dimY =",j,"/",dimY)
        for i in range(0, dimX):

            # get difference between red and an `other' colour channel
            pt = origImage[j, i]
            textureImage[j, i][0] = (pt[0] - pt[1])
            textureImage[j, i][1] = (pt[0] - pt[2])
            #textureImage[j, i][2] = pt[0]

            #print("textureImage[",j,",",i,"]=",textureImage[j,i])
            # get a coordinate list of unclassified pixel coordinate
            remaining_pxls.append([(pt[0] - pt[1]), (pt[0] - pt[2])])

            # plot3D layer creation
            #new_pt = textureImage[j, i]
            #ax.scatter(new_pt[0], new_pt[1], new_pt[2])

    # plot the (r-g) % difference and (r-b) % difference
    #plt.figure()
    #plt.scatter(textureImage[:,:,0], textureImage[:,:,1])
    #plt.xlabel("r-g")
    #plt.ylabel("r-b")
    #plt.show()

    # keep track off original remaining pixels
    orig_remaining_pxls = remaining_pxls.copy()

    # classify clustered regions
    #print("remaining_pxls=", remaining_pxls)
    cluster_list = {
        "label_0" : []
    }
    rad = 8
    lbl_no = 0
    end_counter = 1
    print("No. of remaining pxls (start) = ", len(remaining_pxls))
    print("cluster_list (start) = ", cluster_list)
    # keep finding clusters until all pxls have been labelled
    while len(orig_remaining_pxls) > 0:
        alive_direction = []
        dead_direction = []
        lbl_no += 1
        print("Cluster [", lbl_no,"] determination in progress...")

        # check the points in cluster label 1 have been removed
        if lbl_no == 2:
            plt.figure()
            plt.scatter((np.array(orig_remaining_pxls).T)[0], (np.array(orig_remaining_pxls).T)[1], s=1)
            plt.show()

        # setup graph
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        #ax.scatter(textureImage[:, :, 0], textureImage[:, :, 1], s=1)
        ax.scatter((np.array(orig_remaining_pxls).T)[0], (np.array(orig_remaining_pxls).T)[1], s=1)

        # put a criteria here to set all_outer_dead = True if all outer bubbles are dead...
        # edgy_img = np.zeros((255, 255))
        brdr = 2 * rad
        edgy_img = np.zeros((255 + 2 * brdr, 255 + 2 * brdr))
        for u in range(0, edgy_img.shape[0]):
            for v in range(0, edgy_img.shape[1]):
                # make a border around the image so that it does not go out of bounds?
                if u < brdr or v < brdr or u > 255 + brdr or v > 255 + brdr:
                    edgy_img[u, v] = -0.1

        # randomly select a pixel coordinate in the existing list
        chosen_one = remaining_pxls[randint(0, len(remaining_pxls))]
        chosen_one = [255//2, 255//2]
        #chosen_one = [0, 2]
        print("chosen_one=",chosen_one)

        # initial nucleation
        #remaining_pxls, cluster_list, alive, dead = clstr_nucleate(chosen_one, rad, lbl_no, orig_remaining_pxls,
        #                                                           cluster_list, border=brdr, iter_max=17, init=True)
        cluster_list, alive, dead = clstr_nucleate(chosen_one, rad, lbl_no, orig_remaining_pxls,
                                                             cluster_list, border=brdr, iter_max=17, init=True)
        alive_direction += alive
        dead_direction += dead

        # populate dead_directions with value 1 in edgy_img
        for edge in dead_direction:
            edgy_img[edge[0], edge[1]] = 1

        #print("\t alive_direction=", alive_direction)
        #print("\t dead_direction=", dead_direction)

        # reiterate nucleation until all outer bubbles are dead
        all_outer_dead = False
        # initiate new alive directions with current list
        alive = alive_direction.copy()
        while not all_outer_dead:
            # for any outermost bubbles that found new pixels, nucleate another bubble around them
            #for dirs in alive_direction:
            idx = -1
            while idx < len(alive_direction):
                idx += 1
                dirs = alive_direction[idx]
                print("\t Iteration: ", idx,"/",len(alive_direction))
                # re-nucleates on alive circles that are 2*rad distance away from initial point
                #if abs(dirs[0]-chosen_one[0])==2*rad or abs(dirs[1]-chosen_one[1])==2*rad:

                # re-nucleates on alive circles that are 2*rad distance away from initial point
                if not ((dirs[0] == chosen_one[0]) and (dirs[1] == chosen_one[1])):
                    #print("\t I'm outer alive!: ",dirs, "--> (", abs(dirs[0]-chosen_one[0]),",",abs(dirs[1]-chosen_one[1]))
                    cluster_list, alive, dead = clstr_nucleate(dirs, rad, lbl_no, orig_remaining_pxls,
                                                                               cluster_list, border = brdr)
                    # make sure alive directions are not overwritten by dead ones!
                    alive_direction += alive
                    for al in alive_direction:
                        for dd in dead:
                            if al[0] == dd[0] and al[1] == dd[1]:
                                dead.remove(dd)
                                #del dead[dd]
                    dead_direction += dead
                    #print("\t alive_direction=", alive_direction)
                    #print("\t dead_direction=", dead_direction)


                    #print("dead_direction=",dead_direction)
                    #print("No. of dead directions =", len(dead_direction))

                    # multiple (5) attempts at finding a random path within the dead directions
                    attempt = 0
                    while len(dead) > 0 and not all_outer_dead and attempt < 5:
                        attempt += 1
                        # populate dead_directions with value 1 in edgy_img
                        for edge in dead_direction:
                            edgy_img[edge[0], edge[1]] = 1

                        #print("attempt=",attempt)
                        all_outer_dead, dead_path = randomPathEdgeRace(edgy_img, adj_size = rad, border = brdr, showPath = ShowPath)

                        # dead_path needs shifting by border in the x- and y-direction

                        # if enclosed path is too small, add enclosed pxls to label 0
                        if all_outer_dead and len(dead_path)<9:
                            clstr_pxls = enclosed_points(remaining_pxls, dead_path, alive_direction, rad)
                            #print(">>>clstr_pxls=",clstr_pxls)
                            #print(">>>len(clstr_pxls)=",len(clstr_pxls))

                            # do not label points with no or only one alive circle inside the enclosed path
                            if len(clstr_pxls) == 0:
                                # disregard an enclosed path found with no alive circles
                                all_outer_dead = False
                            elif len(clstr_pxls) == 1:
                                # do not label points residing in only a single alive circle
                                all_outer_dead = False

                                # remove from current label, put into label 0 instead
                                cluster_list["label_0"] += clstr_pxls
                                for bad in clstr_pxls:
                                    for inst in cluster_list["label_" + str(lbl_no)]:
                                        if bad[0]==inst[0] and bad[1]==inst[1]:
                                            del cluster_list["label_" + str(lbl_no)][inst]
                    #print("\t all_outer_dead=",all_outer_dead)

                    # break for loop if enclosed path found: no need to iterate over more alive directions
                    if all_outer_dead:
                        break
        # shift back all coordinates in alive_direction, dead_direction and dead_path

        # graphics
        for i in range(0,len(alive_direction)):
            coor = alive_direction[i]
            alive_direction[i] = [coor[0]-brdr, coor[1]-brdr]
            ax.add_patch(Circle((coor[0]-brdr, coor[1]-brdr), rad, facecolor=(0, 0, 0, 0), edgecolor='green'))
        for i in range(0, len(dead_direction)):
            coor = dead_direction[i]
            dead_direction[i] = [coor[0] - brdr, coor[1] - brdr]
            ax.add_patch(Circle((coor[0]-brdr, coor[1]-brdr), rad, facecolor=(0, 0, 0, 0), edgecolor='red'))
        for i in range(0, len(dead_path)):
            coor = dead_path[i]
            dead_path[i] = [coor[0] - brdr, coor[1] - brdr]
            ax.add_patch(Circle((coor[0]-brdr, coor[1]-brdr), rad, facecolor=(0, 0, 0, 0), edgecolor='blue'))

        # plot the (r-g) % difference and (r-b) % difference
        plt.xlabel("r-g")
        plt.ylabel("r-b")
        plt.show()

        enc_list = enclosed_points(remaining_pxls, dead_path, alive_direction, rad)
        if len(enc_list) > 0:
            for enc in enc_list:
                y = 0
                # iterate through the list which may have indices removed!
                while y < len(remaining_pxls):
                    item = remaining_pxls[y]
                    if enc[0]==item[0] and enc[1]==item[1]:
                        # if the coordinate in enc_list is found in remaining_pxls, remove it
                        del remaining_pxls[y]
                    else:
                        # otherwise move to the next index in remaining_pxls
                        y += 1
            #remaining_pxls.remove(enc_list)
        else:
            print("No pixels in the path found")

        # update original remaining pixels to remove the pxls that were just labelled
        # repeat cluster finding until remaining_pxls is empty
        orig_remaining_pxls = remaining_pxls.copy()

        print("No. of remaining pxls (end) = ", len(remaining_pxls))
        print("cluster_list (end) = ", cluster_list)

    print("Labelled all data!")
    exit()


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

    edge_array = mergeChannelsTracedImage(
        np.multiply((np.absolute(gradImage.T) < (1 - imCut) * 255), (np.absolute(gradImage.T) > imCut * 255)),
        origImage.shape)
    #plt.figure()
    #plt.imshow(mrgIm2)

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

# object tracing one-layer gradient method
def traceObjectsInImage_gradient(origImage):
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

    edge_array = mergeChannelsTracedImage(
        np.multiply((np.absolute(gradImage.T) < (1 - imCut) * 255), (np.absolute(gradImage.T) > imCut * 255)),
        origImage.shape)
    #plt.figure()
    #plt.imshow(mrgIm2)

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

    # reduce gradient array to original image shape. Max pool gradient array using 2x2 kernel
    return block_reduce(mrgdImg, (2, 2), np.max)

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
def edgePxlPos(edge_img, border = 0):
    pos_vec = []
    shape = edge_img.shape
    #print("Edge shape=",shape)
    #print("edge_img=",edge_img)
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            if edge_img[i,j] > 0.:
                #print("edge_img[i,j]=",edge_img[i,j])
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


def edgeKill(edg_img, coord, radius, task = "border-count"):
    #print("@", coord)

    # initial counter and pre-define useful values
    count = 0
    border_size = 2*radius + 1

    # run over the square of pixels surrounding "radius"-pixels around coord
    for i in range(0, border_size**2):
        dx = (i % border_size) - radius
        dy = (i // border_size) - radius
        #print("-->(",coord[0]+dx,",",coord[1]+dy,")")
        if task == "wipe-edges":
            # remove all edge pixels --> change to non-edge pixels
            edg_img[coord[0]+dx, coord[1]+dy] = 0.
            #print("killing (",coord[0]+dx,",",coord[1]+dy,")")
        elif task == "border-count":
            # make sure we are looking at a border pixel
            if np.abs(dx)==radius or np.abs(dy)==radius :
                try:
                    if edg_img[coord[0]+dx, coord[1]+dy] == 0.:
                        # increment count by 1 to say we found another non-edge pixel on the border
                        count = count + 1
                    else:
                        # break out of the for loop if we find an edge on the border (return the original edge image)
                        #print("Broke because border pixel has value ",edg_img[coord[0]+dx, coord[1]+dy])
                        break
                except IndexError:
                    # the central edge pixel is too close to the image perimeter
                    continue
        else:
            print("Task is not specified. Stopping the code...")
            exit()

    # check if the border is all non-edge
    if count == (8*radius):
        #print("Border is all non-edge")
        '''
        # for debug
        if radius == 1:
            plt.figure()
            plt.imshow(edg_img[protPxl((coord[0] - radius), edg_img.shape[0]):protPxl((coord[0] + radius + 1),
                                                                                      edg_img.shape[0]),
                       protPxl((coord[1] - radius), edg_img.shape[1]):protPxl((coord[1] + radius + 1),
                                                                              edg_img.shape[1])])
            plt.show()
        '''
        edg_img = edgeKill(edg_img, coord, radius = (radius - 1), task = "wipe-edges")

    '''
    # for debug
    if task=="wipe-edges" and radius == 0:
        print("Wiped all edges")
        
        plt.figure()
        plt.imshow(edg_img[protPxl((coord[0] - radius), edg_img.shape[0]):protPxl((coord[0] + radius + 1),
                                                                                  edg_img.shape[0]),
                   protPxl((coord[1] - radius), edg_img.shape[1]):protPxl((coord[1] + radius + 1),
                                                                          edg_img.shape[1])])
        plt.show()
    '''
    return edg_img


# determine positions of edge vectors, return (? x 2) array.
def edgeKiller(edge_img, objectTolerance = 1):
    # iterate through pixels distance from chosen edge pixel, going up to the specified objectTolerance value
    for r in range(1, objectTolerance):
        ## improve by returning coordinates to kill, instead of the whole edge image
        # iterate through edge pixels (updated edge_pos vector)
        edge_pos = edgePxlPos(edge_img)
        for k in range(0, edge_pos.shape[0]):
            edge_img = edgeKill(edge_img, edge_pos[k], radius = r)
    return edge_img

# determine positions of edge vectors, return (? x 2) array.
# edge_bias details how many edge pixels must be adjacent in ...
# ... the same direction before considering it an extendable edge
def edgeFiller(edge_img, edge_bias = 10):
    edge_pos = edgePxlPos(edge_img)
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

# determine the pixels around this pixel in each direction
def pxlThickness(edgy_img, pxl):
    pxl_lst = [0, 0, 0, 0]
    # print("Initial edge pixel=",initEdgePxl,"--> Value of ", edgy_img[initEdgePxl[0],initEdgePxl[1]])
    horizontal_lst = pxlLen(edgy_img, pxl, [0, 1],
                            edge=pxlLen(edgy_img, pxl, [0, -1], edge=np.expand_dims(pxl, axis=0)))
    vertical_lst = pxlLen(edgy_img, pxl, [1, 0],
                          edge=pxlLen(edgy_img, pxl, [-1, 0], edge=np.expand_dims(pxl, axis=0)))
    pos_grad_lst = pxlLen(edgy_img, pxl, [-1, 1],
                          edge=pxlLen(edgy_img, pxl, [1, -1], edge=np.expand_dims(pxl, axis=0)))
    neg_grad_lst = pxlLen(edgy_img, pxl, [1, 1],
                          edge=pxlLen(edgy_img, pxl, [-1, -1], edge=np.expand_dims(pxl, axis=0)))

    #print("Horizontal length = \t", len(horizontal_lst))
    #print("Vertical length = \t", len(vertical_lst))
    #print("+ Gradient length = \t", len(pos_grad_lst))
    #print("- Gradient length = \t", len(neg_grad_lst))
    pxl_lst = [horizontal_lst, vertical_lst, pos_grad_lst, neg_grad_lst]
    pxl_radii = [len(horizontal_lst), len(vertical_lst), len(pos_grad_lst), len(neg_grad_lst)]

    return pxl_lst, pxl_radii

# unit vector dot product of two vectors (dot divided by both magnitudes)
def unit_vec_dot(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / ( np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)) )

# select a random direction vector and return the path traversed
def random_step_direction(edgy_img, new_pxl, start_line, prev_path_vec, adj_size, step_dist = 1, debug = False):
    if debug:
        # save the original pixel
        print("@",new_pxl)

    try:
        # using the start line coordinates, determine the vector in which the start line points
        if len(start_line) == 1:
            # determine the axis in which two non-edge pixels are either side of new_pxl
            if edgy_img[new_pxl[0]-adj_size,new_pxl[1]] == 0. and edgy_img[new_pxl[0]+adj_size,new_pxl[1]] == 0.:
                start_vec = [adj_size, 0]
            elif edgy_img[new_pxl[0]-adj_size,new_pxl[1]-adj_size] == 0. and edgy_img[new_pxl[0]+adj_size,new_pxl[1]+adj_size] == 0.:
                start_vec = [adj_size, adj_size]
            elif edgy_img[new_pxl[0]-adj_size,new_pxl[1]+adj_size] == 0. and edgy_img[new_pxl[0]+adj_size,new_pxl[1]-adj_size] == 0.:
                start_vec = [adj_size, -1*adj_size]
            else:
                start_vec = [0, adj_size]
        else:
            start_vec = np.sign([start_line[0][0]-start_line[-1][0], start_line[0][1]-start_line[-1][1]])
    except IndexError:
        print("Exceeded image perimeter. This special case has yet to be accounted for")
        return []

    if debug:
        print("1. start_vec=", start_vec)


    # determine the vector orthogonal to the start_line's vector
    if np.abs(start_vec[0]*start_vec[1]) == 0:
        path_vec = [start_vec[1], start_vec[0]]
    else:
        path_vec = [-start_vec[0],start_vec[1]]
    #print("path_vec=",path_vec)

    # NOTE: first argument is in the y-direction!
    # NOTE: second argument is in the x-direction!

    # find a vector that runs along the edge for roughly half the thickness of the pixel
    all_edge = True
    went_dist = False
    vec_list = [[-adj_size, -adj_size], [-adj_size, 0], [-adj_size, adj_size], [0, -adj_size],
                [0, adj_size], [adj_size, -adj_size], [adj_size, 0], [adj_size, adj_size]]
    #print("vec_list=",vec_list)
    bad_vecs = []
    if debug:
        print("\t 2. prev_path_vec=", prev_path_vec)
    while all_edge and not went_dist and len(vec_list) > 0:

        # reset all_edge boolean to iterate through each path_vec direction
        all_edge = True
        step_path = []

        # check if this is the first random step (i.e. no previous path vector) or not
        if prev_path_vec[0] == 0  and prev_path_vec[1] == 0:
            #print("start_vec=",start_vec)
            vec_list.remove([start_vec[0], start_vec[1]])
            vec_list.remove([-1*start_vec[0], -1*start_vec[1]])
        else:
            # remove path vectors that do not satisfy the scalar product constraint
            for k in range(0, len(vec_list)):
                #pos_path_vec = np.array(vec_list[k])
                pos_path_vec = vec_list[k]
                #print("-vec_list[k] =", vec_list[k], " has dot prod=",unit_vec_dot(prev_path_vec, pos_path_vec), " and adjacent edge=",edgy_img[new_pxl[0]+pos_path_vec[0],new_pxl[1]+pos_path_vec[1]])
                if not (prev_path_vec[0] == 0 and prev_path_vec[1] == 0):
                    # print("-/-dot product =", unit_vec_dot(prev_path_vec, pos_path_vec))
                    try:
                        if unit_vec_dot(prev_path_vec, pos_path_vec) < 0. or edgy_img[new_pxl[0]+pos_path_vec[0],new_pxl[1]+pos_path_vec[1]]==0:
                            bad_vecs.append(vec_list[k])
                    except IndexError:
                        print("Exceeded image perimeter. This special case has yet to be accounted for")
                        return []
            for k in range(0, len(bad_vecs)):
                vec_list.remove(bad_vecs[k])
            #print("Only ", len(vec_list), " vectors survived")

        # randomly go through all possible adjacent directions
        if len(vec_list) > 0:
            path_vec = vec_list[randint(0, len(vec_list) - 1)]
        else:
            if debug:
                print("All vectors eliminated")
            break
        if debug:
            print("\t \t  3. path_vec=", path_vec)

        # move half the value of pixel thickness in the path_vec direction
        for i in range(1, step_dist + 1):
            new_y, new_x = new_pxl + np.multiply(i, path_vec)
            #print("---value of (", new_y, ",", new_x, ")=", edgy_img[new_y, new_x])
            all_edge = (edgy_img[new_y, new_x] > 0.)
            # stop if we hit a non-edge: try another direction
            if not all_edge:
                break
            # stop if went the desired distance: successful path found!
            if i==step_dist:
                went_dist = True

    if debug:
        print("\t \t \t all_edge=", all_edge, "\t went_dist=", went_dist, "\t len(vec_list)=", len(vec_list))
        print("\t \t \t plausible directions=", vec_list)

    if len(vec_list) > 0:
        # move half the value of pixel thickness in the path_vec direction
        # this time storing the pixel coordinates in a list to return
        # only store if the path_direction is valid, and the first adjacent pxl is an edge
        if debug:
            print("\t \t \t \t 4. (", new_y-path_vec[0], ",", new_x-path_vec[1], ")=", edgy_img[new_y-path_vec[0], new_x-path_vec[1]],"\n")
        if edgy_img[new_y-path_vec[0], new_x-path_vec[1]] > 0.:
            for i in range(1, step_dist):
                new_y, new_x = new_pxl + np.multiply(i, path_vec)
                step_path.append([new_y,new_x])
        else:
            if debug:
                print("No plausible direction")

    if debug:
        print("\t \t \t \t \t 5. step_path=",step_path)

    return step_path

# edge_img is a rectangular array of 1s and 0s
# adj_size details how far an "adjacent" pixel is considered
def randomPathEdgeRace(edgy_img, adj_size = 1, border = 0, showPath = False):
    # update the edge pixel position list
    #print("len(edgy_img>0)=", len(edgy_img > 0))
    posList = edgePxlPos(edgy_img, border = border)
    #print("posList=", posList)

    if showPath:
        graph_img = np.copy(edgy_img)

    # pick random edge pixel to start from
    initEdgePxl = posList[randint(0,posList.shape[0]-1)]
    #initEdgePxl = [56, 150]
    pxl_lst, pxl_radii = pxlThickness(edgy_img, initEdgePxl)
    init_start_line = (pxl_lst[np.argmin(pxl_radii)])
    #print("init_start_line=",init_start_line)
    #print("Start line shape = ", start_line.shape)

    # view the initial pixel region with the start line and initial pixel "shown"
    #for i in range(0, len(init_start_line)):
    #    edgy_img[init_start_line[i,0], init_start_line[i,1]] = 2.0
    #edgy_img[initEdgePxl[0],initEdgePxl[1]]= 2.5
    #plt.figure()
    #plt.imshow(edgy_img)
    #init_rad = np.max(pxl_radii)
    #plt.figure()
    #plt.imshow(edgy_img[(initEdgePxl[0] - init_rad - 1):(initEdgePxl[0] + init_rad + 1),
    #           (initEdgePxl[1] - init_rad -1):(initEdgePxl[1] + init_rad + 1)])
    #plt.show()

    # pick (but remember) a direction orthogonal to the smallest thickness to tend the path toward)
    crossed_start_line = False
    step = 0
    new_pxl = initEdgePxl
    edge_path = np.expand_dims(initEdgePxl, axis = 0)
    path_vec = [0, 0]
    #while (not crossed_start_line) and (step < 5000):
    while not crossed_start_line:
        # update start_line from previous pxl_radii data
        start_line = (pxl_lst[np.argmin(pxl_radii)])
        prev_pxl = new_pxl
        #print("start_line=",start_line)

        # pick a semi-random direction, roughly orthogonal to the start line and return the path
        step_path = random_step_direction(edgy_img, new_pxl, start_line, path_vec, adj_size, step_dist = len(start_line)+1, debug = DebugStepPath)
        #print("step_path: ", np.array(step_path))

        if len(step_path) < 1:
            # break random path as no path can be found
            #print("No path found or image perimeter detected")
            break
        else:
            new_pxl = step_path[-1]
            edge_path = np.concatenate((edge_path, np.array(step_path)), axis=0)
            #print("@ step=", step, ",\t edge_path: ", edge_path)

        # check if we cross the initial start line
        #print("init_start_line=",init_start_line)
        for i in range(0, len(step_path)):
            #print("step_path[i]=",step_path[i])
            #print("step_path[i] in init_start_line =", step_path[i] in init_start_line)
            for j in range(0, len(init_start_line)):
                if step_path[i][0] == init_start_line[j][0] and step_path[i][1] == init_start_line[j][1]:
                    crossed_start_line = True
                    #print("Crossed the start line!")

        # determine thicknesses of new pixel
        pxl_lst, pxl_radii = pxlThickness(edgy_img, new_pxl)
        step = step + 1

        # update start line and new pxl starting point
        start_line = (pxl_lst[np.argmin(pxl_radii)])
        # take the new pixel as the midpoint of the start line
        #new_pxl = start_line[len(start_line) // 2 - 1]
        # take the new pixel as the final point of the last path
        new_pxl = step_path[-1]

        #print("start_line=",start_line)
        #print("new_pxl=", new_pxl)

        if showPath:
            # graphics
            for i in range(0, len(step_path)):
                graph_img[step_path[i][0], step_path[i][1]] = 1.5
            for i in range(0, len(start_line)):
                graph_img[start_line[i, 0], start_line[i, 1]] = 2.0
            graph_img[new_pxl[0], new_pxl[1]] = 2.25
            #print("step_path=",step_path)
            #print("start_line=", start_line)
            #print("new_pxl=",new_pxl)

            print("\t \t [Display plot, pause code]")
            if new_pxl[0]<0 or new_pxl[0]>graph_img.shape[0] or new_pxl[1]<0 or new_pxl[1]>graph_img.shape[1]:
                print("\t \t \t >[new_pxl=",new_pxl," out of scope. Bad graph]")
            plt.figure()
            plt.imshow(graph_img[protPxl(new_pxl[0] - 49, graph_img.shape[0]):protPxl(new_pxl[0] + 50, graph_img.shape[0]),
                       protPxl(new_pxl[1] - 49, graph_img.shape[1]):protPxl(new_pxl[1] + 50, graph_img.shape[1])])
            #plt.imshow(graph_img)
            plt.show()

        # determine previous path_vec
        step_path.insert(0, prev_pxl)
        path_vec = np.sign([step_path[-1][0] - step_path[0][0], step_path[-1][1] - step_path[0][1]])
        path_vec = [path_vec[0]*adj_size,path_vec[1]*adj_size]

    # generate edge_path image
    #edge_path_img = np.zeros(edgy_img.shape)
    #edge_path_img[edge_path.T[0],edge_path.T[1]] = 2.5
    #plt.figure()
    #plt.imshow(edge_path_img)

    #edgy_img[new_pxl[0], new_pxl[1]] = 2.5
    #plt.figure()
    #plt.imshow(edgy_img[protPxl(new_pxl[0] - 29, edgy_img.shape[0]):protPxl(new_pxl[0] + 30, edgy_img.shape[0]),
    #           protPxl(new_pxl[1] - 29, edgy_img.shape[1]):protPxl(new_pxl[1] + 30, edgy_img.shape[1])])
    #plt.figure()
    #plt.imshow(edgy_img)
    #plt.figure()
    #plt.imshow(edgy_img+edge_path_img)
    #plt.show()

    # ensure we return the right array
    #if crossed_start_line:
        # add coordinate on the start line to complete the edge
        # check direction in which the final path vector crossed?
    #else:
    #    objPerimeter = ""

    return crossed_start_line, edge_path

# determine the edge pixels before the first non-edge pixel in a pre-determined direction
def pxlLen(edgy_img, init_pxl, pxl_dir, edge):
    # initialise increments
    init_x, init_y = init_pxl
    dx, dy = pxl_dir
    dx_0, dy_0 = pxl_dir

    #print("Ingoing edge: ", edge)
    try:
        # store pixel positions in a list
        while edgy_img[init_x + dx, init_y + dy] > 0.:
            edge = np.append(edge, np.array([[init_x + dx, init_y + dy]]), axis = 0)
            # increment value of edge length by 1
            dx = dx + dx_0
            dy = dy + dy_0
            #print("@(",init_x + dx,",",init_y + dy,")=",edgy_img[init_x + dx, init_y + dy])
    except IndexError:
        print("Exceeded image perimeter. This special case has yet to be accounted for")

    #print("Outgoing edge: ", edge)
    return edge

# function to generate and display edge path on image
def edge_show(edgy_images, objEdgeArray):
    # generate edge_path image
    edge_path_img = np.zeros(edgy_images.shape)
    edge_path_img[objEdgeArray.T[0], objEdgeArray.T[1]] = 2.5
    # plt.figure()
    # plt.imshow(edge_path_img)

    plt.figure()
    plt.imshow(edgy_images + edge_path_img)
    plt.show()