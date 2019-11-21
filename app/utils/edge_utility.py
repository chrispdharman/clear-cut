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
        self.debug_step_path = False
        self.show_path = False
        self.time_cluster = False
        self.time_nucleation = False
        self.time_counter = False

        # increase csv file size limit
        csv.field_size_limit(sys.maxsize)

    def cluster_counter(self, chosen_one, pxl_list, R = 10, return_count = False):
        '''
        Count if there are any coordinates surrounding the chosen_one
        ### BOTTLENECK IN THE CLUSTER ALGORITHM
        Can this be reduced to less than 0.63 seconds?
        '''
        coord_list = []
        if self.time_counter:
            t_prev = time.time()

        #for coord in range(0, len(pxl_list)):
        if return_count:
            # list comprehension style!
            coord_list += [
                coord
                for coord in pxl_list
                if math.sqrt((coord[0] - chosen_one[0])**2 + (coord[1] - chosen_one[1])**2 ) <= R
            ]
            return coord_list
        
        # go through all pxl_list
        for k in range(0, len(pxl_list)):
            x, y = pxl_list[k]
            #  return the coordinate if at least one pxl is found
            if self.__within_radius(x, y, chosen_one, R):
                return [x, y]

        if self.time_counter:
            print("\t \t Counting pixels (return_count=",return_count,"): ", time.time() - t_prev, "seconds")

        return coord_list

    def new_direction(self, chosen_one, prev_directions):
        '''
        Check if this is a new direction or not
        '''
        seen_direction = [
            chosen_one[0] == prev_one[0] and chosen_one[1] == prev_one[1]
            for prev_one in prev_directions
        ]
        return not any(seen_direction)

    def reduce_iter(self, i):
        return i - (i>5)*8

    def enclosed_points(self, remaining_pxls, dead_path, alive_direction, rad):
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
                        enc_list += cluster_counter(cood, remaining_pxls, R=rad, return_count=True)
        #print("enc_list=",enc_list)
        return enc_list

    # cluster bubble nucleate building block code
    # randomly select a pixel coordinate in the existing list
    def clstr_nucleate(self, point, rad, lbl_no, remaining_pxls, cluster_list, border=0, iter_max = 9, init = False, end_counter = 1):
        timeIt = TimeNucleation
        if timeIt:
            t_0 = time.time()
        #print("\t Nucleating...")
        # keep finding points in a clustered region
        iter = 0
        fully_nucleated = False
        alive_direction = []
        dead_direction = []
        while not fully_nucleated:
            # update counter, classification label and create new list
            start_counter = end_counter
            iter += 1

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

            if timeIt:
                print("\t \t 1.1 Determine new direction: ",time.time()-t_prev, "seconds")
                t_prev = time.time()

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
            #print("After: ", cluster_list["label_" + str(lbl_no)])

            # update dictionary after cluster algorithm completes
            # cluster_list["label_" + str(lbl_no)] += inc_list

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

    # def rotate image 90 deg CW shortcut
    def rotIm(self, img):
        return np.rot90(img, 1, (1,0))

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
                newImg = self.img_crop(img, edge="h")
                k_h = 2
                break

        # determine lowest denominator in image width
        k_w = 2
        while ( (img_w % k_w) != 0 ):
            k_w += 1
            if (k_w > img_w/2):
                print("Error: the image width is a prime number. Cannot determine pooling kernel size.")

                # function to remove one pixel layer off the image "width"
                newImg = self.img_crop(img, edge = "w")
                k_w = 2
                break

        #print("Newish shape=", newImg.shape)
        return k_h, k_w, newImg

    # determine average image size
    def img_mean(self, imgshp):
        return (imgshp[0]+imgshp[1])/2

    # cut off pixel pixel of the image and return it
    def img_crop(self, im, edge = "both"):
        if edge=="h":
            new_image = im[:im.shape[0]-1, :, :]
        elif edge=="w":
            new_image = im[:, :im.shape[1]-1, :]
        else:
            new_image = im[:im.shape[0]-1, :im.shape[1]-1, :]
        return new_image

    def edge_pixel_positions(self, edge_img, border = 0):
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

    # return pxl at perimeter if you go out of bounds
    def protPxl(self, pxl, max_cap):
        # return bounded value
        if pxl < 0:
            return 0
        elif pxl > max_cap:
            return max_cap - 1
        else:
            return pxl

    def edge_kill(self, edg_img, coord, radius, task="border-count"):
        # Initial counter and pre-define useful values
        count = 0
        pixel_out_of_image = 0
        border_size = 2 * radius + 1

        # Run over the square of pixels surrounding "radius"-pixels around coord
        for i in range(0, border_size**2):
            dx = (i % border_size) - radius
            dy = (i // border_size) - radius
            
            x = coord[0] + dx
            y = coord[1] + dy
            
            if task == "wipe-edges":
                # remove all edge pixels --> change to non-edge pixels
                edg_img[x, y] = 0.
                
            elif task == "border-count":
                # Validate sure we are looking at a border pixel
                if np.abs(dx)==radius or np.abs(dy)==radius :
                    try:
                        if edg_img[x, y] == 0.:
                            # Found a non-edge pixel on the border
                            count += 1
                        else:
                            # Found an edge pixel on the border (return the original edge image)
                            break

                    except IndexError:
                        # The central edge pixel is too close to the image perimeter
                        pixel_out_of_image += 1

            else:
                print("Task is not specified. Stopping the code...")
                exit()

        # check if the border is all non-edge
        if count == (8 * radius) - pixel_out_of_image:
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
            edg_img = edge_kill(edg_img, coord, radius = (radius - 1), task = "wipe-edges")

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

    def edge_killer(self, edge_image, pixel_tolerance=1):
        """
        Iterate through pixels distance from chosen edge pixel, going up to the specified pixel_tolerance value.
        """
        for r in range(1, pixel_tolerance):
            # Iterate through edge pixels (updated edge_pos vector)
            # TODO: can improve by returning coordinates to kill, instead of the whole edge image
            edge_coordinates = edge_pixel_positions(edge_image)
            number_of_edge_pixels = edge_coordinates.shape[0]

            for k in range(0, number_of_edge_pixels):
                edge_image = edge_kill(edge_image, edge_coordinates[k], radius=r)

        return edge_image

    # determine positions of edge vectors, return (? x 2) array.
    # edge_bias details how many edge pixels must be adjacent in ...
    # ... the same direction before considering it an extendable edge
    def edge_filler(self, edge_img, edge_bias = 10):
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
    def pxlThickness(self, edgy_img, pxl):
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
    def unit_vec_dot(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / ( np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)) )

    # select a random direction vector and return the path traversed
    def random_step_direction(self, edgy_img, new_pxl, start_line, prev_path_vec, adj_size, step_dist = 1, debug = False):
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
    def randomPathEdgeRace(self, edgy_img, adj_size = 1, border = 0, showPath = False):
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
    def pxlLen(self, edgy_img, init_pxl, pxl_dir, edge):
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
    def edge_show(self, edgy_images, objEdgeArray):
        # generate edge_path image
        edge_path_img = np.zeros(edgy_images.shape)
        edge_path_img[objEdgeArray.T[0], objEdgeArray.T[1]] = 2.5
        # plt.figure()
        # plt.imshow(edge_path_img)

        plt.figure()
        plt.imshow(edgy_images + edge_path_img)
        plt.show()

    def __within_radius(self, x, y, chosen_one, R=10):
        return math.sqrt((x - chosen_one[0]) ** 2 + (y - chosen_one[1]) ** 2) <= R
