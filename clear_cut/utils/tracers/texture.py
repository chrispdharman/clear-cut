import os

from clear_cut.utils.tracers.base import BaseTracer


class TextureTracer(BaseTracer):

    def trace_objects_in_image(self, image=None, results_path=None, model_no=None):
        
        # object tracing texture method
        # create results/texture directory if it doesn't yet exist
        if not os.path.isdir(results_path):
            os.mkdir(results_path)

        timeIt = TimeCluster
        # gradImage: create numpy 2D array of size (2n-1) of the original
        dimY, dimX, chanls = origImage.shape

        # append an image (in x-direction) for each of the separate channels
        textureImage = np.zeros(shape=(dimY, dimX, 2), dtype = float)
        #textureImage = np.zeros(shape=(dimY, dimX, 3), dtype=float)

        # plot the red pixel pixel value versus the (r-g) % difference and (r-b) % difference
        # plot3D pre-setup
        #fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        if timeIt:
            t_prev = time.time()
        # loop over each dimension, populating the textureImage with various labels
        remaining_pxls = []
        for j in range(0, dimY):
            #print("j/dimY =",j,"/",dimY)
            for i in range(0, dimX):

                # get difference between red and an `other' colour channel
                pt = (origImage[j, i])
                pt = [int(pt[0]), int(pt[1]), int(pt[2])]
                textureImage[j, i][0] = (pt[0] - pt[1])
                textureImage[j, i][1] = (pt[0] - pt[2])
                #textureImage[j, i][2] = pt[0]

                #print("textureImage[",j,",",i,"]=",textureImage[j,i])
                # get a coordinate list of unclassified pixel coordinate
                remaining_pxls.append([(pt[0] - pt[1]), (pt[0] - pt[2])])

                # plot3D layer creation
                #new_pt = textureImage[j, i]
                #ax.scatter(new_pt[0], new_pt[1], new_pt[2])

        if timeIt:
            print("\t \t 1. Populate textureImage: ", time.time() - t_prev, "seconds")
            t_prev = time.time()

        # plot the (r-g) % difference and (r-b) % difference
        plt.figure()
        plt.scatter(textureImage[:,:,0], textureImage[:,:,1], s=1)
        plt.xlabel("r-g")
        plt.ylabel("r-b")
        plt.show()

        # create new model using clustering algorithm
        model = ""
        # use existing model
        model = "10004078"

        # check if model path exists or not, then act accordingly
        if not os.path.isdir(results_path + "/model_" + str(model)):
            # create results/texture/model_{timestamp} directory
            model_path = results_path + "/model_" + str(model_no)
            os.mkdir(model_path)

            # keep track off original remaining pixels
            orig_remaining_pxls = remaining_pxls.copy()

            # classify clustered regions
            #print("remaining_pxls=", remaining_pxls)
            cluster_list = {
                "label_0" : []
            }
            #rad = 6
            rad = 5
            lbl_no = 0
            clr_map = plt.get_cmap('tab10')
            print("No. of remaining pxls (start) = ", len(remaining_pxls))
            print("cluster_list (start) = ", cluster_list)
            # keep finding clusters until all pxls have been labelled
            while len(orig_remaining_pxls) > 0:
                alive_direction = []
                dead_direction = []
                lbl_no += 1
                print("Cluster [", lbl_no,"] determination in progress...")

                # check the points in cluster label 1 have been removed
                '''if lbl_no == 2:
                    plt.figure()
                    plt.scatter((np.array(orig_remaining_pxls).T)[0], (np.array(orig_remaining_pxls).T)[1], s=1)
                    plt.show()'''

                if timeIt:
                    t_prev = time.time()
                # setup graph
                fig, ax = plt.subplots(1)
                ax.set_aspect('equal')
                #ax.scatter(textureImage[:, :, 0], textureImage[:, :, 1], s=1)
                ax.scatter((np.array(orig_remaining_pxls).T)[0], (np.array(orig_remaining_pxls).T)[1], s=1)
                if timeIt:
                    print("\t \t 2. Configure graph: ", time.time() - t_prev, "seconds")
                    t_prev = time.time()

                # put a criteria here to set all_outer_dead = True if all outer bubbles are dead...
                # edgy_img = np.zeros((255, 255))
                '''brdr = 2 * rad
                edgy_img = np.zeros((255 + 2 * brdr, 255 + 2 * brdr))
                for u in range(0, edgy_img.shape[0]):
                    for v in range(0, edgy_img.shape[1]):
                        # make a border around the image so that it does not go out of bounds?
                        if u < brdr or v < brdr or u > 255 + brdr or v > 255 + brdr:
                            edgy_img[u, v] = -0.1'''

                # randomly select a pixel coordinate in the existing list
                chosen_one = remaining_pxls[randint(0, len(remaining_pxls)-1)]
                #chosen_one = [255 // 2, 255 // 2]   # test case 1
                #chosen_one = [245, 0]   # test case 2
                #chosen_one = [0, 245]   # test case 3
                #chosen_one = [0, 2]     # test case 4
                #chosen_one = [250, 250]     # test case 5
                #print("chosen_one=",chosen_one)

                # initial nucleation
                #remaining_pxls, cluster_list, alive, dead = clstr_nucleate(chosen_one, rad, lbl_no, orig_remaining_pxls,
                #                                                           cluster_list, border=brdr, iter_max=17, init=True)
                #cluster_list, alive, dead = clstr_nucleate(chosen_one, rad, lbl_no, orig_remaining_pxls,
                #                                                     cluster_list, border=brdr, iter_max=17, init=True)
                cluster_list, alive, dead = clstr_nucleate(chosen_one, rad, lbl_no, orig_remaining_pxls,
                                                        cluster_list, iter_max=17, init=True)
                if timeIt:
                    print("\t \t 3. Cluster nucleate...: ", time.time() - t_prev, "seconds")
                    t_prev = time.time()

                alive_direction += alive
                dead_direction += dead

                # populate dead_directions with value 1 in edgy_img
                #for edge in dead_direction:
                #    edgy_img[edge[0], edge[1]] = 1

                #print("\t alive_direction=", alive_direction)
                #print("\t dead_direction=", dead_direction)

                # reiterate nucleation until all outer bubbles are dead
                #all_outer_dead = False
                # initiate new alive directions with current list
                #while not all_outer_dead:
                # for any outermost bubbles that found new pixels, nucleate another bubble around them
                #for dirs in alive_direction:
                idx = 0
                #while idx < len(alive_direction):
                #print("alive_direction=", alive_direction)
                while idx < len(alive_direction):
                    dirs = alive_direction[idx]
                    #print(" >>> dirs=", dirs)
                    idx += 1
                    #print("\t Iteration: ", idx,"/",len(alive_direction))

                    # re-nucleates on alive circles that are 2*rad distance away from initial point
                    #if abs(dirs[0]-chosen_one[0])==2*rad or abs(dirs[1]-chosen_one[1])==2*rad:

                    # re-nucleates on alive circles that are 2*rad distance away from initial point
                    if not ((dirs[0] == chosen_one[0]) and (dirs[1] == chosen_one[1])):
                        if timeIt:
                            t_prev = time.time()
                        #print("\t I'm outer alive!: ",dirs, "--> (", abs(dirs[0]-chosen_one[0]),",",abs(dirs[1]-chosen_one[1]))
                        cluster_list, alive, dead = clstr_nucleate(dirs, rad, lbl_no, orig_remaining_pxls, cluster_list)
                        if timeIt:
                            print("\t \t 3. Cluster nucleate...: ", time.time() - t_prev, "seconds")
                            t_prev = time.time()

                        # remove entries in alive that are already in alive_direction
                        for item in alive_direction:
                            # delete elements that are in alive_direction
                            # only increment index if the element is not in alive_direction
                            '''for curr in alive:
                                if item[0]==curr[0] and item[1]==curr[1]:
                                    alive.remove(curr)'''
                            if item in alive:
                                alive.remove(item)
                        if timeIt:
                            print("\t \t 4. Remove alive_direction duplicates: ", time.time() - t_prev, "seconds")
                            t_prev = time.time()

                        # make sure alive directions are not overwritten by dead ones!
                        alive_direction += alive

                        # remove duplicates in alive_direction
                        #alive_direction = np.unique(alive_direction, axis=0).tolist()

                        # remove alive directions if found in dead
                        for al in alive_direction:
                            '''for dd in dead:
                                if al[0] == dd[0] and al[1] == dd[1]:
                                    dead.remove(dd)'''
                            if al in dead:
                                dead.remove(al)
                        dead_direction += dead
                        #print("\t alive_direction=", alive_direction)
                        #print("\t dead_direction=", dead_direction)
                        if timeIt:
                            print("\t \t 5. Remove dead directions in alive_direction: ", time.time() - t_prev, "seconds")
                            t_prev = time.time()

                        #print("dead_direction=",dead_direction)
                        #print("No. of dead directions =", len(dead_direction))

                # graphics
                for coor in alive_direction:
                    ax.add_patch(Circle((coor[0], coor[1]), rad, facecolor=(0, 0, 0, 0), edgecolor=clr_map(lbl_no)))
                #for coor in dead_direction:
                #    ax.add_patch(Circle((coor[0], coor[1]), rad, facecolor=(0, 0, 0, 0), edgecolor=clr_map(lbl_no)))

                # plot the (r-g) % difference and (r-b) % difference
                plt.xlabel("r-g")
                plt.ylabel("r-b")
                plt.savefig(model_path+"/cluster_"+str(lbl_no)+".png")
                #plt.show()

                # remove duplicates in alive_direction
                alive_direction = np.unique(alive_direction, axis=0).tolist()

                # get all pixels inside the cluster
                enc_list = []
                for alv in alive_direction:
                    enc_list += cluster_counter(alv, remaining_pxls, R=rad, return_count=True)
                #print("len(enc_list)=",len(enc_list))

                # update cluster_list (this label) with pxls that were just found
                cluster_list["label_" + str(lbl_no)] += enc_list

                # remove any coordinates in remaining_pxls if they are with the cluster (enc_list)
                #t0 = time.time()
                updated_remaining_pxls = []
                for pxl in remaining_pxls:
                    if pxl not in enc_list:
                        updated_remaining_pxls.append(pxl)

                #print("Took ",time.time()-t0," seconds to remove enclosed pxls from remaining_pxls list")

                # repeat cluster finding until remaining_pxls is empty
                orig_remaining_pxls = updated_remaining_pxls
                remaining_pxls = orig_remaining_pxls.copy()
                #orig_remaining_pxls = remaining_pxls.copy()

                #print("No. of remaining pxls (end) = ", len(orig_remaining_pxls))
                #print("cluster_list (end) = ", cluster_list)

            # output final cluster_list dictionary
            print("Labelled all data!")

            # output dictionary to results folder
            write_file = csv.writer(open(model_path + "/cluster_list.txt", 'w'))
            for key, val in cluster_list.items():
                write_file.writerow([key, val])
        # if model directory does exist
        else:
            # read in cluster_list
            with open(results_path + "/model_" + str(model) + "/cluster_list.txt") as dict_file:
                reader = csv.reader(dict_file)
                cluster_list = {rows[0]:(rows[1].split()) for rows in reader}
        print("cluster_list=",cluster_list)

        # plot the (r-g) % difference and (r-b) % difference
        plt.xlabel("r-g")
        plt.ylabel("r-b")
        plt.show()

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
        mrgIm1 = self.merge_channels_of_traced_image(gradImage.T, origImage.shape)
        #plt.figure()
        #plt.imshow(mrgIm1)

        edge_array = self.merge_channels_of_traced_image(
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
        print("Is ",image.shape," = ",edge_array.shape,"?")
        return edge_array
            