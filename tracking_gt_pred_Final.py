
# Title:        Basic Tracking module
# Author:       Max Ronecker
# Description:  A tracking module using the YOLO network generated output and 
#               simple features to generate tracks of multiple targets

#### Initialisation

# Load packages
import csv
import cv2 #used for image processing
import numpy as np # used for math operations
import matplotlib as mpl # used for generating plots
import pandas as pd # 
from matplotlib import pyplot as plt
import glob
import time
import os
import track_lib as tl
import math
from mpl_toolkits.mplot3d import Axes3D
# Define Variables 
track_length = 10; #how many trackpoints are stored
path_number = 5;
delete_single = 0; #determine if single detection tracks will be deleted
def main():
    
    img_path = 'img/*.png' #define in which the image sequence is stored
   
    
    tracking_path = 'tracking_result/' # path in which the tracking results are stored 

    gt_path_all = 'GT/*.txt' # path in which the data about the ground truth is stored
    
    
    
    video_folder_path = 'videos' # path in which a video sequence of the tracking is stored
  
    test_gt_folder_path = 'videos/pics/GT/'# path in which the signal imagess of the video sequence are stored
    
   
    

    gt_names = tl.read_multiple_files(gt_path_all) #reads all the names of the gt data files
    gt_names.sort(key=tl.tokenize) # sort them in the right order 1 to n
    print('Start tracking module')
  
    # Read all images in the folder
    img_names = tl.read_images(img_path)
    
    # Create Video object 
    video = tl.create_video_obj(img_names,'test_gt',video_folder_path)
    
    paths = [] # list with all tracks
    
    path_temp = [] # path with current tracks up to the last 10 elements
 
    ## Work with Groundtruth###############################################
    #######################################################################
    
    counter = 0
    
    for imgname in img_names[0:153]:
        ###### READ IMAGE  ######   
        img = cv2.imread(imgname)   # Read image from file into mat
        img_copy = img.copy()       # Make copy to modify image  
        
        
##       ###### MERGE ######     
        gt_data = tl.merge_gt(gt_names,len(gt_names))   # Merge Groundtruth data
        
        ## Create variables for coordinates
        x_v = []
        y_v = []
        w_v = []
        h_v = []
        
        
        ###### Initial step #######
        if counter == 0:
            ##### Init with number of paths
            total = 0
            detections = 0
            
            while total < path_number:
                #print('Total number of assignments: ',total)
                #print('Total number of detections checked: ',detections)
                x, y ,w, h = tl.get_groundtruth_data(gt_data[detections],counter)
                if w > 0:
                    x_v.append(x)
                    y_v.append(y)
                    w_v.append(w)
                    h_v.append(h)
                    paths.append([[counter,x,y,w,h]])
                    path_temp.append([[counter,x,y,w,h]])
                    total += 1
                    detections += 1
                else:
                    detections += 1
                if detections == (len(gt_data)-1):
                    total = path_number
            for n in range(0,len(x_v)):
                tl.draw_box(img_copy,x_v[n],y_v[n],w_v[n],h_v[n],(250,0,0))
##      
        ##### First Assignment            
        if counter == 1:
            num_of_det = tl.gt_count_det(gt_data,counter)
            x_v = []
            y_v = []
            w_v = []
            h_v = []
            for n in range(0,len(gt_data)):
                x, y, w, h = tl.get_groundtruth_data(gt_data[n],counter)
                if w > 0:
                    x_v.append(x)
                    y_v.append(y)
                    w_v.append(w)
                    h_v.append(h)
            
            ## Compute delta
            delta = tl.compute_delta(path_temp,x_v,y_v,num_of_det)
            
            ## Assign paths
            paths, path_temp, img_copy, delta = tl.gt_assignment(delta,paths,path_temp,counter,img_copy,x_v,y_v,w_v,h_v)
            
            ## Case less detections than paths:
            ## --> Delete paths which didn't get an assignment
            if num_of_det < len(paths):
            
                for t in range(0,len(paths)):
                    if(len(paths[t])<2):
                        del paths[t]
                        del path_temp[t]
            
            ## Case more detection than paths:
            ## --> Create new track starting points
            if num_of_det > len(paths):
               
                for t in range (0,num_of_det):
                    used_bit = 0
                    for ele in delta[t]:
                        if ele == 600:
                            used_bit = 1
                    if used_bit == 0:
                        paths.append([[counter,x_v[t],y_v[t],w_v[t],h_v[t]]])
                        path_temp.append([[counter,x_v[t],y_v[t],w_v[t],h_v[t]]])
                        
        #### Generalized step:
        #### It is assumed that there is at least one track with more 
        #### than one element which enables the system to make predictions
        #### This step part is the generalized prediction and assignment method
        #### used for most of the program.
        if counter > 1:
            num_of_det = tl.gt_count_det(gt_data,counter)
            x_v = []
            y_v = []
            w_v = []
            h_v = []
            
            
            ##### GET DETECTIONS #####
            for n in range(0,len(gt_data)):
                x, y, w, h = tl.get_groundtruth_data(gt_data[n],counter)
                if w > 0:
                    x_v.append(x)
                    y_v.append(y)
                    w_v.append(w)
                    h_v.append(h)
                    
            ##### MAKE PREDICTIONS #####
            x_pred = []
            y_pred = []
            
            ## Store predictions in vector:
            ## For every track with more than one element a prediction is
            ## calculated using a simple motion model. In case the track only has
            ## one element the last detection assigned to the track is used.
            for n in range (0,len(path_temp)):
                if len(path_temp[n])>1:
                    ## Motion Model 
                    x_pred.append(paths[n][0][1] + (paths[n][0][1]-paths[n][1][1]))
                    y_pred.append(paths[n][0][2] + (paths[n][0][2]-paths[n][1][2]))
                else:
                    ## Used most recent detection
                    x_pred.append(paths[n][0][1])
                    y_pred.append(paths[n][0][2])
            ##### Compute Delta
            ##### A matrix containing the distance of every prediction/last element
            ##### to every detection. Therefore a matrix with the following structure
            ##### is created:
            #####
            #####              Track 1 | Track 2 | Track 3 | ... | Track n |
            ##### Detection 1|Delta 11 |Delta 12 |Delta 13 | ....|Delta 1n |
            ##### Detection 2|Delta 21 |Delta 22 |Delta 23 | ....|Delta 2n |
            ##### Detection 3|Delta 31 |Delta 32 |Delta 33 | ....|Delta 3n |
            #####     ...    |   ...   |   ...   |   ...   | ....|   ...   |
            ##### Detection m|Delta m1 |Delta m2 |Delta m3 | ....|Delta mn |
            #####
            delta = tl.compute_delta_pred(x_pred,y_pred,x_v,y_v,num_of_det)
            
            ##### Compute longest element
            longest_length = tl.longest(path_temp)
            
            ##### Assign detections to the path
            ##### Assign detections based on the smallest delta.
            ##### In case more detections than tracks are given the remaining
            ##### detections will serve as new track starting points. In case
            ##### there are less detectiosn than tracks the made predictions will
            ##### be assigned.
            paths, path_temp, img_copy, delta = tl.gt_assignment_pred(delta,paths,path_temp,counter,img_copy,x_v,y_v,w_v,h_v,x_pred,y_pred)
            
            
            ##### Delete inaccurate tracks
            ##### Tracks with more than 5 predictions will be deleted
            
            max_predictions = 5
            index_list = []
            for i, single_path in enumerate(path_temp):
                if single_path[0][0] < (counter-max_predictions):
                    index_list.append(i)
            
            for i in sorted(index_list, reverse=True):
                del paths[i]
                del path_temp[i]
            
        
        #### Counter used to access the correct GT data
        counter += 1
        
        #### Draw line ##############################
        img_copy = tl.draw_gt_path(path_temp,img_copy)
        #############################################
        #### Save modified images and add image to video   
        cv2.imshow('test',img_copy) #Display image
        cv2.waitKey(100) # Wait time in ms
        cv2.imwrite(test_gt_folder_path + 'test_gt' +str(counter)+'.jpg',img_copy)
        video.write(img_copy) #write frame to videofile
        
    ####### PLOT DATA
    mpl.rcParams['legend.fontsize'] = 10

    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    
    oneshot = 0
    for ele in paths:
        
        for z,x,y,w,h in ele:
            if oneshot == 0:
                xv = np.array([x],np.int32)
                yv = np.array([y],np.int32)
                zv = np.array([z],np.int32)
                oneshot = 1
            else:
                xv = np.append(np.int32(x),xv)
                yv = np.append(np.int32(y),yv)
                zv = np.append(np.int32(z),zv)
        ax1.plot(xv, yv,zv, label='Paths')
        oneshot = 0
    plt.show()
   

    ###### Write to CSV #######
    delete_counter = 0
    index_list = []
    ## Remove paths with only one element ##
    ## Uncomment/Comment One Element is single detection
    if(delete_single == 1):
        for i, single_path in enumerate(paths):
            if len(single_path) == 1:
                index_list.append(i)
            
        for i in sorted(index_list, reverse=True):
            del paths[i]
            delete_counter += 1
    print(delete_counter)

    tl.write_to_csv(paths,tracking_path)
   
    ###### Close Video
    cv2.destroyAllWindows()
    video.release()

    


# Used in case is run standalone
    
if __name__ == '__main__':
    
    main()


