#!/usr/bin/env python3

# Title:        Tracking Library
# Author:       Max Ronecker
# Description:  Library with different tracking functions
#


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
import re
import math
digits = re.compile(r'(\d+)')

def create_gt_list(gt_path):
    gt_file = open(gt_path,'r')
    gt_list = gt_file.read().split('\n') # [x,y,w,h]
    
    return gt_list

def read_images(img_path):
    img_names = sorted(glob.glob(img_path))
    
    return img_names

def read_multiple_files(path):
    files = glob.glob(path)
      
    return files


import re


def tokenize(filename):
    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))



def create_video_obj(img_names, title, video_folder_path):
    video_name = os.path.join(video_folder_path, title +'.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Video Format
    img_template = cv2.imread(img_names[1]) # Take Parameters of first image as template
    height , width , layers =  img_template.shape # Get Parameters
    
    video= cv2.VideoWriter(video_name,fourcc,10,(width,height)) #Video Object
    
    print('The video with the title "'+title+'" has been created')
    
    return video

def get_groundtruth_data(gt_list,counter):
    # Get ground truth data (Function)
    gt_temp = gt_list[counter]  # get respective gt data
    gt_img = gt_temp.split(',') # [x,y,w,h]
    x = int(gt_img[0]) #x-position
    y = int(gt_img[1]) #y-position
    w = int(gt_img[2]) #width
    h = int(gt_img[3]) #height
    
    return x,y,w,h

def get_yolo_data(yolo_list):
    # Get yolo output data (Function)
    yolo_output = [[0] * 5 for i in range(len(yolo_list)-1)]
    for c in range(0,len(yolo_list)-1):
        yolo_temp = yolo_list[c]  # get respective gt data
        yolo_data = yolo_temp.split(',') # [class,x,y,w,h]
        classification = str(yolo_data[0]) #class
        x = int(yolo_data[1])
        y = int(yolo_data[2]) #y-position
        w = int(yolo_data[3]) #width
        h = int(yolo_data[4]) #height
        yolo_output[c] = [classification,x,y,w,h]
    
    return yolo_output

def draw_box(img,x,y,w,h,color):
    # Draw rectangke in the img
    cv2.rectangle(img,(x-int(w/2),y-int(h/2)),(x+int(w/2),y+int(h/2)),color,2)
    
    return

def store_path(pts,track_length,x,y):  
    ele = pts.shape
    if ele[0] == track_length:
        
        pts = np.delete(pts,track_length-1,0)
        pts = np.append(np.int32([x,y]),pts)
        pts = pts.reshape((-1,1,2))
        
    else:
        pts = np.append(np.int32([x,y]),pts)
        pts = pts.reshape((-1,1,2))
        
    return pts
    
def colour_selection(classification):
    colour = (0,0,0)
    
    if classification == 'Car':
        colour = (255,0,0)
    elif(classification == 'Pedestrian'):
        colour = (0,255,0)
    elif(classification == 'Cyclist'):
        colour = (0,0,255)
    elif(classification == 'Van'):
        colour == (125,125,0)
    
    
    return colour
    

def gt_assignment(delta,paths,path_temp,counter,img,x,y,w,h):
    
    track_length = 10
    colour = (0,0,200)
    
    for i in range(0,len(paths)):
                
        if i < (delta.shape[0]): # ensure that no detection are used double    
            min_value = np.amin(delta)
            min_row , min_col = np.where(delta == min_value)
            paths[min_col[0]].insert(0,[counter,x[min_row[0]],y[min_row[0]],w[min_row[0]],h[min_row[0]]])
            path_temp[min_col[0]].insert(0,[counter,x[min_row[0]],y[min_row[0]],w[min_row[0]],h[min_row[0]]])
            delta[min_row[0],min_col[0]] = 600 
            draw_box(img,x[min_row[0]],y[min_row[0]],w[min_row[0]],h[min_row[0]],colour)
            ## Limit length to 10 Elements
            if len(path_temp[min_col[0]]) > track_length:
                path_temp[min_col[0]].pop()
    
#    for i in range(0,delta.shape[0]):
#        print('Reached second')
#        if 600 not in delta[i]:
#            paths.append([[counter,x[i],y[i],w[i],h[i]]])
#            path_temp.append([[counter,x[i],y[i],w[i],h[i]]])
#            print('Added new Path')
#        
             
    
    return paths, path_temp, img, delta


def gt_assignment_pred(delta,paths,path_temp,counter,img,x,y,w,h,x_pred,y_pred):
    
    track_length = 10
    colour = (0,0,200)
    max_delta = 20 #Defines the area in which detections are assigned
    for i in range(0,len(paths)):
                
        if i < (delta.shape[0]): # ensure that no detection are used double    
            min_value = np.amin(delta)
            if min_value < max_delta:
                min_row , min_col = np.where(delta == min_value)
                paths[min_col[0]].insert(0,[counter,x[min_row[0]],y[min_row[0]],w[min_row[0]],h[min_row[0]]])
                path_temp[min_col[0]].insert(0,[counter,x[min_row[0]],y[min_row[0]],w[min_row[0]],h[min_row[0]]])
                delta[min_row[0],min_col[0]] = 600 
                draw_box(img,x[min_row[0]],y[min_row[0]],w[min_row[0]],h[min_row[0]],colour)
            ## Limit length to 10 Elements
                if len(path_temp[min_col[0]]) > track_length:
                    path_temp[min_col[0]].pop()
    ##### More detections then path
    if delta.shape[0] > len(path_temp):
         
        for i in range(0,delta.shape[0]):
            #print('Reached second')
            if 600 not in delta[i]:
                paths.append([[counter,x[i],y[i],w[i],h[i]]])
                path_temp.append([[counter,x[i],y[i],w[i],h[i]]])
                draw_box(img,x[i],y[i],w[i],h[i],colour)

                #print('Added new Path')
    # Wrong must be not equal to counter 
    max_length = longest(path_temp)
    if delta.shape[0] < len(path_temp):
        for i, single_path in enumerate(path_temp):
            #if len(single_path) < max_length and len(single_path) > 1:
            if single_path[0][0] != counter and len(single_path) > 1:

                paths.append([[counter,x_pred[i],y_pred[i],single_path[0][3],single_path[0][4]]])
                path_temp.append([[counter,x_pred[i],y_pred[i],single_path[0][3],single_path[0][4]]])
                draw_box(img,x_pred[i],y_pred[i],single_path[0][3],single_path[0][4],(100,0,0))
                #print('Added pred')
#    if delta.shape[0] < len(path_temp):
#        for single_path in path_temp:
#            
             
    
    return paths, path_temp, img, delta

def longest(l):
    longest_length = 0
    for n in range(0,len(l)):
        length = len(l[n])
        if length > longest_length:
            longest_length = length
    
    return longest_length
    

def compute_delta(paths,x,y,num_det):
    delta = np.zeros((num_det,len(paths))) 
                                                        
            # Maybe more efficient methoood                                            
    for i in range(0,num_det): # rows: number of current detections
                
        for n in range(0,len(paths)): # columns = current number of paths
            #print('i:',i, 'n:',n,'image counter:',image_counter)
        
            xd = paths[n][0][1] - x[i]
            yd = paths[n][0][2] - y[i]
            delta[i][n] = math.sqrt(pow(xd,2)+pow(yd,2))
            
    return delta


def compute_delta_pred(x_pred,y_pred,x,y,num_det):
    delta = np.zeros((num_det,len(x_pred))) 
                                                        
            # Maybe more efficient methoood                                            
    for i in range(0,num_det): # rows: number of current detections
                
        for n in range(0,len(x_pred)): # columns = current number of paths
            #print('i:',i, 'n:',n,'image counter:',image_counter)
        
            xd = x_pred[n] - x[i]
            yd = y_pred[n] - y[i]
            delta[i][n] = math.sqrt(pow(xd,2)+pow(yd,2))
            
    return delta




def draw_gt_path(path_temp,img_copy):
    
    colour = (250,0,0)
    miniCount = 0
    for path_ele in path_temp:
            # get current x and y values
        oneshot = 0
        miniCount += 1
        for c,x,y,w,h in path_ele:
            
            if oneshot == 0:
                pts = np.array([x,y],np.int32)
                oneshot = 1
                    
            else:
                pts = np.append(pts,np.int32([x,y]))
                pts = pts.reshape((-1,1,2))
        #print(miniCount)
        #print(pts)
        if(len(pts)>2):        
            cv2.polylines(img_copy,[pts],False,colour,lineType = cv2.LINE_AA)
        if(miniCount == 1):
            colour = (0,250,0)
        elif(miniCount==2):
            colour = (0,0,250)
            
            
    return img_copy



def merge_gt(gt_names,length):
    gt_data=[]
    for gt_name in gt_names[0:length]:
        #print(gt_name)
        temp_list = create_gt_list(gt_name)
        gt_data.append(temp_list)
    return gt_data
    
def write_to_csv(paths,tracking_path):
    for n in range(0,len(paths)):
        with open(tracking_path +"output"+str(n)+".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(paths[n])
    return


def gt_count_det(gt_data,counter):
    num_of_det = 0
    for n in range(0,(len(gt_data)-1)):
        x, y, w, h = get_groundtruth_data(gt_data[n],counter)
        if w > 0:
            num_of_det+=1
    
    return num_of_det