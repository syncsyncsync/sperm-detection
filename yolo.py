from distutils.command.build_ext import build_ext
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import os
import cv2

# import file names list form 11.txt 
def import_label_list(path,label_path):
    label_list = []
    with open(path) as f:
        for line in f:
            # remove directory path and only keep file name
            filename=line.split('/')[-1].strip()
            
            # add label_path to filename
            filename = label_path + '/' + filename
            
            label_list.append(filename)

    return label_list


def load_mp4(path):
    cap = cv2.VideoCapture(path)
    file_list = import_label_list("archive/VISEM_Tracking_Train_v4/Train/11/11.txt", 'archive/VISEM_Tracking_Train_v4/Train/11/labels/')
    
    frames = []
    frames_only = []
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        print(ret)
        label_file = file_list[count]
        #detected_frame =detection_frame(frame);
        cv2.waitKey(200)
        
        if count == 0:
           frame, frame_only = plot_frame_with_label(frame, label_file)
        else:
           frame, frame_only = plot_frame_with_label(frame, label_file, previou_frame = frame_only)
        count = count + 1
        
        frames.append(frame)
        frames_only.append(frame_only)
        
        
        
    cap.release()
    count == count + 1
    return frames

#convert (cx,cy,w,h) to (top left x, top left y, bottom right x, bottom right y)
def convert_cxcywh_to_xywh(cx,cy,w,h,size_flame_w,size_flame_h):
    tl_x = int((cx - w/2)*size_flame_w)
    tl_y = int((cy - h/2)*size_flame_h)
    br_x = int((cx + w/2)*size_flame_w)
    br_y = int((cy + w/2)*size_flame_h)
    
    return tl_x,tl_y,br_x,br_y
     

# imort lable list and plot bounding box on the frame
def plot_frame_with_label(frame, label_file, previou_frame = None):
    # read label file
    label = pd.read_csv(label_file, header=None, sep=' ')
    
    # size of frame 
    size_flame_w = frame.shape[1]
    size_flame_h = frame.shape[0]
    
    frame_only = np.zeros((size_flame_h,size_flame_w), np.uint8)
    if previou_frame is None:
        previou_frame = frame_only
    
    # plot bounding box
    for i in range(label.shape[0]):
        tl_x, tl_y, br_x, br_y = convert_cxcywh_to_xywh(label.iloc[i,1],label.iloc[i,2],label.iloc[i,3],label.iloc[i,4],size_flame_w,size_flame_h)
        cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (0,255,0), 2)
        cv2.rectangle(frame_only, (tl_x, tl_y), (br_x, br_y), (255,255,255), -1)
    # binary image of flame only rectangle is white and background is black
    
    # remove previous frame from current frame
    #frame_only = np.zeros((size_flame_h,size_flame_w), np.uint8) - previou_frame
    frame_diff = frame_only - previou_frame
    # plot rectangle
    cv2.imshow('frame',frame)
    cv2.imshow('frame_only',frame_only)
    cv2.imshow('frame_diff',frame_diff)
    cv2.waitKey(100)    
    return frame, frame_only
    
# cv2 detection using opencv tracking algorithm 
def detection_frame(frame):

    # show frame
    
    return frame

if __name__ == "__main__":
    
    frames = load_mp4("archive/VISEM_Tracking_Train_v4/Train/11/11.mp4")
    print(len(frames))