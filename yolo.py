

from distutils.command.build_ext import build_ext
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import os
import cv2

#define rgb color set pattern for each class
def set_color(class_num):
    color = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(0,0,0)]
    return color[class_num]


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




def load_mp4(path, debug=False):
    # import image
    cap = cv2.VideoCapture(path)
    # get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('input movies Movie fps: ',fps)

    # import label list
    file_list = import_label_list("sample/11.txt", 'sample/labels/')
    #count = 0
    # loop through all frames
    #while(cap.isOpened()):

    for count, label_file in enumerate(file_list):
        
        # stop at debug mode
        if debug:
            if count > 10:
                break

        # read frame
        ret, frame = cap.read()
        if ret == False:
            break
        
        #init loop  ------------
        if count == 0:

            frames = []
            frames_only = []

            size_flame_w = frame.shape[1]
            size_flame_h = frame.shape[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video_out = cv2.VideoWriter('output.mp4', fourcc, fps, (640,480))
            video_out_diff = cv2.VideoWriter('output_diff.mp4', fourcc, fps, (640,480))

            assert video_out.isOpened()
            assert video_out_diff.isOpened()

            frame, frame_binary, frame_diff = plot_frame_with_label(frame, label_file)            
        else:
            frame, frame_binary, frame_diff = plot_frame_with_label(frame, label_file, previou_frame = frame_binary)
        # ------------------------------------------------------------------------------------------

        count = count + 1
        #frames.append(frame)
        frames_only.append(frame_diff)    
        video_out.write(frame)
        video_out_diff.write(frame_diff)
        
    cap.release()
    return frames



#convert (cx,cy,w,h) to (top left x, top left y, bottom right x, bottom right y)
def convert_cxcywh_to_xywh(cx,cy,w,h,size_flame_w,size_flame_h):
    tl_x = int((cx - w/2)*size_flame_w)
    tl_y = int((cy - h/2)*size_flame_h)
    br_x = int((cx + w/2)*size_flame_w)
    br_y = int((cy + w/2)*size_flame_h)
    
    return tl_x,tl_y,br_x,br_y



def calc_vel(frame, previou_frame, tl_x, tl_y, br_x, br_y, class_num=0):
    # 1. crop from frame_diff by tl_x, tl_y, br_x, br_y 
    frame_crop = frame[tl_y:br_y, tl_x:br_x]
    previous_crop = previou_frame[tl_y:br_y, tl_x:br_x]
    # 2. calculate optical flow
    # 2.1 convert to gray scale

     # check channel if not gray scale, convert to gray scale
    if len(frame_crop.shape) == 3:
        frame_crop_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
    else:
        frame_crop_gray = frame_crop

    # check channel if not gray scale, convert to gray scale
    if len(previous_crop.shape) == 3:
        previous_crop_gray = cv2.cvtColor(previous_crop, cv2.COLOR_BGR2GRAY)
    else:
        previous_crop_gray = previous_crop
    # 2.2 calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(previous_crop_gray,frame_crop_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 2.3 calculate velocity
    vel = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    vel = np.mean(vel)
    print(vel, 'pix/s', 'vel_x:',flow[...,0].mean(), 'vel_y:',flow[...,1].mean())

    return [flow[...,0].mean(),flow[...,1].mean()]



# imort lable list and plot bounding box on the frame
def plot_frame_with_label(frame, label_file, previou_frame = None):
    # read label file
    label = pd.read_csv(label_file, header=None, sep=' ')
    
    # size of frame 
    size_flame_w = frame.shape[1]
    size_flame_h = frame.shape[0]
    
    frame_binary = np.zeros((size_flame_h,size_flame_w), np.uint8)
    frame_diff = np.zeros((size_flame_h,size_flame_w), np.uint8)
    if previou_frame is None:
        previou_frame = frame_diff
    
    # plot bounding box
    active = 0
    non_active = 0
    for i in range(label.shape[0]):
        tl_x, tl_y, br_x, br_y = convert_cxcywh_to_xywh(label.iloc[i,1],label.iloc[i,2],label.iloc[i,3],label.iloc[i,4],size_flame_w,size_flame_h)
        # add rectangle to frame 
        
        if label.iloc[i,0] == 0:
            cv2.rectangle(frame_binary, (tl_x, tl_y), (br_x, br_y), (255,255,255), -1)
            # measurement velocity using optical flow
            v_x, v_y = calc_vel(frame_binary, previou_frame, tl_x, tl_y, br_x, br_y, class_num=0)
            pt2_x = int((tl_x + br_x)/2 + 20*v_x)
            pt2_y = int((tl_y + br_y)/2 + 20*v_y)

            tail_length =20
            hit_x = int((tl_x + br_x)/2 - tail_length*v_x)
            hit_y = int((tl_y + br_y)/2 - tail_length*v_y)

            if abs(v_x) > 1e-10 or abs(v_y) > 1e-10:
                cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), set_color(label.iloc[i,0]) , 2)
                cv2.circle(frame , (hit_x, hit_y), 5, (0,0,250), 2)
                active = active + 1
            else:
                cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (125,0,0) , 2)
                non_active = non_active + 1

            frame = cv2.arrowedLine(frame, ( (tl_x+br_x)//2, (tl_y+br_y)//2), (pt2_x , pt2_y),  (0,255,0), 3)
            
            frame = cv2.arrowedLine(frame, ( (tl_x+br_x)//2, (tl_y+br_y)//2), (pt2_x , pt2_y),  (0,255,0), 3)
    # binary image of flame only rectangle is white and background is black
    
    # remove previous frame from current frame
    #frame_diff = np.zeros((size_flame_h,size_flame_w), np.uint8) - previou_frame
    if previou_frame is None:
        frame_diff = frame_binary
    else:
        frame_diff = frame_binary - previou_frame

    
    cv2.putText(frame, 'active: '+str(active)+ '  non: '+str(non_active), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)    
    # plot rectangle
    cv2.imshow('frame',frame)
    cv2.imshow('frame_binary',frame_binary)
    cv2.imshow('frame_diff',frame_diff)
    cv2.waitKey(100)    
    # show active and non active in the frame by text
    
    return frame, frame_binary, frame_diff
    
# cv2 detection using opencv tracking algorithm 
def detection_frame(frame):

    # show frame
    
    return frame


if __name__ == "__main__":
    
    frames = load_mp4("sample/11.mp4", debug=False)
    print(len(frames))