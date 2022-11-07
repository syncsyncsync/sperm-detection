# -*- coding: utf-8 -*-

"""
Author: syncsyncsync
Date: 2022/11/05
Description: Tracking object 
"""     



import os
import cv2
import torch
import logging
from operator import index
from dataclasses import dataclass

import numpy as np 
import pandas as pd 

from abc import ABC, abstractmethod

#from distutils.command.build_ext import build_ext

from sperm_data.scripts.pre_process import label_path


ALMOST_ZERO_VALUE = 1e-10

#define rgb color set pattern for each class
def set_color(class_num):
    color = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(0,0,0)]
    return color[class_num]

# import file names list form 11.txt 
def import_label_list(path):
    label_list = []
    with open(path) as f:
        for line in f:
            #remove newline from line
            line = line.rstrip()
            label_list.append(line.rstrip())            
            #open file in the line and read it as list
            temp_labels = np.array(pd.read_csv(line, header=None, sep=' '))
            
            
    return label_list


'''
class Iterator(ABC):
    @abstractmethod
    def is_next_frame(self):
        pass
    @abstractmethod
    def next(self):
        pass
'''


@dataclass                                           
class FrameBasicInfo(object):
    '''
    Basic information of video input
    '''
    
    # for file input 
    video_path: str
    label_path: str
    size_frame_h: int
    size_frame_w: int
    total_frames: int
    
    # for camera input
    camera_id: int
    fps: int
    
@dataclass                                               
class Frame():
    '''
    A basic object defined for each frames 
    '''
    img: cv2.Mat
    label: np.ndarray
    previous_index: int
    next_index: int
    index: int
    status : bool
    label_file_name: str
    img_file_name: str
    
class LabelClass(object):
    '''
        description: 
        input: get label list file path as input, 
        return: labels as numpy array which is read from of label files listed in the file.
    '''
    
    def __init__(self, label_path):
        self.label_list = []
        self.count = -1
        with open(label_path) as f:
            for line in f:
                #remove newline from line
                line = line.rstrip()
                self.label_list.append(line.rstrip())        
    
    def __iter__(self): 
        self.count = -1
        return self
        
    def __len__(self):
        return self.label_list.__len__()
    
    def __next__(self):
        self.count += 1
        
        if self.count < self.label_list.__len__():
            #open file in the line and read it as list
            temp_labels = np.array(pd.read_csv(self.label_list[self.count], header=None, sep=' '))
            return temp_labels
        else:
            raise StopIteration
             
    def get_label_path(self):
        return self.label_list[self.count]
        
@dataclass
class DetectedResult(FrameBasicInfo):
    class_num: int
    cx: int
    cy: int
    w : int
    h : int
    
    def cxcywh(self):
        _cx = int(self.cx * self.size_frame_w)
        _cy = int(self.cy *  self.size_frame_h)
        _w = int(self.w  *  self.size_frame_w)
        _h = int(self.h  *  self.size_frame_h)
        return {'cx':_cx, 'cy':_cy, 'w':_w, 'h':_h}
    
    def to_xywh_normarized(self):
        _tl_x = int((self.cx - self.w/2))
        _tl_y = int((self.cy - self.h/2))
        _br_x = int((self.cx + self.w/2))
        _br_y = int((self.cy + self.w/2))
        return {'_tl_x':_tl_x, '_tl_y':_tl_y, '_br_x':_br_x, '_br_y':_br_y}
    
    def to_xywh(self, size_frame_w, size_frame_h):
        tl_x = int((self.cx - self.w/2) * size_frame_w)
        tl_y = int((self.cy - self.h/2) * size_frame_h)
        br_x = int((self.cx + self.w/2) * size_frame_w)
        br_y = int((self.cy + self.w/2) * size_frame_h)
        return {'tl_x':tl_x, 'tl_y':tl_y, 'br_x':br_x, 'br_y':br_y}


@dataclass                                           
class FrameDetectionResult(FrameBasicInfo):
    '''
      Output
    '''
    raw_frame: cv2.Mat
    result_frame: cv2.Mat
    result_frame_binary: cv2.Mat
    result_frame_diff: cv2.Mat
    result_detection_bbox: list[DetectedResult]
    result_detection_number: list
        
# count line number of a file, if blank line, skip it
def count_line_number(file_path):
    '''
    for DEBUG
    '''
    with open(file_path) as f:
        line_count = 0        
        for line in f:
            if not line.strip():
                continue
            else:
                line_count += 1
                
        print("total line number: ", line_count)
    return line_count

    
def print_csv_file(file_path):
    '''
    print csv file for DEBUG
    '''
    with open(file_path) as f:
        line_count = 0        
        for line in f:
            # if line is empty, skip it
             
             
            if not line.strip():
                continue
            else:
                line_count += 1
                
            print(line)
        print("total line number: ", line_count)
    return line_count
    

class VideoIterator(Frame,FrameBasicInfo):
    '''
     Extract frame from video
    '''
    def __init__(self, video_path, label_path=None):
        self.frame = Frame(img=None,previous_index=None,next_index=None,index=None,label=[],status=None,label_file_name=None,img_file_name=None)
        self.old_frame = Frame(img=None,previous_index=None,next_index=None,index=None,label=[],status=None,label_file_name=None,img_file_name=None)
        self.video_path = video_path
        self.label_path = label_path
        self.videoCapture = cv2.VideoCapture(video_path)    
        self.frame_index = 0
        self.next_index = 1
        self.size_frame_h = int(self.videoCapture.get(3))
        self.size_frame_w = int(self.videoCapture.get(4))    
        self.total_frames  = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

        if label_path != None:
            self.Labels = LabelClass(label_path)
            
    def is_next_frame(self):
        return self.frame.index < self.total_frames 


    def __update__(self, frame_image):
        self.old_frame = self.frame
        
        self.frame.img = frame_image
            
        self.frame_index = self.frame_index + 1
        self.frame.previous_index = self.frame.index
        self.frame.index = self.frame_index
        self.frame.next_index = self.frame_index + 1  
        

    def __skip__(self):
        # skip frame
        self.Labels.__next__()
        self.frame.status = False
        self.frame_index += 1
        self.next_index += 1
        # skip label
       
        logging.info("skip frame: %d", self.frame_index)
        
        return self.frame_index
    
    def __next__(self):
        _ret, _frame_image = self.videoCapture.read()
        
        if self.label_path != None:
            self.frame.label  = self.Labels.__next__()
            self.frame.label_file_name = self.Labels.get_label_path()    
            if self.frame.label.shape[0]  == 0:
                logging.error("label file is not found")
                raise StopIteration
            
        # check if frame is correctly read
        if not _ret:
            if self.frame.index == self.total_frames:
                raise StopIteration
            else:
                self.__skip__()
                # skip this frame
        else:
            self.__update__(_frame_image)
            
            #self.frame_index = self.frame_index + 1
            #self.frame.index = self.frame_index
            #self.frame.previous_index = self.frame_index - 1
            #self.frame.next_index = self.frame_index + 1    

        # DEBUG 
        logging.debug('frame index: %d', self.frame_index)
        logging.debug('label path ', self.Labels.get_label_path())
        
        print(self.Labels.get_label_path())
        
        #if self.Labels.get_label_path() == 'sample/labels/11_frame_25.txt':
        #  print('here')
            
        debug_count_1 = count_line_number(self.Labels.get_label_path())
        debug_count_2 = self.frame.label.shape[0]
        
        assert debug_count_1 == debug_count_2, "label file is not correct"
        
        return self.frame
    
    def __iter__(self):
        return self    
    
    def plot_frame_with_label(self):
        out_img = self.frame.img.copy()
        frame_binary = np.zeros((self.size_frame_h, self.size_frame_w), dtype=np.uint8)
        frame_diff = np.zeros((self.size_frame_h, self.size_frame_w), dtype=np.uint8)
    
        if self.old_frame is None:
            self.old_frame = frame_diff
        
        # plot bounding box
        active = 0
        non_active = 0
        for i in range(self.frame.label.shape[0]):
            tl_x, tl_y, br_x, br_y = convert_cxcywh_to_xywh(self.frame.label[i,1],self.frame.label[i,2],self.frame.label[i,3], self.frame.label[i,4], self.size_frame_w, self.size_frame_h)
        # add rectangle to frame 
      
        if self.frame.label[i,0] == 0:
            cv2.rectangle(frame_binary, (tl_x, tl_y), (br_x, br_y), (255,255,255), -1)
            # measurement velocity using optical flow
            v_x, v_y = calc_vel(frame_binary, self.old_frame, tl_x, tl_y, br_x, br_y, class_num=0)
            pt2_x = int((tl_x + br_x)/2 + 20*v_x)
            pt2_y = int((tl_y + br_y)/2 + 20*v_y)

            #tail_length =20
            #hit_x = int((tl_x + br_x)/2 - tail_length*v_x)
            #hit_y = int((tl_y + br_y)/2 - tail_length*v_y)

            if abs(v_x) > ALMOST_ZERO_VALUE or abs(v_y) > ALMOST_ZERO_VALUE: 
                cv2.rectangle(out_img, (tl_x, tl_y), (br_x, br_y), set_color(self.frame.label.iloc[i,0]) , 2)
            else:
                cv2.rectangle(out_img, (tl_x, tl_y), (br_x, br_y), (125,0,0) , 2)
                non_active = non_active + 1

            out_img = cv2.arrowedLine(out_img, ( (tl_x+br_x)//2, (tl_y+br_y)//2), (pt2_x , pt2_y),  (0,255,0), 3)
            out_img = cv2.arrowedLine(out_img, ( (tl_x+br_x)//2, (tl_y+br_y)//2), (pt2_x , pt2_y),  (0,255,0), 3)
        
        return out_img
    
    
def read_mp4_v2(video_path,label_path=None,debug=True):
    video_frames = VideoIterator(video_path, label_path)
    
    for video_frame in video_frames:
        print("tst")
        
            # stop at debug mode
        if debug:
            if video_frame.index > 10: 
                break
            else:
                print(video_frame.index)
                
        video_frames.plot_frame_with_label()
                
'''
def load_mp4(path, debug=False):
    # import image
    cap = cv2.VideoCapture(path)
    # get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('input movies Movie fps: ',fps)

    # import label list
    file_list = import_label_list("sample/11.txt")
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

            size_frame_w = frame.shape[1]
            size_frame_h = frame.shape[0]
            
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
def convert_cxcywh_to_xywh(cx,cy,w,h,size_frame_w,size_frame_h):
    tl_x = int((cx - w/2)*size_frame_w)
    tl_y = int((cy - h/2)*size_frame_h)
    br_x = int((cx + w/2)*size_frame_w)
    br_y = int((cy + w/2)*size_frame_h)
    
    return tl_x,tl_y,br_x,br_y
'''


def calc_vel(frame, previous_frame, tl_x, tl_y, br_x, br_y, class_num=0):
    # 1. crop from frame_diff by tl_x, tl_y, br_x, br_y 
    frame_crop = frame.img[tl_y:br_y, tl_x:br_x]
    previous_crop = previous_frame.img[tl_y:br_y, tl_x:br_x]
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
    size_frame_w = frame.shape[1]
    size_frame_h = frame.shape[0]
    
    frame_binary = np.zeros((size_frame_h,size_frame_w), np.uint8)
    frame_diff = np.zeros((size_frame_h,size_frame_w), np.uint8)
    if previou_frame is None:
        previou_frame = frame_diff
    
    # plot bounding box
    active = 0
    non_active = 0
    for i in range(label.shape[0]):
        tl_x, tl_y, br_x, br_y = convert_cxcywh_to_xywh(label.iloc[i,1],label.iloc[i,2],label.iloc[i,3],label.iloc[i,4],size_frame_w,size_frame_h)
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

            if abs(v_x) > ALMOST_ZERO_VALUE or abs(v_y) > ALMOST_ZERO_VALUE: 
                cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), set_color(label.iloc[i,0]) , 2)
                cv2.circle(frame , (hit_x, hit_y), 5, (0,0,250), 2)
                active = active + 1
            else:
                cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (125,0,0) , 2)
                non_active = non_active + 1

            frame = cv2.arrowedLine(frame, ( (tl_x+br_x)//2, (tl_y+br_y)//2), (pt2_x , pt2_y),  (0,255,0), 3)
            
            frame = cv2.arrowedLine(frame, ( (tl_x+br_x)//2, (tl_y+br_y)//2), (pt2_x , pt2_y),  (0,255,0), 3)
    # binary image of frame only rectangle is white and background is black
    
    # remove previous frame from current frame
    #frame_diff = np.zeros((size_frame_h,size_frame_w), np.uint8) - previou_frame
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
    
#
##   Results -- Result of frames  -- detections ---  detection bbox_index =0   class_num, cx, cy, w, h  
##           ---                                ---  detection bbox_index =1  DetectedResult class_num, cx, cy, w, h 

# mockup of dummy detection 
def dummy_detection(frame, label_file, previou_frame = None):
    
    label = pd.read_csv(label_file, header=None, sep=' ')

    return frame



if __name__ == "__main__":
    read_mp4_v2("sample/11.mp4", "sample/11.txt")
    #frames = load_mp4("sample/11.mp4", debug=True)
    #print(len(frames))
    
    
    