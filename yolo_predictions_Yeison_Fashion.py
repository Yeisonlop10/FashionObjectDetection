#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import os
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 
from yaml.loader import SafeLoader


class YOLO_Pred():
    def __init__(self,onnx_model,data_yaml):
        # load YAML
        with open(data_yaml,mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        # load YOLO model
        self.yolo = YOLO(onnx_model)
        # self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    
    def predictions(self, image):

        results = self.yolo.predict(image, conf = 0.3)

        # img = cv2.imread(image)

        for result in results:
            for box in result.boxes:
                colors = self.generate_colors(int(box.cls[0]))
                cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), colors, 2)
                cv2.putText(image, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, colors, 1)
        return image  
    
    def generate_colors(self,ID):
        np.random.seed(10)
        colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])
        
        
    
    
    



