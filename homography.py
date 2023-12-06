#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:40:28 2023

@author: elbertyi
"""
import csv
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

points = []

def getClick (event, x,y, flags, params):
  if event == cv2.EVENT_LBUTTONDOWN:
    print(x, ' ', y)
    points.append([x,y])

#img1 = cv2.imread('parking_lot.png')
img1 = cv2.imread('centroid_img_2.jpg')

# Begin Testing
img1 = np.pad(img1,[(150,150),(150,150),(0,0)])

# End Testing

height = img1.shape[0]
width = img1.shape[1]
cv2.imshow('image',img1)


cv2.setMouseCallback('image',getClick)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

p1 = points[0]
p2 = points[1]
p3 = points[2]
p4 = points[3]


width_1 = np.sqrt(np.absolute(p2[0]-p1[0])**2+np.absolute(p2[1]-p1[1])**2)
width_2 = np.sqrt(np.absolute(p3[0]-p4[0])**2+np.absolute(p3[1]-p4[1])**2)
height_1 = np.sqrt(np.absolute(p4[0]-p1[0])**2+np.absolute(p1[1]-p4[1])**2)
height_2 = np.sqrt(np.absolute(p3[0]-p2[0])**2+np.absolute(p2[1]-p3[1])**2)

width_1_int = int(width_1)
width_2_int = int(width_2)
height_1_int = int(height_1)
height_2_int = int(height_2)

new_width = np.maximum(width_1_int,width_2_int)
new_height = np.maximum(height_1_int,height_2_int)

input_points = np.float32([p1,p2,p3,p4])
output = np.float32([[0,0],
                    [new_width,0],
                    [new_width,new_height],
                    [0,new_height]])

transform = cv2.getPerspectiveTransform(input_points,output)
out = cv2.warpPerspective(img1,transform,(new_width,new_height))
out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
# new_img =  np.zeros((5000,5000,3))

# for i in range(0,3):
#     for w in range(0,width+1):
#         for h in range(0,height+1):
#             temp = np.matmul(transform,np.array([w,h,1]))
#             new_w = int(temp[0]/temp[2])
#             print(new_w)
#             new_h = int(temp[1]/temp[2])
#             new_img[new_w,new_h,i] = img1[w,h,i]
            
        
        
plt.imshow(out)

# Get coordinates
data = pd.read_csv('image_centroids2.csv')
data = data.to_numpy()
data = data[:,1:3]

num_ones = data.shape[0]
vec_ones = np.ones((num_ones,1))

homogeneous_coordinates = np.hstack((data,vec_ones))
homogeneous_coordinates = np.transpose(homogeneous_coordinates)

transformed_coordinates = np.matmul(np.linalg.inv(transform),homogeneous_coordinates)
intermediate = transformed_coordinates

for i in range(0,num_ones):
    intermediate[0,i] = transformed_coordinates[0,i]/transformed_coordinates[2,i]
    intermediate[1,i] = transformed_coordinates[1,i]/transformed_coordinates[2,i]
    

    
intermediate = intermediate[0:2,:]
intermediate = np.transpose(intermediate)

for_eric = pd.DataFrame(intermediate,columns = ['x-pixel','y-pixel'])
for_eric.to_csv('/Users/elbertyi/Documents/for_eric.csv')
    