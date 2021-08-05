import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import os
import cv2

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


imagetype= os.listdir("flowers\\")
print(imagetype)

path="flowers//"

image_size=60
images=[]
labels=[]

for i in imagetype:
    flower_path=path+str(i)
    data_list=[j for j in os.listdir(flower_path) if j.endswith(".jpg")]

    for data in data_list:
        image=cv2.imread(flower_path + '/'+ data)
        image=cv2.resize(image,(image_size,image_size))
        images.append(image)
        labels.append(i)


images=np.array(images)
images=images.astype('float32')/255
print(images)