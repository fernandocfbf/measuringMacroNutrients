import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import matplotlib.image as mpimg
from skimage.transform import resize
import math
import skimage.measure
import cv2
from tqdm import tqdm

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

def le_imagens(path, data, labels, vector):
    with os.scandir(path) as entries:
        for entry in tqdm(entries):
            diretorio = path + entry.name 
            image = cv2.imread(diretorio) 
            image = cv2.resize(image, (64, 64))
            
            data.append(image)
            labels.append(vector)
            
    return data, labels
    
    
def load_data():
    
    path = os.getcwd() #pega a pasta atual
    parent_path = os.path.abspath(os.path.join(path, os.pardir)) #pega a pasta parente
    np.random.seed(1)    

    data = [] #variável que armarzenará todos as imagens de pizza
    labels = [] #variável que armarzenará todos as imagens de não pizza
    
    data, labels = le_imagens(parent_path+'/data/non_images/', data, labels, [1,0,0,0,0,0,0])
    data, labels = le_imagens(parent_path+'/data/pizza/', data, labels, [0,1,0,0,0,0,0])
    data, labels = le_imagens(parent_path+'/data/hot_dog/', data, labels, [0,0,1,0,0,0,0])
    data, labels = le_imagens(parent_path+'/data/rice/', data, labels, [0,0,0,1,0,0,0])
    data, labels = le_imagens(parent_path+'/data/sushi/', data, labels, [0,0,0,0,1,0,0])
    data, labels = le_imagens(parent_path+'/data/donuts/', data, labels, [0,0,0,0,0,1,0])
    data, labels = le_imagens(parent_path+'/data/french_fries/', data, labels, [0,0,0,0,0,0,1])
    return np.array(data, dtype="float")/255.0, np.array(labels)