import numpy as np
import os
from skimage.transform import resize
import cv2
from tqdm import tqdm


def le_imagens(path, data, labels, vector, size):
    
    with os.scandir(path) as entries:
        for entry in tqdm(entries):
            diretorio = path + entry.name 
            image = cv2.imread(diretorio) 
            image = cv2.resize(image, (size, size))
            
            data.append(image)
            labels.append(vector)
            
    return data, labels
    
    
def load_data(size=64, augmentation=0):
    
    path = os.getcwd() #pega a pasta atual
    parent_path = os.path.abspath(os.path.join(path, os.pardir)) #pega a pasta parente
    np.random.seed(1)    

    data = [] #variável que armarzenará todos as imagens de pizza
    labels = [] #variável que armarzenará todos as imagens de não pizza
    
    data, labels = le_imagens(parent_path+'/data/pizza/', data, labels, [1,0,0,0,0,0], size)
    data, labels = le_imagens(parent_path+'/data/hot_dog/', data, labels, [0,1,0,0,0,0], size)
    data, labels = le_imagens(parent_path+'/data/rice/', data, labels, [0,0,1,0,0,0], size)
    data, labels = le_imagens(parent_path+'/data/sushi/', data, labels, [0,0,0,1,0,0], size)
    data, labels = le_imagens(parent_path+'/data/donuts/', data, labels, [0,0,0,0,1,0], size)
    data, labels = le_imagens(parent_path+'/data/french_fries/', data, labels, [0,0,0,0,0,1], size)
    
    return np.array(data, dtype="float")/255.0, np.array(labels)