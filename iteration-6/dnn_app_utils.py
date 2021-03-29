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

            
def le_imagens(path, lista, classe, aug_path):
    with os.scandir(path) as entries:
        for entry in tqdm(entries):
            diretorio = path + entry.name #pega o diretório das fotos
            imagem = mpimg.imread(diretorio) #le a imagem atual
            img_resized = resize(imagem, (64,64)) #redimensiona a imagem para ser 64 x 64
            matriz = np.array(img_resized) #cria um array com cada imagem (64 x 64 x 3) 
            lista.append([classe,matriz]) #adiciona a matriz na lista "data"
    return lista
    
    
def load_data():
    
    path = os.getcwd() #pega a pasta atual
    parent_path = os.path.abspath(os.path.join(path, os.pardir)) #pega a pasta parente
    np.random.seed(1)    

    
    pizza_data = [] #variável que armarzenará todos as imagens de pizza
    non_pizza_data = [] #variável que armarzenará todos as imagens de não pizza
    
    pizza_data = le_imagens(parent_path+'/data/pizza/', pizza_data, 1, parent_path+'/data/pizza_aug/')
    non_pizza_data = le_imagens(parent_path+'/data/non-pizza/', non_pizza_data, 0, parent_path+'/data/non-pizza_aug/')
        
    pizza_concat = pizza_data
    non_pizza_concat = non_pizza_data
              
    pizza_imgs = np.array(pizza_concat, dtype=object) #cria uma matriz com todos os train_examples
    non_pizza_imgs = np.array(non_pizza_concat, dtype=object)
    
    all_images = []

    for img in non_pizza_imgs:
        all_images.append(img)

    for img in pizza_imgs:
        all_images.append(img)
        
    np.random.shuffle(all_images)
    
    X = [] #armazena as imagens
    Y = [] #armazena  1 ou 0
    
    for par in all_images:
        X.append(par[1]) #adiciona a imagem
        Y.append(par[0]) #adiciona 1 ou 0
    
    return X, Y