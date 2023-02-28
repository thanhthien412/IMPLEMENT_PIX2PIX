
from binascii import Error
import tensorflow as tf
from os import listdir
from numpy import asarray
from matplotlib import pyplot
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
import os
from tqdm import tqdm
from pix2pix_model import define_discriminator,define_gan,define_generator,train
from numba import cuda
print("Put the initial weight")
batch=int(input())
print('=================')
print("file")
file=input()
print('=================')
if(batch<0):
    raise NotImplementedError
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
device = cuda.get_current_device()
print(device)
device.reset()
def load_data(path, size=(256,512)):
    train_src_list, train_tar_list = list(), list()
    val_src_list, val_tar_list = list(), list()
    _,train,_=listdir(path)
    train=os.path.join(path,'%s'%train)
    for filename in tqdm(listdir(train)):
        pixels = load_img(os.path.join(train,filename), target_size=size)
        pixels = img_to_array(pixels)
        map_img,sat_img=pixels[:,:256],pixels[:,256:]
        train_src_list.append(sat_img)
        train_tar_list.append(map_img)
    """ 
    val=os.path.join(path,'%s'%val)   
    for filename in tqdm(listdir(val)):
        pixels = load_img(os.path.join(val,filename), target_size=size)
        pixels = img_to_array(pixels)
        sat_img,map_img=pixels[:,:256],pixels[:,256:]
        val_src_list.append(sat_img)
        val_tar_list.append(map_img)    
    """   
    return [asarray(train_src_list), asarray(train_tar_list)],[asarray(val_src_list), asarray(val_tar_list)]

def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]



[Train,Targer],_=load_data(file)
train_data=[Train,Targer]
#val_data=[Val,Val_target]
image_shape=(256,256,3)
train_dataset=preprocess_data(train_data)
#val_dataset=preprocess_data(val_data)
g_model=define_generator()
d_model=define_discriminator()
gan_model = define_gan(g_model, d_model, image_shape)
train(d_model,g_model,gan_model,train_dataset,batch)
#g_model.load_weights('weight/')
#predict(g_model,val_dataset)
