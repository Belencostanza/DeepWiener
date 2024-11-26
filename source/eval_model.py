#author: Belén Costanza


import sys,os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import tensorflow.keras.losses
import time
import losses as loss
from read_input import load_config

############# read-config ##########################

if len(sys.argv) != 2:
    print("Usage: python your_script.py input_file")
    sys.exit(1)

input_file = sys.argv[1]

#Dictionary with the parameters of the input file:
input_data = load_config(input_file)

data_folder = input_data["name_data_folder"]
result_folder = input_data["name_result_folder"]

npad = int(input_data["npad"])
npixels = int(input_data["npixels"])
Lsize = int(input_data["Lsize"])

iteration = int(input_data["iter"])
modelb = int(input_data['modelb'])
mask_mode = int(input_data["mask"])

variance_map = np.load(data_folder + 'variance_QU_planck.npy')
variance_bmap = np.load(data_folder + 'variance_mapB_planck.npy')
masks = np.load(data_folder + 'masks_QU.npy')

if mask_mode == 0:
    print('using mask1')
    mask = masks[0]
else:
    print('using mask2')
    mask = masks[1]


if iteration == 1:
    map_rescale_factor_q = float(input_data["factor1"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model1"]
    name_result = input_data["name_result_cnn1"]
    name_test_noise = input_data["name_test_noise1"]
elif iteration == 2:
    map_rescale_factor_q = float(input_data["factor2"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model2"]
    name_result = input_data["name_result_cnn2"]
    name_test_noise = input_data["name_test_noise2"]
elif iteration == 3:
    map_rescale_factor_q = float(input_data["factor3"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model3"]
    name_result = input_data["name_result_cnn3"]
    name_test_noise = input_data["name_test_noise3"]
elif iteration == 4:
    map_rescale_factor_q = float(input_data["factor4"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model4"]
    name_result = input_data["name_result_cnn4"]
    name_test_noise = input_data["name_test_noise4"]
elif iteration == 5:
    map_rescale_factor_q = float(input_data["factor5"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model5"]
    name_result = input_data["name_result_cnn5"]
    name_test_noise = input_data["name_test_noise5"]

if modelb == 1: 
    name_test_noise = input_data["name_test_noiseb"]
    map_rescale_factor_q = float(input_data["factor3"])
    map_rescale_factor_b = float(input_data["factorb"])
    map_rescale_factor_e = float(input_data["factore"])
    name_model = input_data["name_modelb"]
    name_result = input_data["name_result_cnnb"]

lossj3 = loss.loss_functions(Lsize, npixels, mask, variance_map, variance_bmap, map_rescale_factor_q, map_rescale_factor_b)

if modelb == 1:
    loss_b_j3 = lossj3.loss_b_j3
    tf.keras.losses.custom_loss = loss_b_j3
    model = load_model(result_folder + name_model, custom_objects={'loss_b_j3':loss_b_j3})

elif modelb == 0 and iteration == 1:
    print('using loss_wiener_j3')
    loss_wiener_j3 = lossj3.loss_wiener_j3
    tf.keras.losses.custom_loss = loss_wiener_j3
    model = load_model(result_folder + name_model, custom_objects={'loss_wiener_j3':loss_wiener_j3})

elif modelb == 0 and iteration != 1:
    print('using loss_wiener_j3_iter')
    loss_wiener_j3_iter = lossj3.loss_wiener_j3_iter
    tf.keras.losses.custom_loss = loss_wiener_j3_iter
    model = load_model(result_folder + name_model, custom_objects={'loss_wiener_j3_iter':loss_wiener_j3_iter})


data_test=np.load(data_folder + name_test_noise)
images_test = data_test

def periodic_padding(images,npad):
    if len(images.shape)==4:
        images = np.pad(images,pad_width=((0,0),(npad,npad),(npad,npad),(0,0)),mode='wrap')
    if len(images.shape)==3:
        images = np.pad(images,pad_width=((npad,npad),(npad,npad),(0,0)),mode='wrap')
    return images

images_test = periodic_padding(images_test, npad)
if modelb == 0:
    print('qu data')
    images_test[:,:,:,[0,1]] *= map_rescale_factor_q
    result = model.predict(images_test)
    result = result/map_rescale_factor_q
else: #modelb == 1: (B-models)
    images_test[:,:,:,0] *= map_rescale_factor_b
    images_test[:,:,:,1] *= map_rescale_factor_e
    result = model.predict(images_test)
    result = result/map_rescale_factor_b

np.save(result_folder + name_result, result)

