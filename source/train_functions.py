#author: Belén Costanza


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os 


from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomUniform


from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm



from numpy.random import seed
from read_input import load_config

import losses as loss

#import optuna

tf.keras.backend.set_floatx('float32')

strategy = tf.distribute.MirroredStrategy()


######### read config ############
#Warning message if you have more than one argument:
if len(sys.argv) != 2:
    print("Usage: python your_script.py input_file")
    sys.exit(1)

input_file = sys.argv[1]

#Dictionary with the parameters of the input file:
input_data = load_config(input_file)

data_folder = input_data["name_data_folder"]
result_folder = input_data["name_result_folder"]

initial_epoch = int(input_data["initial_epoch"])
epochs = int(input_data["epochs"])
seed = int(input_data["seed"])
lr=float(input_data["learning_rate"])

np.random.seed(seed)
tf.random.set_seed(seed)

Lsize = int(input_data["Lsize"])
npad = int(input_data["npad"])
npixels = int(input_data["npixels"])
batch_size = int(input_data["batch_size"])
num_train_examples = int(input_data["nsims_train"]) #defines how long a epoch is 
num_valid_examples = int(input_data["nsims_valid"])

modelb = int(input_data['modelb'])
iteration = int(input_data["iter"])
print('iteration:', iteration)
print('using model_b', modelb)

if iteration == 1:
    filename_train = input_data["name_train1"] 
    filename_valid = input_data["name_valid1"]
    map_rescale_factor_q = float(input_data["factor1"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model1"]
    name_loss = input_data["name_loss1"]    
elif iteration == 2:
    filename_train = input_data["name_train2"] 
    filename_valid = input_data["name_valid2"]
    map_rescale_factor_q = float(input_data["factor2"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model2"]
    name_loss = input_data["name_loss2"]
elif iteration == 3:
    filename_train = input_data["name_train3"] 
    filename_valid = input_data["name_valid3"]
    map_rescale_factor_q = float(input_data["factor3"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model3"]
    name_loss = input_data["name_loss3"]
elif iteration == 4:
    filename_train = input_data["name_train4"] 
    filename_valid = input_data["name_valid4"]
    map_rescale_factor_q = float(input_data["factor4"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model4"]
    name_loss = input_data["name_loss4"]
elif iteration == 5: 
    filename_train = input_data["name_train5"] 
    filename_valid = input_data["name_valid5"]
    map_rescale_factor_q = float(input_data["factor5"])
    map_rescale_factor_b = float(input_data["factorb"])
    name_model = input_data["name_model5"]
    name_loss = input_data["name_loss5"]


if modelb == 1 and iteration == 0:
    print('load bdata')
    filename_train = input_data["name_trainb"] 
    filename_valid = input_data["name_validb"]
    map_rescale_factor_q = float(input_data["factor3"])
    map_rescale_factor_b = float(input_data["factorb"])
    map_rescale_factor_e = float(input_data["factore"])
    name_model = input_data["name_modelb"]
    name_loss = input_data["name_lossb"]


mask_mode = int(input_data["mask"])

variance_map = np.load(data_folder + 'variance_QU_planck.npy')
variance_bmap = np.load(data_folder + 'variance_mapB_planck.npy')

masks = np.load(data_folder + 'masks_QU.npy')
if mask_mode == 0: 
    mask = masks[0]
else:
    mask = masks[1]


name_path = input_data["name_path"]

filters0 = int(input_data["filters0"])
filters1 = int(input_data["filters1"])
filters2 = int(input_data["filters2"])
filters3 = int(input_data["filters3"])  #32
filters4 = int(input_data["filters4"])  #32
filters5 = int(input_data["filters5"])  #32

save_model_path = result_folder + "modelb_" + str(modelb) + "_mask_" + str(mask_mode) + "_iter_" + str(iteration) + "/" + name_path
restore_model = os.path.exists(save_model_path+".index")

###################### records ######################################

def parse_QU(element): 
    
    npad = 256
    npixels = 256
        
    keys_to_features = {'qobs': tf.io.FixedLenFeature([], tf.string),
                        'uobs': tf.io.FixedLenFeature([], tf.string),
                        'mask': tf.io.FixedLenFeature([], tf.string),
                        'inho': tf.io.FixedLenFeature([], tf.string)}
    
    parsed_features = tf.io.parse_single_example(element, keys_to_features)
    
    qobs = parsed_features['qobs']
    uobs = parsed_features['uobs']
    mask = parsed_features['mask']
    inho = parsed_features['inho']
    
    qobs = tf.io.decode_raw(qobs, out_type=tf.float64)*map_rescale_factor_q
    uobs = tf.io.decode_raw(uobs, out_type=tf.float64)*map_rescale_factor_q
    mask = tf.io.decode_raw(mask, out_type=tf.float64)
    inho = tf.io.decode_raw(inho, out_type=tf.float64)

    qobs = tf.reshape(qobs, shape=[npixels, npixels, 1])
    uobs = tf.reshape(uobs, shape=[npixels, npixels, 1])
    mask = tf.reshape(mask, shape=[npixels, npixels, 1])
    inho = tf.reshape(inho, shape=[npixels, npixels, 1])
    
    def periodic_padding(images,npad):
        images = np.pad(images, pad_width=((npad,npad),(npad,npad),(0,0)),mode="wrap")
        return images
    
    parsed_features['qobs_pad'] = tf.py_function(periodic_padding, [qobs,npad], tf.float64 )
    parsed_features['uobs_pad'] = tf.py_function(periodic_padding, [uobs,npad], tf.float64 )
    parsed_features['mask_pad'] = tf.py_function(periodic_padding, [mask,npad], tf.float64 )
    parsed_features['inho_pad'] = tf.py_function(periodic_padding, [inho,npad], tf.float64 )

        
    image = tf.concat([parsed_features['qobs_pad'],parsed_features['uobs_pad'], parsed_features['mask_pad'], parsed_features['inho_pad']],axis=-1)
    label = tf.concat([qobs,uobs], axis=-1)

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
        
    return image, label

def parse_QU_iter(element): 
    
    npad= 256
    npixels = 256
        
    keys_to_features = {'qobs': tf.io.FixedLenFeature([], tf.string),
                        'uobs': tf.io.FixedLenFeature([], tf.string),
                        'ecnn': tf.io.FixedLenFeature([], tf.string),
                        'mask': tf.io.FixedLenFeature([], tf.string),
                        'inho': tf.io.FixedLenFeature([], tf.string)}
    
    parsed_features = tf.io.parse_single_example(element, keys_to_features)
    
    qobs = parsed_features['qobs']
    uobs = parsed_features['uobs']
    ecnn = parsed_features['ecnn']
    mask = parsed_features['mask']
    inho = parsed_features['inho']
    
    qobs = tf.io.decode_raw(qobs, out_type=tf.float64)*map_rescale_factor_q
    uobs = tf.io.decode_raw(uobs, out_type=tf.float64)*map_rescale_factor_q
    ecnn = tf.io.decode_raw(ecnn, out_type=tf.float64)*map_rescale_factor_q
    mask = tf.io.decode_raw(mask, out_type=tf.float64)
    inho = tf.io.decode_raw(inho, out_type=tf.float64)

    qobs = tf.reshape(qobs, shape=[npixels, npixels, 1])
    uobs = tf.reshape(uobs, shape=[npixels, npixels, 1])
    ecnn = tf.reshape(ecnn, shape=[npixels, npixels, 1])
    mask = tf.reshape(mask, shape=[npixels, npixels, 1])
    inho = tf.reshape(inho, shape=[npixels, npixels, 1])
    
    def periodic_padding(images,npad):
        images = np.pad(images, pad_width=((npad,npad),(npad,npad),(0,0)),mode="wrap")
        return images
    
    parsed_features['qobs_pad'] = tf.py_function(periodic_padding, [qobs,npad], tf.float64 )
    parsed_features['uobs_pad'] = tf.py_function(periodic_padding, [uobs,npad], tf.float64 )
    parsed_features['mask_pad'] = tf.py_function(periodic_padding, [mask,npad], tf.float64 )
    parsed_features['inho_pad'] = tf.py_function(periodic_padding, [inho,npad], tf.float64 )

        
    image = tf.concat([parsed_features['qobs_pad'],parsed_features['uobs_pad'], parsed_features['mask_pad'], parsed_features['inho_pad']],axis=-1)
    label = tf.concat([qobs,uobs,ecnn], axis=-1)

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
        
    return image, label

def parse_B(element): 
    
    npad = 256
    npixels = 256
        
    keys_to_features = {'bobs': tf.io.FixedLenFeature([], tf.string),
                        'ecnn': tf.io.FixedLenFeature([], tf.string),
                        'mask': tf.io.FixedLenFeature([], tf.string),
                        'inho': tf.io.FixedLenFeature([], tf.string)}
    
    parsed_features = tf.io.parse_single_example(element, keys_to_features)
    
    bobs = parsed_features['bobs']
    etot = parsed_features['ecnn']
    mask = parsed_features['mask']
    inho = parsed_features['inho']
    
    bobs = tf.io.decode_raw(bobs, out_type=tf.float64)*map_rescale_factor_b
    etot = tf.io.decode_raw(etot, out_type=tf.float64)*map_rescale_factor_e
    mask = tf.io.decode_raw(mask, out_type=tf.float64)
    inho = tf.io.decode_raw(inho, out_type=tf.float64)

    bobs = tf.reshape(bobs, shape=[npixels, npixels, 1])
    etot = tf.reshape(etot, shape=[npixels, npixels, 1])
    mask = tf.reshape(mask, shape=[npixels, npixels, 1])
    inho = tf.reshape(inho, shape=[npixels, npixels, 1])

    
    def periodic_padding(images,npad):
        images = np.pad(images, pad_width=((npad,npad),(npad,npad),(0,0)),mode="wrap")
        return images
    
    parsed_features['bobs_pad'] = tf.py_function(periodic_padding, [bobs,npad], tf.float64 )
    parsed_features['etot_pad'] = tf.py_function(periodic_padding, [etot,npad], tf.float64 )
    parsed_features['mask_pad'] = tf.py_function(periodic_padding, [mask,npad], tf.float64 )
    parsed_features['inho_pad'] = tf.py_function(periodic_padding, [inho,npad], tf.float64 )

        
    image = tf.concat([parsed_features['bobs_pad'],parsed_features['etot_pad'], parsed_features['mask_pad'], parsed_features['inho_pad']],axis=-1)
    label = bobs

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    
    
    return image, label


##################### networks #########################################

#cropping
def get_cropdim(input,target):
    inputdim=input.get_shape().as_list()[-3]
    targetdim=target.get_shape().as_list()[-2]
    
    if (inputdim-targetdim)%2 == 0:
        cropx=int((inputdim-targetdim)/2)
        cropy=cropx
    else: 
        cropx=int((inputdim-targetdim-1)/2)
        cropy=int((inputdim-targetdim+1)/2)
    return (cropx,cropy),(cropx,cropy)

def encoder_block(filters,image,nonlineal):
    y=layers.Conv2D(filters,(5,5),strides=(2,2),padding="valid", kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=seed))(image)
    if nonlineal:
        y = layers.Activation(activation="relu")(y)
#    y=layers.BatchNormalization()(y)
    return y

def decoder_block(filters,image,nonlineal):
    z=layers.UpSampling2D(size=(2,2))(image)
    z=layers.Conv2D(filters,(5,5),strides=(1,1),padding="valid",kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed = seed))(z)
    if nonlineal:
        z = layers.Activation(activation="relu")(z)
#    z=layers.BatchNormalization()(z)
    return z


#multiplicacion entre mascara y mapas
def multiplication(input1,input2,channels_out):
    channelnr_1 = input1.get_shape().as_list()[-1]
    channelnr_2 = input2.get_shape().as_list()[-1]
    
    #split channels
    channels_1 = []
    for i in range(channelnr_1):
        chan = layers.Lambda(lambda x: x[:,:,:,i:i+1])(input1)
        channels_1.append(chan)    
    channels_2 = []
    for i in range(channelnr_2):
        chan = layers.Lambda(lambda x: x[:,:,:,i:i+1])(input2)
        channels_2.append(chan)     
   
    #multiply and concat
    channels_multi = []
    for chan1 in channels_1:
        for chan2 in channels_2:
            multi = layers.Multiply()([chan1,chan2])
            channels_multi.append(multi)
    
    #also add linear channels
    for chan1 in channels_1:
        channels_multi.append(chan1)

    if (len(channels_multi)>1):
        multilayer = layers.Concatenate()(channels_multi)    
    else:
        multilayer = channels_multi[0] 
    
    #now 1x1 convo this
    outputs = layers.Conv2D(filters=channels_out, kernel_size=(1, 1), padding='same', strides=1,kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=seed))(multilayer)    
    return outputs


#########################################


def network(img_shape, channels_out):
    
    inputs=layers.Input(shape=img_shape)

    mapdata = layers.Lambda(lambda x: x[:,:,:,0:2])(inputs) #Qobs, Uobs (0,1)
    maskdata = layers.Lambda(lambda x: x[:,:,:,2:3])(inputs) #Mask (2)
    inhodata = layers.Lambda(lambda x: x[:,:,:,2:4])(inputs) #Mask and inho (2,3)

    nonlineal1 = True
    nonlineal2 = False

    #NO LINEAL (inho)

    #no lineal encoder  

    encoder0_nonlin=layers.Conv2D(filters0,(5,5),strides=(1,1),padding="valid",kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=seed))(inhodata)
    encoder0_nonlin=layers.Activation(activation="relu")(encoder0_nonlin)
    #encoder0_nonlin = layers.Activation(activation="relu")
    encoder1_nonlin=encoder_block(filters1,encoder0_nonlin,nonlineal1)
    #encoder1_nonlin = layers.Activation(activation="relu")
    encoder2_nonlin=encoder_block(filters2,encoder1_nonlin,nonlineal1)
    #encoder2_nonlin = layers.Activation(activation="relu")
    encoder3_nonlin=encoder_block(filters3,encoder2_nonlin,nonlineal1)
    #encoder3_nonlin = layers.Activation(activation="relu")
    encoder4_nonlin=encoder_block(filters4,encoder3_nonlin,nonlineal1)
    #encoder4_nonlin = layers.Activation(activation="relu")
    encoder5_nonlin=encoder_block(filters5,encoder4_nonlin,nonlineal1)
    #encoder5_nonlin = layers.Activation(activation="relu")
    encoder6_nonlin=encoder_block(filters5,encoder5_nonlin,nonlineal1)

    #decoder no lineal 
    decoder6_nonlin=decoder_block(filters5, encoder6_nonlin,nonlineal1)
    encoder5_nonlin_crop=layers.Cropping2D(cropping=get_cropdim(encoder5_nonlin,decoder6_nonlin))(encoder5_nonlin)
    skipcon5_nonlin=layers.concatenate([encoder5_nonlin_crop,decoder6_nonlin],axis=3)
    decoder5_nonlin=decoder_block(filters5, skipcon5_nonlin,nonlineal1)
    encoder4_nonlin_crop=layers.Cropping2D(cropping=get_cropdim(encoder4_nonlin,decoder5_nonlin))(encoder4_nonlin)
    skipcon4_nonlin=layers.concatenate([encoder4_nonlin_crop,decoder5_nonlin],axis=3)
    decoder4_nonlin=decoder_block(filters4,skipcon4_nonlin,nonlineal1)
    encoder3_nonlin_crop=layers.Cropping2D(cropping=get_cropdim(encoder3_nonlin,decoder4_nonlin))(encoder3_nonlin)
    skipcon3_nonlin=layers.concatenate([encoder3_nonlin_crop,decoder4_nonlin],axis=3)
    decoder3_nonlin=decoder_block(filters3,skipcon3_nonlin,nonlineal1)
    encoder2_nonlin_crop=layers.Cropping2D(cropping=get_cropdim(encoder2_nonlin,decoder3_nonlin))(encoder2_nonlin)
    skipcon2_nonlin=layers.concatenate([encoder2_nonlin_crop,decoder3_nonlin],axis=3)
    decoder2_nonlin=decoder_block(filters2,skipcon2_nonlin,nonlineal1)
    encoder1_nonlin_crop=layers.Cropping2D(cropping=get_cropdim(encoder1_nonlin,decoder2_nonlin))(encoder1_nonlin)
    skipcon1_nonlin=layers.concatenate([encoder1_nonlin_crop,decoder2_nonlin],axis=3)
    decoder1_nonlin=decoder_block(filters1,skipcon1_nonlin,nonlineal1)

    #decoder0_lin=layers.Conv2D(1,(5,5),strides=(1,1),padding="valid")(decoder1_nonlin)


    #NO LINEAL (mask) 

    encoder0_nonlin_2=layers.Conv2D(filters0,(5,5),strides=(1,1),padding="valid",kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=seed))(maskdata)
    encoder0_nonlin_2=layers.Activation(activation="relu")(encoder0_nonlin_2)
    encoder0_nonlin_2=multiplication(encoder0_nonlin_2,encoder0_nonlin,channels_out=filters0)
    encoder1_nonlin_2=encoder_block(filters1,encoder0_nonlin_2,nonlineal1)
    encoder1_nonlin_2=multiplication(encoder1_nonlin_2,encoder1_nonlin,channels_out=filters1)
    encoder2_nonlin_2=encoder_block(filters2,encoder1_nonlin_2,nonlineal1)
    encoder2_nonlin_2=multiplication(encoder2_nonlin_2,encoder2_nonlin,channels_out=filters2)
    encoder3_nonlin_2=encoder_block(filters3,encoder2_nonlin_2,nonlineal1)
    encoder4_nonlin_2=encoder_block(filters4,encoder3_nonlin_2,nonlineal1)
    encoder5_nonlin_2=encoder_block(filters5,encoder4_nonlin_2,nonlineal1)
    encoder6_nonlin_2=encoder_block(filters5,encoder5_nonlin_2,nonlineal1)


    #decoder
    decoder6_nonlin_2=decoder_block(filters5,encoder6_nonlin_2,nonlineal1)
    decoder6_nonlin_2=multiplication(decoder6_nonlin_2,decoder6_nonlin,channels_out=filters5)

    encoder5_nonlin_crop_2=layers.Cropping2D(cropping=get_cropdim(encoder5_nonlin_2,decoder6_nonlin_2))(encoder5_nonlin_2)
    skipcon5_nonlin_2=layers.concatenate([encoder5_nonlin_crop_2,decoder6_nonlin_2],axis=3)
    decoder5_nonlin_2=decoder_block(filters5,skipcon5_nonlin_2,nonlineal1)
    decoder5_nonlin_2=multiplication(decoder5_nonlin_2,decoder5_nonlin,channels_out=filters5)

    encoder4_nonlin_crop_2=layers.Cropping2D(cropping=get_cropdim(encoder4_nonlin_2,decoder5_nonlin_2))(encoder4_nonlin_2)
    skipcon4_nonlin_2=layers.concatenate([encoder4_nonlin_crop_2,decoder5_nonlin_2],axis=3)
    decoder4_nonlin_2=decoder_block(filters4,skipcon4_nonlin_2,nonlineal1)
    decoder4_nonlin_2=multiplication(decoder4_nonlin_2,decoder4_nonlin,channels_out=filters4)

    encoder3_nonlin_crop_2=layers.Cropping2D(cropping=get_cropdim(encoder3_nonlin_2,decoder4_nonlin_2))(encoder3_nonlin_2)
    skipcon3_nonlin_2=layers.concatenate([encoder3_nonlin_crop_2,decoder4_nonlin_2],axis=3)
    decoder3_nonlin_2=decoder_block(filters3,skipcon3_nonlin_2,nonlineal1)
    decoder3_nonlin_2=multiplication(decoder3_nonlin_2,decoder3_nonlin,channels_out=filters3)

    encoder2_nonlin_crop_2=layers.Cropping2D(cropping=get_cropdim(encoder2_nonlin_2,decoder3_nonlin_2))(encoder2_nonlin_2)
    skipcon2_nonlin_2=layers.concatenate([encoder2_nonlin_crop_2,decoder3_nonlin_2],axis=3)
    decoder2_nonlin_2=decoder_block(filters2,skipcon2_nonlin_2,nonlineal1)
    decoder2_nonlin_2=multiplication(decoder2_nonlin_2,decoder2_nonlin,channels_out=filters2)

    encoder1_nonlin_crop_2=layers.Cropping2D(cropping=get_cropdim(encoder1_nonlin_2,decoder2_nonlin_2))(encoder1_nonlin_2)
    skipcon1_nonlin_2=layers.concatenate([encoder1_nonlin_crop_2,decoder2_nonlin_2],axis=3)
    decoder1_nonlin_2=decoder_block(filters1,skipcon1_nonlin_2,nonlineal1)
    decoder1_nonlin_2=multiplication(decoder1_nonlin_2,decoder1_nonlin,channels_out=filters1)

    #encoder0_nonlin_crop=layers.Cropping2D(cropping=get_cropdim(encoder0_lin,decoder1_lin))(encoder0_lin)
    #skipcon0_lin=layers.concatenate([encoder0_lin_crop,decoder1_lin],axis=3)

    #decoder0_lin=layers.Conv2D(1,(5,5),strides=(1,1),padding="valid")(skipcon0_lin)

    #LINEAL (data)

    #encoder lineal
    encoder0_lin=layers.Conv2D(filters0,(5,5),strides=(1,1),padding="valid",kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=seed))(mapdata)
    encoder0_lin=multiplication(encoder0_lin,encoder0_nonlin_2,channels_out=filters0)
    encoder1_lin=encoder_block(filters1,encoder0_lin,nonlineal2)
    encoder1_lin=multiplication(encoder1_lin,encoder1_nonlin_2,channels_out=filters1)
    encoder2_lin=encoder_block(filters2,encoder1_lin,nonlineal2)
    encoder2_lin=multiplication(encoder2_lin,encoder2_nonlin_2,channels_out=filters2)
    encoder3_lin=encoder_block(filters3,encoder2_lin,nonlineal2)
    encoder4_lin=encoder_block(filters4,encoder3_lin,nonlineal2)
    encoder5_lin=encoder_block(filters5,encoder4_lin,nonlineal2)
    encoder6_lin=encoder_block(filters5,encoder5_lin,nonlineal2)


    #decoder
    decoder6_lin=decoder_block(filters5,encoder6_lin,nonlineal2)
    decoder6_lin=multiplication(decoder6_lin,decoder6_nonlin_2,channels_out=filters5)

    encoder5_lin_crop=layers.Cropping2D(cropping=get_cropdim(encoder5_lin,decoder6_lin))(encoder5_lin)
    skipcon5_lin=layers.concatenate([encoder5_lin_crop,decoder6_lin],axis=3)
    decoder5_lin=decoder_block(filters5,skipcon5_lin,nonlineal2)
    decoder5_lin=multiplication(decoder5_lin,decoder5_nonlin_2,channels_out=filters5)

    encoder4_lin_crop=layers.Cropping2D(cropping=get_cropdim(encoder4_lin,decoder5_lin))(encoder4_lin)
    skipcon4_lin=layers.concatenate([encoder4_lin_crop,decoder5_lin],axis=3)
    decoder4_lin=decoder_block(filters4,skipcon4_lin,nonlineal2)
    decoder4_lin=multiplication(decoder4_lin,decoder4_nonlin_2,channels_out=filters4)

    encoder3_lin_crop=layers.Cropping2D(cropping=get_cropdim(encoder3_lin,decoder4_lin))(encoder3_lin)
    skipcon3_lin=layers.concatenate([encoder3_lin_crop,decoder4_lin],axis=3)
    decoder3_lin=decoder_block(filters3,skipcon3_lin,nonlineal2)
    decoder3_lin=multiplication(decoder3_lin,decoder3_nonlin_2,channels_out=filters3)

    encoder2_lin_crop=layers.Cropping2D(cropping=get_cropdim(encoder2_lin,decoder3_lin))(encoder2_lin)
    skipcon2_lin=layers.concatenate([encoder2_lin_crop,decoder3_lin],axis=3)
    decoder2_lin=decoder_block(filters2,skipcon2_lin,nonlineal2)
    decoder2_lin=multiplication(decoder2_lin,decoder2_nonlin_2,channels_out=filters2)

    encoder1_lin_crop=layers.Cropping2D(cropping=get_cropdim(encoder1_lin,decoder2_lin))(encoder1_lin)
    skipcon1_lin=layers.concatenate([encoder1_lin_crop,decoder2_lin],axis=3)
    decoder1_lin=decoder_block(filters1,skipcon1_lin,nonlineal2)
    decoder1_lin=multiplication(decoder1_lin,decoder1_nonlin_2,channels_out=filters1)

    encoder0_lin_crop=layers.Cropping2D(cropping=get_cropdim(encoder0_lin,decoder1_lin))(encoder0_lin)
    skipcon0_lin=layers.concatenate([encoder0_lin_crop,decoder1_lin],axis=3)

    decoder0_lin=layers.Conv2D(channels_out,(5,5),strides=(1,1),padding="valid",kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=seed))(skipcon0_lin)

    #decoder0_lin = decoder0_lin - tf.reduce_mean(decoder0_lin)
    outputs = decoder0_lin
    
    return inputs, outputs

###############################################

#############read-DATASET##########################
dataset_train_raw = tf.data.TFRecordDataset(data_folder + filename_train)
dataset_valid_raw = tf.data.TFRecordDataset(data_folder + filename_valid)

if iteration == 1 and modelb == 0:
    print('using parse_QU')
    dataset_train = dataset_train_raw.map(parse_QU)
    dataset_valid = dataset_valid_raw.map(parse_QU)
elif iteration != 1 and modelb == 0:
    print('using parse_QU_iter')
    dataset_train = dataset_train_raw.map(parse_QU_iter)
    dataset_valid = dataset_valid_raw.map(parse_QU_iter)
elif modelb == 1:
    print('using parse_B')
    dataset_train = dataset_train_raw.map(parse_B)
    dataset_valid = dataset_valid_raw.map(parse_B)

dataset_train_parsed = dataset_train.shuffle(buffer_size=100,reshuffle_each_iteration=True).repeat().batch(batch_size)
dataset_valid_parsed = dataset_valid.repeat().batch(batch_size)

iterator_train = iter(dataset_train_parsed)
iterator_valid = iter(dataset_valid_parsed)

############# MODEL ##########################
with strategy.scope():
    img_shape = (npixels+2*npad, npixels+2*npad,4)
    if modelb == 0:
        channels_out = 2
    else: 
        channels_out = 1

    inputs, outputs = network(img_shape,channels_out)

    lossj3 = loss.loss_functions(Lsize, npixels, mask, variance_map, variance_bmap, map_rescale_factor_q, map_rescale_factor_b)

    if modelb == 1:
        print('using loss_b_j3')
        lossfunc = lossj3.loss_b_j3
    elif modelb == 0 and iteration == 1:
        print('using loss_wiener_j3')
        lossfunc = lossj3.loss_wiener_j3
    elif modelb == 0 and iteration != 1:
        print('using loss_wiener_j3_iter')
        lossfunc = lossj3.loss_wiener_j3_iter

    #lossfunc = losses.loss_wiener_j3
    optim = optimizers.Adam(learning_rate=lr)
    model=Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optim,loss=lossfunc)

    ######### TRAIN  ####################


    cp = tf.keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_loss',save_best_only=True, save_weights_only=True, verbose=1)
    callback_csv = tf.keras.callbacks.CSVLogger(result_folder + name_loss, append=True)

    if restore_model:
        print("WE TRAIN FROM PREVIOUS CHECKPOINT")
    else:
        print("WE TRAIN FROM START")

    if restore_model:
        print("loading weights from",save_model_path)
        model.load_weights(save_model_path)


    history = model.fit(iterator_train,
	  steps_per_epoch = int(np.ceil(num_train_examples / float(batch_size))),
          epochs=epochs,          
          validation_data=iterator_valid,
          validation_steps=int(np.ceil(num_valid_examples / float(batch_size))),	
          initial_epoch=initial_epoch,
          callbacks=[cp,callback_csv])

    model.save(result_folder + name_model)




        

