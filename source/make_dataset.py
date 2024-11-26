#author: Belén Costanza

import sys, platform, os
import numpy as np
import matplotlib.pyplot as plt
import scipy

import camb
from camb import model, initialpower

#import healpy as hp

import tensorflow as tf
import nifty7 as ift
import wf_noise as wf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import tensorflow.keras.losses

import utilities
import losses as loss


class make_dataset:

    def __init__(self, nsims_train, nsims_valid, nsims_test, filename_train, filename_valid, variance_map, variance_bmap, mask, npad, npixels, Lsize):

        """ Class to create the dataset in each iteration 

        Params: 

        nsims_train = training set size (number of maps) 
        nsims_valid = validation set size (number of maps)
        nsims_test = test set size (number of maps)
        filename_train = training set name 
        filename_valid = validation set name
        variance map = variance in each pixel for QU noise 
        variace bmap = variance in each pixel for B noise
        mask = it can be Mask1 or Mask2 defined in the paper 
        npad = size of the padding to prepare the network input (256)
        npixels = number of pixels (256)
        Lsize = size map in deg  (20)
        
        """

        self.nsims_train = nsims_train
        self.nsims_valid = nsims_valid
        self.nsims_test = nsims_test

        self.filename_train = filename_train
        self.filename_valid = filename_valid

        self.variance_map = variance_map
        self.variance_bmap = variance_bmap
        self.mask = mask

        self.npad = npad
        self.npixels = npixels 
        self.Lsize = Lsize

    def get_res(self):

        angular_resolution_deg = self.Lsize/self.npixels
        angular_resolution_arcmin = angular_resolution_deg*60
        rad2arcmin = 180.*60./np.pi
        d2r = np.pi/180.
        dx = self.Lsize*d2r / float(self.npixels)

        return dx, angular_resolution_arcmin 

    def get_nifty(self, angular_resolution_arcmin):

        cl_EE_for_map, cl_BB_for_map = utilities.signal_spectrum()

        signal_dic={'cls':cl_EE_for_map}
        noise_dic={'type': 'from varmap','varmap': self.variance_map}
        wfmap_e = wf.wfilter_TT(self.npixels,angular_resolution_arcmin,signal_dic,noise_dic)

        signal_dic={'cls':cl_BB_for_map}
        wfmap_b = wf.wfilter_TT(self.npixels,angular_resolution_arcmin,signal_dic,noise_dic)

        return wfmap_e, wfmap_b

    def get_models(self, name_model, map_rescale_factor_q, map_rescale_factor_b, modelb=0, modelqu_1 = True):

        lossj3 = loss.loss_functions(self.Lsize, self.npixels, self.mask, self.variance_map, self.variance_bmap, map_rescale_factor_q, map_rescale_factor_b)

        if modelb==1:
            loss_b_j3 = lossj3.loss_b_j3
            tf.keras.losses.custom_loss = loss_b_j3
            return load_model(name_model, custom_objects={'loss_b_j3':loss_b_j3})            
        if modelb==0 and modelqu_1 == True:
            loss_wiener_j3 = lossj3.loss_wiener_j3
            tf.keras.losses.custom_loss = loss_wiener_j3
            return load_model(name_model, custom_objects={'loss_wiener_j3':loss_wiener_j3})
        elif modelb==0 and modelqu_1 == False:
            loss_wiener_j3_iter = lossj3.loss_wiener_j3_iter
            tf.keras.losses.custom_loss = loss_wiener_j3_iter
            return load_model(name_model, custom_objects={'loss_wiener_j3_iter':loss_wiener_j3_iter})

    def get_nn_outputs(self, qobs, uobs, model, map_rescale_factor_q, dx, tfac):

        qcnn, ucnn = utilities.prediction(qobs, uobs, self.mask, model, map_rescale_factor_q, self.variance_map, self.npad)  
        efft_cnn, bfft_cnn = utilities.transf_eb(qcnn, ucnn, self.npixels, dx)
        ecnn = np.fft.irfft2(efft_cnn)*tfac
        bcnn = np.fft.irfft2(bfft_cnn)*tfac
        qecnn, uecnn = utilities.transf_qu_eb(ecnn, bcnn, self.npixels, dx, only_e=True, only_b=False)

        return qecnn, uecnn, ecnn 


    def _bytes_feature_image(self, image):
        value = tf.compat.as_bytes(image.tostring())
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def get_maps_records(self, iter, name_models, factors, modelb, mask_mode):

        nsims = self.nsims_train + self.nsims_valid
        dx, angular_resolution_arcmin = self.get_res()
        tfac = 1./(np.sqrt((dx * dx) / (self.npixels * self.npixels)))  

        wfmap_e, wfmap_b = self.get_nifty(angular_resolution_arcmin)

        #load models once
        if iter == 2: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

        elif iter == 3: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

        elif iter == 4: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

            print('loading model3')
            name_model3 = name_models['name_model3']
            map_rescale_factor_q3 = factors['factor3']
            model3 = self.get_models(name_model3, map_rescale_factor_q3, map_rescale_factor_b, modelb=0, modelqu_1=False)

        elif iter == 5: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

            print('loading model3')
            name_model3 = name_models['name_model3']
            map_rescale_factor_q3 = factors['factor3']
            model3 = self.get_models(name_model3, map_rescale_factor_q3, map_rescale_factor_b, modelb=0, modelqu_1=False)

            print('loading model4')
            name_model4 = name_models['name_model4']
            map_rescale_factor_q4 = factors['factor4']
            model4 = self.get_models(name_model4, map_rescale_factor_q4, map_rescale_factor_b, modelb=0, modelqu_1=False)

        elif modelb == 1 and mask_mode == 0: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

        elif modelb == 1 and mask_mode == 1: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

            print('loading model3')
            name_model3 = name_models['name_model3']
            map_rescale_factor_q3 = factors['factor3']
            model3 = self.get_models(name_model3, map_rescale_factor_q3, map_rescale_factor_b, modelb=0, modelqu_1=False)


        with tf.io.TFRecordWriter(self.filename_train) as writer_train, tf.io.TFRecordWriter(self.filename_valid) as writer_valid:

            for map_id in range(nsims):
                print ("map", map_id)

                mapa_e = wfmap_e.make_signal_map()
                mapa_b = wfmap_b.make_signal_map()

                narray_CMB_map_e = mapa_e.val
                narray_CMB_map_b = mapa_b.val

                noise=wfmap_e.make_noise_map()
                noise_array = noise.val
            
                power_e = utilities.power(narray_CMB_map_e, dx, self.npixels)
                power_b = utilities.power(narray_CMB_map_b, dx, self.npixels)
            
                power_q, power_u = utilities.transf_qu(power_e, power_b, self.npixels, dx)
                qsky = np.fft.irfft2(power_q)*tfac 
                usky = np.fft.irfft2(power_u)*tfac
            
                data_q = self.mask*(qsky+noise_array)  #qobs
                data_u = self.mask*(usky+noise_array)  #uobs

                if iter == 1: 

                    example = tf.train.Example(
                        features = tf.train.Features(
                        feature={
                        'qobs':self._bytes_feature_image(data_q),   #Qobs
                        'uobs':self._bytes_feature_image(data_u),   #Uobs
                        'esky':self._bytes_feature_image(narray_CMB_map_e),  #Esky
                        'bsky':self._bytes_feature_image(narray_CMB_map_b),  #Bsky
                        'qsky':self._bytes_feature_image(qsky),   #Qsky
                        'usky':self._bytes_feature_image(usky),   #Usky
                        'mask':self._bytes_feature_image(self.mask),
                        'inho':self._bytes_feature_image(self.variance_map)
                        }))
                    if map_id<self.nsims_train:
                        writer_train.write(example.SerializeToString())
                    if map_id>=self.nsims_train and map_id<(self.nsims_train+self.nsims_valid):
                        writer_valid.write(example.SerializeToString())

                if iter == 2:

                    qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)

                    qres = self.mask*(data_q - qecnn)  
                    ures = self.mask*(data_u - uecnn)

                    etot = ecnn

                    example = tf.train.Example(
                        features = tf.train.Features(
                        feature={
                        'qobs':self._bytes_feature_image(qres),   #Qobs
                        'uobs':self._bytes_feature_image(ures),   #Uobs
                        'ecnn':self._bytes_feature_image(etot),  #Esky
                        'bsky':self._bytes_feature_image(narray_CMB_map_b),  #Bsky
                        'mask':self._bytes_feature_image(self.mask),
                        'inho':self._bytes_feature_image(self.variance_map)
                        }))
                    if map_id<self.nsims_train:
                        writer_train.write(example.SerializeToString())
                    if map_id>=self.nsims_train and map_id<(self.nsims_train+self.nsims_valid):
                        writer_valid.write(example.SerializeToString())

                if iter == 3:

                    qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)

                    qres = self.mask*(data_q - qecnn)  
                    ures = self.mask*(data_u - uecnn)

                    qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)
 
                    qecorr = qecnn + qecnn_res
                    qres_2 = data_q*self.mask - qecorr*self.mask 
                    uecorr = uecnn + uecnn_res
                    ures_2 = data_u*self.mask - uecorr*self.mask

                    etot = ecnn + ecnn_res

                    example = tf.train.Example(
                        features = tf.train.Features(
                        feature={
                        'qobs':self._bytes_feature_image(qres_2),   #Qobs
                        'uobs':self._bytes_feature_image(ures_2),   #Uobs
                        'ecnn':self._bytes_feature_image(etot),  #Esky
                        'bsky':self._bytes_feature_image(narray_CMB_map_b),  #Bsky
                        'mask':self._bytes_feature_image(self.mask),
                        'inho':self._bytes_feature_image(self.variance_map)
                        }))
                    if map_id<self.nsims_train:
                        writer_train.write(example.SerializeToString())
                    if map_id>=self.nsims_train and map_id<(self.nsims_train+self.nsims_valid):
                        writer_valid.write(example.SerializeToString())

                if iter == 4: 

                    qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)

                    qres = self.mask*(data_q - qecnn)  
                    ures = self.mask*(data_u - uecnn)

                    qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)

                    qecorr = qecnn + qecnn_res
                    qres_2 = data_q*self.mask - qecorr*self.mask 
                    uecorr = uecnn + uecnn_res
                    ures_2 = data_u*self.mask - uecorr*self.mask

                    qecnn_res2, uecnn_res2, ecnn_res2 = self.get_nn_outputs(qres_2, ures_2, model3, map_rescale_factor_q3, dx, tfac)

                    qecorr2 = qecnn + qecnn_res + qecnn_res2
                    qres_3 = data_q*self.mask - qecorr2*self.mask 
                    uecorr2 = uecnn + uecnn_res + uecnn_res2
                    ures_3 = data_u*self.mask - uecorr2*self.mask

                    etot = ecnn + ecnn_res + ecnn_res2

                    example = tf.train.Example(
                        features = tf.train.Features(
                        feature={
                        'qobs':self._bytes_feature_image(qres_3),   #Qobs
                        'uobs':self._bytes_feature_image(ures_3),   #Uobs
                        'ecnn':self._bytes_feature_image(etot),  #Esky
                        'bsky':self._bytes_feature_image(narray_CMB_map_b),  #Bsky
                        'mask':self._bytes_feature_image(self.mask),
                        'inho':self._bytes_feature_image(self.variance_map)
                        }))
                    if map_id<self.nsims_train:
                        writer_train.write(example.SerializeToString())
                    if map_id>=self.nsims_train and map_id<(self.nsims_train+self.nsims_valid):
                        writer_valid.write(example.SerializeToString())

                if iter == 5: 

                    qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)
 
                    qres = self.mask*(data_q - qecnn)  
                    ures = self.mask*(data_u - uecnn)

                    qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)

                    qecorr = qecnn + qecnn_res
                    qres_2 = data_q*self.mask - qecorr*self.mask 
                    uecorr = uecnn + uecnn_res
                    ures_2 = data_u*self.mask - uecorr*self.mask

                    qecnn_res2, uecnn_res2, ecnn_res2 = self.get_nn_outputs(qres_2, ures_2, model3, map_rescale_factor_q3, dx, tfac)

                    qecorr2 = qecnn + qecnn_res + qecnn_res2
                    qres_3 = data_q*self.mask - qecorr2*self.mask 
                    uecorr2 = uecnn + uecnn_res + uecnn_res2
                    ures_3 = data_u*self.mask - uecorr2*self.mask

                    qecnn_res3, uecnn_res3, ecnn_res3 = self.get_nn_outputs(qres_3, ures_3, model4, map_rescale_factor_q4, dx, tfac)

                    qecorr3 = qecnn + qecnn_res + qecnn_res2 + qecnn_res3
                    qres_4 = data_q*self.mask - qecorr3*self.mask 
                    uecorr3 = uecnn + uecnn_res + uecnn_res2 + uecnn_res3
                    ures_4 = data_u*self.mask - uecorr3*self.mask 

                    etot = ecnn + ecnn_res + ecnn_res2 + ecnn_res3

                    example = tf.train.Example(
                        features = tf.train.Features(
                        feature={
                        'qobs':self._bytes_feature_image(qres_4),   #Qobs
                        'uobs':self._bytes_feature_image(ures_4),   #Uobs
                        'ecnn':self._bytes_feature_image(etot),  #Esky
                        'bsky':self._bytes_feature_image(narray_CMB_map_b),  #Bsky
                        'mask':self._bytes_feature_image(self.mask),
                        'inho':self._bytes_feature_image(self.variance_map)
                        }))
                    if map_id<self.nsims_train:
                        writer_train.write(example.SerializeToString())
                    if map_id>=self.nsims_train and map_id<(self.nsims_train+self.nsims_valid):
                        writer_valid.write(example.SerializeToString())


                if modelb == 1 and mask_mode==0: 

                    qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)
 
                    qres = self.mask*(data_q - qecnn)  
                    ures = self.mask*(data_u - uecnn)

                    qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)

                    qecorr = qecnn + qecnn_res
                    qres_2 = data_q*self.mask - qecorr*self.mask 
                    uecorr = uecnn + uecnn_res
                    ures_2 = data_u*self.mask - uecorr*self.mask

                    efft_res2, bfft_res2 = utilities.transf_eb(qres_2, ures_2, self.npixels, dx)
                    bres = np.fft.irfft2(bfft_res2)*tfac  #we only took the b-mode of qres_2, ures_2

                    etot = ecnn + ecnn_res

                    example = tf.train.Example(
                        features = tf.train.Features(
                        feature={
                        'bobs':self._bytes_feature_image(bres),   #Qobs
                        'ecnn':self._bytes_feature_image(etot),  #Esky
                        'bsky':self._bytes_feature_image(narray_CMB_map_b),  #Bsky
                        'mask':self._bytes_feature_image(self.mask),
                        'inho':self._bytes_feature_image(self.variance_bmap)
                        }))
                    if map_id<self.nsims_train:
                        writer_train.write(example.SerializeToString())
                    if map_id>=self.nsims_train and map_id<(self.nsims_train+self.nsims_valid):
                        writer_valid.write(example.SerializeToString())

                if modelb == 1 and mask_mode == 1: 

                    qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)
 
                    qres = self.mask*(data_q - qecnn)  
                    ures = self.mask*(data_u - uecnn)

                    qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)

                    qecorr = qecnn + qecnn_res
                    qres_2 = data_q*self.mask - qecorr*self.mask 
                    uecorr = uecnn + uecnn_res
                    ures_2 = data_u*self.mask - uecorr*self.mask

                    qecnn_res2, uecnn_res2, ecnn_res2 = self.get_nn_outputs(qres_2, ures_2, model3, map_rescale_factor_q3, dx, tfac)

                    qecorr2 = qecnn + qecnn_res + qecnn_res2
                    qres_3 = data_q*self.mask - qecorr2*self.mask 
                    uecorr2 = uecnn + uecnn_res + uecnn_res2
                    ures_3 = data_u*self.mask - uecorr2*self.mask

                    efft_res3, bfft_res3 = utilities.transf_eb(qres_3, ures_3, self.npixels, dx)
                    bres = np.fft.irfft2(bfft_res3)*tfac  #we only took the b-mode of qres_3, ures_3

                    etot = ecnn + ecnn_res + ecnn_res2

                    example = tf.train.Example(
                        features = tf.train.Features(
                        feature={
                        'bobs':self._bytes_feature_image(bres),   #Qobs
                        'ecnn':self._bytes_feature_image(etot),  #Esky
                        'bsky':self._bytes_feature_image(narray_CMB_map_b),  #Bsky
                        'mask':self._bytes_feature_image(self.mask),
                        'inho':self._bytes_feature_image(self.variance_bmap)
                        }))
                    if map_id<self.nsims_train:
                        writer_train.write(example.SerializeToString())
                    if map_id>=self.nsims_train and map_id<(self.nsims_train+self.nsims_valid):
                        writer_valid.write(example.SerializeToString())


    def get_maps_test(self, filename_test, filename_test_noiseless, iter, name_models, factors, modelb, mask_mode):

        nsims = self.nsims_test
        dx, angular_resolution_arcmin = self.get_res()
        tfac = 1./(np.sqrt((dx * dx) / (self.npixels * self.npixels)))  

        wfmap_e, wfmap_b = self.get_nifty(angular_resolution_arcmin)

        if iter == 2:

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

        elif iter == 3: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

        elif iter == 4: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

            print('loading model3')
            name_model3 = name_models['name_model3']
            map_rescale_factor_q3 = factors['factor3']
            model3 = self.get_models(name_model3, map_rescale_factor_q3, map_rescale_factor_b, modelb=0, modelqu_1=False)

        elif iter == 5: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

            print('loading model3')
            name_model3 = name_models['name_model3']
            map_rescale_factor_q3 = factors['factor3']
            model3 = self.get_models(name_model3, map_rescale_factor_q3, map_rescale_factor_b, modelb=0, modelqu_1=False)

            print('loading model4')
            name_model4 = name_models['name_model4']
            map_rescale_factor_q4 = factors['factor4']
            model4 = self.get_models(name_model4, map_rescale_factor_q4, map_rescale_factor_b, modelb=0, modelqu_1=False)

        elif modelb == 1 and mask_mode == 0: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

        elif modelb == 1 and mask_mode == 1: 

            print('loading model1')
            name_model1 = name_models['name_model1']
            map_rescale_factor_q1 = factors['factor1']
            map_rescale_factor_b = factors['factorb'] #only to call loss class
            model1 = self.get_models(name_model1, map_rescale_factor_q1, map_rescale_factor_b, modelb=0, modelqu_1=True)

            print('loading model2')
            name_model2 = name_models['name_model2']
            map_rescale_factor_q2 = factors['factor2']
            model2 = self.get_models(name_model2, map_rescale_factor_q2, map_rescale_factor_b, modelb=0, modelqu_1=False)

            print('loading model3')
            name_model3 = name_models['name_model3']
            map_rescale_factor_q3 = factors['factor3']
            model3 = self.get_models(name_model3, map_rescale_factor_q3, map_rescale_factor_b, modelb=0, modelqu_1=False)


        dataset_test_noiseless = np.zeros((nsims, self.npixels, self.npixels, 4))
        dataset_test_noise = np.zeros((nsims, self.npixels, self.npixels, 4))

        for map_id in range(nsims):
            print ("map", map_id)


            mapa_e = wfmap_e.make_signal_map()
            mapa_b = wfmap_b.make_signal_map()

            narray_CMB_map_e = mapa_e.val
            narray_CMB_map_b = mapa_b.val

            noise=wfmap_e.make_noise_map()
            noise_array = noise.val
            
            power_e = utilities.power(narray_CMB_map_e, dx, self.npixels)
            power_b = utilities.power(narray_CMB_map_b, dx, self.npixels)
            
            power_q, power_u = utilities.transf_qu(power_e, power_b, self.npixels, dx)
            qsky = np.fft.irfft2(power_q)*tfac 
            usky = np.fft.irfft2(power_u)*tfac
            
            data_q = self.mask*(qsky+noise_array)  #qobs
            data_u = self.mask*(usky+noise_array)  #uobs

            dataset_test_noiseless[map_id,:,:,0]=narray_CMB_map_e  #Esky
            dataset_test_noiseless[map_id,:,:,1]=narray_CMB_map_b  #Bsky
            dataset_test_noiseless[map_id,:,:,2]=qsky
            dataset_test_noiseless[map_id,:,:,3]=usky

            if iter == 1: 

                dataset_test_noise[map_id,:,:,0]=data_q   #Qobs
                dataset_test_noise[map_id,:,:,1]=data_u   #Uobs
                dataset_test_noise[map_id,:,:,2]=self.mask
                dataset_test_noise[map_id,:,:,3]=self.variance_map
    

            if iter == 2:

                qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)

                qres = self.mask*(data_q - qecnn)  
                ures = self.mask*(data_u - uecnn)

                etot = ecnn

                dataset_test_noise[map_id,:,:,0]=qres   #Qobs
                dataset_test_noise[map_id,:,:,1]=ures   #Uobs
                dataset_test_noise[map_id,:,:,2]=self.mask
                dataset_test_noise[map_id,:,:,3]=self.variance_map


            if iter == 3:

                qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)

                qres = self.mask*(data_q - qecnn)  
                ures = self.mask*(data_u - uecnn)

                qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)
 
                qecorr = qecnn + qecnn_res
                qres_2 = data_q*self.mask - qecorr*self.mask 
                uecorr = uecnn + uecnn_res
                ures_2 = data_u*self.mask - uecorr*self.mask

                etot = ecnn + ecnn_res

                dataset_test_noise[map_id,:,:,0]=qres_2   #Qobs
                dataset_test_noise[map_id,:,:,1]=ures_2   #Uobs
                dataset_test_noise[map_id,:,:,2]=self.mask
                dataset_test_noise[map_id,:,:,3]=self.variance_map


            if iter == 4: 

                qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)

                qres = self.mask*(data_q - qecnn)  
                ures = self.mask*(data_u - uecnn)

                qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)

                qecorr = qecnn + qecnn_res
                qres_2 = data_q*self.mask - qecorr*self.mask 
                uecorr = uecnn + uecnn_res
                ures_2 = data_u*self.mask - uecorr*self.mask

                qecnn_res2, uecnn_res2, ecnn_res2 = self.get_nn_outputs(qres_2, ures_2, model3, map_rescale_factor_q3, dx, tfac)

                qecorr2 = qecnn + qecnn_res + qecnn_res2
                qres_3 = data_q*self.mask - qecorr2*self.mask 
                uecorr2 = uecnn + uecnn_res + uecnn_res2
                ures_3 = data_u*self.mask - uecorr2*self.mask

                etot = ecnn + ecnn_res + ecnn_res2

                dataset_test_noise[map_id,:,:,0]=qres_3   #Qobs
                dataset_test_noise[map_id,:,:,1]=ures_3   #Uobs
                dataset_test_noise[map_id,:,:,2]=self.mask
                dataset_test_noise[map_id,:,:,3]=self.variance_map

                
            if iter == 5: 

                qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)
 
                qres = self.mask*(data_q - qecnn)  
                ures = self.mask*(data_u - uecnn)

                qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)

                qecorr = qecnn + qecnn_res
                qres_2 = data_q*self.mask - qecorr*self.mask 
                uecorr = uecnn + uecnn_res
                ures_2 = data_u*self.mask - uecorr*self.mask

                qecnn_res2, uecnn_res2, ecnn_res2 = self.get_nn_outputs(qres_2, ures_2, model3, map_rescale_factor_q3, dx, tfac)

                qecorr2 = qecnn + qecnn_res + qecnn_res2
                qres_3 = data_q*self.mask - qecorr2*self.mask 
                uecorr2 = uecnn + uecnn_res + uecnn_res2
                ures_3 = data_u*self.mask - uecorr2*self.mask

                qecnn_res3, uecnn_res3, ecnn_res3 = self.get_nn_outputs(qres_3, ures_3, model4, map_rescale_factor_q4, dx, tfac)

                qecorr3 = qecnn + qecnn_res + qecnn_res2 + qecnn_res3
                qres_4 = data_q*self.mask - qecorr3*self.mask 
                uecorr3 = uecnn + uecnn_res + uecnn_res2 + uecnn_res3
                ures_4 = data_u*self.mask - uecorr3*self.mask 

                etot = ecnn + ecnn_res + ecnn_res2 + ecnn_res3

                dataset_test_noise[map_id,:,:,0]=qres_4   #Qobs
                dataset_test_noise[map_id,:,:,1]=ures_4   #Uobs
                dataset_test_noise[map_id,:,:,2]=self.mask
                dataset_test_noise[map_id,:,:,3]=self.variance_map


            if modelb == 1 and mask_mode==0: 

                qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)
 
                qres = self.mask*(data_q - qecnn)  
                ures = self.mask*(data_u - uecnn)

                qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)

                qecorr = qecnn + qecnn_res
                qres_2 = data_q*self.mask - qecorr*self.mask 
                uecorr = uecnn + uecnn_res
                ures_2 = data_u*self.mask - uecorr*self.mask

                efft_res2, bfft_res2 = utilities.transf_eb(qres_2, ures_2, self.npixels, dx)
                bres = np.fft.irfft2(bfft_res2)*tfac  #we only took the b-mode of qres_2, ures_2

                etot = ecnn + ecnn_res

                dataset_test_noise[map_id,:,:,0]=bres   #Qobs
                dataset_test_noise[map_id,:,:,1]=etot   #Uobs
                dataset_test_noise[map_id,:,:,2]=self.mask
                dataset_test_noise[map_id,:,:,3]=self.variance_bmap

            if modelb == 1 and mask_mode == 1: 

                qecnn, uecnn, ecnn = self.get_nn_outputs(data_q, data_u, model1, map_rescale_factor_q1, dx, tfac)
 
                qres = self.mask*(data_q - qecnn)  
                ures = self.mask*(data_u - uecnn)

                qecnn_res, uecnn_res, ecnn_res = self.get_nn_outputs(qres, ures, model2, map_rescale_factor_q2, dx, tfac)

                qecorr = qecnn + qecnn_res
                qres_2 = data_q*self.mask - qecorr*self.mask 
                uecorr = uecnn + uecnn_res
                ures_2 = data_u*self.mask - uecorr*self.mask

                qecnn_res2, uecnn_res2, ecnn_res2 = self.get_nn_outputs(qres_2, ures_2, model3, map_rescale_factor_q3, dx, tfac)

                qecorr2 = qecnn + qecnn_res + qecnn_res2
                qres_3 = data_q*self.mask - qecorr2*self.mask 
                uecorr2 = uecnn + uecnn_res + uecnn_res2
                ures_3 = data_u*self.mask - uecorr2*self.mask

                efft_res3, bfft_res3 = utilities.transf_eb(qres_3, ures_3, self.npixels, dx)
                bres = np.fft.irfft2(bfft_res3)*tfac  #we only took the b-mode of qres_3, ures_3

                etot = ecnn + ecnn_res + ecnn_res2

                dataset_test_noise[map_id,:,:,0]=bres   #Qobs
                dataset_test_noise[map_id,:,:,1]=etot  #Uobs
                dataset_test_noise[map_id,:,:,2]=self.mask
                dataset_test_noise[map_id,:,:,3]=self.variance_bmap

        np.save(filename_test, dataset_test_noise)
        np.save(filename_test_noiseless, dataset_test_noiseless)




            

