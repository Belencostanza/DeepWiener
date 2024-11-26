#author: Belén Costanza


import numpy as np
import sys
import os
import scipy 
from scipy import interpolate

import nifty7 as ift
import wf_noise as wf

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import tensorflow.keras.losses

import camb
from camb import model, initialpower
import utilities


import losses as loss


class PowerSpectrum:

    def __init__(self, Lsize, npixels, npad, nbins, bins, modelb, mask_mode, mask, variance_map, variance_bmap):

        """Class to calculate the noise bias term and the fisher matrix

        Params: 

        Lsize = size map in deg (20)
        npixels = number of pixels (256)
        npad = size of the padding to prepare network input (256)
        nbins = number of bins 
        bins = edges of the bins
        modelb = if we only use the models with QU modelb is equal to 0, if we use the models with B modelb is equal to 1
        mask_mode = equal to 0 for Mask1 and equal to 1 for Mask2 
        mask = it can be Mask1 or Mask2 defined in the paper
        variance_map = variance in each pixel for QU noise 
        variance_bmap = variance in each pixel for B noise

        """

        self.Lsize = Lsize
        self.npixels = npixels 
        self.npad = npad
        self.nbins = nbins
        self.bins = bins
        self.modelb = modelb
        self.mask_mode = mask_mode
        self.mask = mask
        self.variance_map = variance_map
        self.variance_bmap = variance_bmap

    def get_res(self):

        angular_resolution_deg = self.Lsize/self.npixels
        angular_resolution_arcmin = angular_resolution_deg*60
        rad2arcmin = 180.*60./np.pi
        d2r = np.pi/180.
        dx = self.Lsize*d2r / float(self.npixels)

        return dx, angular_resolution_arcmin

    def get_models(self, factors, name_models): #charge models once 

        map_rescale_factor_q1 = factors["factor1"]

        name_model1 = name_models["name_model1"] #modelo que dado Qobs te da Qcnn, Ucnn
        name_model2 = name_models["name_model2"] #modelo que dado (Qobs-Qecnn) te da Qres
        name_model3 = name_models["name_model3"] #modelo que dado (Qobs-Qecnn) te da Qres
        name_model4 = name_models["name_model4"]
        name_model5 = name_models["name_model5"]
        name_modelb = name_models["name_modelb"]

        map_rescale_factor_b = factors["factorb"]

        lossj3 = loss.loss_functions(self.Lsize, self.npixels, self.mask, self.variance_map, self.variance_bmap, map_rescale_factor_q1, map_rescale_factor_b)

        if self.modelb == 1: 
            loss_wiener_j3 = lossj3.loss_wiener_j3
            tf.keras.losses.custom_loss = loss_wiener_j3
            model1 = load_model(name_model1, custom_objects={'loss_wiener_j3':loss_wiener_j3})

            loss_wiener_j3_iter = lossj3.loss_wiener_j3_iter
            tf.keras.losses.custom_loss = loss_wiener_j3_iter
            model2 = load_model(name_model2, custom_objects={'loss_wiener_j3_iter':loss_wiener_j3_iter})
            
            loss_b_j3 = lossj3.loss_b_j3
            tf.keras.losses.custom_loss = loss_b_j3
            modelb = load_model(name_modelb, custom_objects={'loss_b_j3':loss_b_j3})

            if self.mask_mode == 0: 
                return modelb, model1, model2 
            elif self.mask_mode == 1: 
                model3 = load_model(name_model3, custom_objects={'loss_wiener_j3_iter':loss_wiener_j3_iter})
                return modelb, model1, model2, model3

        else:
            loss_wiener_j3 = lossj3.loss_wiener_j3
            tf.keras.losses.custom_loss = loss_wiener_j3
            model1 = load_model(name_model1, custom_objects={'loss_wiener_j3':loss_wiener_j3})

            loss_wiener_j3_iter = lossj3.loss_wiener_j3_iter
            tf.keras.losses.custom_loss = loss_wiener_j3_iter
            model2 = load_model(name_model2, custom_objects={'loss_wiener_j3_iter':loss_wiener_j3_iter})
            model3 = load_model(name_model3, custom_objects={'loss_wiener_j3_iter':loss_wiener_j3_iter})
            model4 = load_model(name_model4, custom_objects={'loss_wiener_j3_iter':loss_wiener_j3_iter})

            if self.mask_mode == 0:
                return model1, model2, model3, model4
            elif self.mask_mode == 1: 
                model5 = load_model(name_model5, custom_objects={'loss_wiener_j3_iter':loss_wiener_j3_iter})
                return model1, model2, model3, model4, model5

    def flat_spectrum(self, cl_angular): 

        lx,ly=np.meshgrid( np.fft.fftfreq( self.npixels, 1/float(self.npixels) ),np.fft.fftfreq( self.npixels, 1/float(self.npixels) ) )
        l = np.sqrt(lx**2 + ly**2) #shape (128,65)
        ell_flat = l.flatten()
    
        ell_flat = ell_flat*18.
   
        ls = np.arange(len(cl_angular))
        inter = scipy.interpolate.interp1d(ls[2:], cl_angular[2:],bounds_error=False,fill_value=np.min(cl_angular[2:]))
        cl_plano = inter(ell_flat)
   
        return ell_flat, cl_plano

    def flat_spectrum_real(self, cl_angular): 

        lx,ly=np.meshgrid( np.fft.rfftfreq( self.npixels, 1/float(self.npixels) ),np.fft.fftfreq( self.npixels, 1/float(self.npixels) ) )
        l = np.sqrt(lx**2 + ly**2) 
        ell_flat = l.flatten()
    
        ell_flat = ell_flat*18.
   
        ls = np.arange(len(cl_angular))
        inter = scipy.interpolate.interp1d(ls[2:], cl_angular[2:],bounds_error=False,fill_value=np.min(cl_angular[2:]))
        cl_plano = inter(ell_flat)
   
        return ell_flat, cl_plano

    def spectrum_unique(self, ell_flat, cl_plane):

        ell_flat_unique, rep = np.unique(ell_flat, return_counts=True)
        cl_flat_unique = np.zeros(ell_flat_unique.shape)

        idx = 0
        for ell_value in ell_flat_unique:
            cl_flat_unique[idx] = cl_plane[np.where(ell_flat == ell_value)][0]
            idx+=1

        return ell_flat_unique, cl_flat_unique

    def bineado(self, ell, cl): 

        bin_indices = np.digitize(ell, self.bins, right=False)

        count = np.zeros(len(self.bins) - 1)
        cl_bin_sum = np.zeros(len(self.bins) - 1)
        el_med_sum = np.zeros(len(self.bins) - 1)

        # Calculate sum of ell values and cl values for each bin
        for i in range(1, len(self.bins)):
            mask = (bin_indices == i)
            count[i - 1] = np.sum(mask)
            cl_bin_sum[i - 1] = np.sum(cl[mask])
            el_med_sum[i - 1] = np.sum(ell[mask] * cl[mask])

        # Calculate the binned results
        el_med = (el_med_sum / cl_bin_sum).astype(int)
        cl_bin = cl_bin_sum / count

        return el_med, cl_bin, count

    def power(self, map):

        dx, _ = self.get_res() 
        tfac = np.sqrt((dx*dx)/(self.npixels*self.npixels))
        fft = np.fft.fft2(map[:,:])*tfac
        fft_shape = fft.shape 
        power = np.real(fft*np.conj(fft))
        power_reshape = np.reshape(power, [fft_shape[0]*fft_shape[1]]) 

        return power_reshape

    def power_real(self, map):

        dx,_ = self.get_res()
        tfac = np.sqrt((dx*dx)/(self.npixels*self.npixels))
        rfft = np.fft.rfft2(map[:,:])*tfac
        rfft_shape = rfft.shape 
        power = np.real(rfft*np.conj(rfft))
        power_reshape = np.reshape(power, [rfft_shape[0]*rfft_shape[1]]) 
        
        return power_reshape, rfft

    def make_map_noseed(self, cle, clb):

        dx, resolution = self.get_res()
        tfac = np.sqrt((dx*dx)/(self.npixels*self.npixels))
        signal_dic={'cls':cle}
        noise_dic={'type': 'from varmap','varmap': self.variance_map}
        wfmap_e = wf.wfilter_TT(self.npixels,resolution,signal_dic,noise_dic)   
        signal_dic={'cls':clb}
        wfmap_b = wf.wfilter_TT(self.npixels,resolution,signal_dic,noise_dic)
  
        signale = wfmap_e.make_signal_map()
        esky = signale.val

        signalb = wfmap_b.make_signal_map()
        bsky = signalb.val

        noise = wfmap_e.make_noise_map()
        noise_array = noise.val

        _,power_e = self.power_real(esky)  #efft
        _,power_b = self.power_real(bsky)  #bfft
            
        power_q, power_u = utilities.transf_qu(power_e, power_b, self.npixels, dx)
        qsky = np.fft.irfft2(power_q)/tfac 
        usky = np.fft.irfft2(power_u)/tfac

        dataq = self.mask*(qsky + noise_array)
        datau = self.mask*(usky + noise_array)

        return dataq, datau, qsky, usky, esky, bsky

    def map_noise(self, cl_ang_e):

        dx, resolution = self.get_res()
        signal_dic={'cls':cl_ang_e}  #esto es indistinto
        noise_dic={'type':'from varmap','varmap': self.variance_map}
        wfmap = wf.wfilter_TT(self.npixels,resolution,signal_dic,noise_dic, mask_dic=None)
        noise = wfmap.make_noise_map()
        noise_array = noise.val
        
        return noise_array

    def periodic_padding(self, images):

        if len(images.shape)==4:
            images = np.pad(images,pad_width=((0,0),(self.npad, self.npad),(self.npad, self.npad),(0,0)),mode='wrap')
        if len(images.shape)==3:
            images = np.pad(images,pad_width=((self.npad, self.npad),(self.npad, self.npad),(0,0)),mode='wrap')
        return images


    def prediction(self, data, model, factor):

    #dato = np.concatenate((dato_q.reshape(1,nx,nx,1), dato_u.reshape(1,nx,nx,1), mask.reshape(1,nx,nx,1), variance_map.reshape(1,nx,nx,1)), axis=-1)
        images_test = self.periodic_padding(data)
        images_test[:,:,:,[0,1]] *= factor

        result = model.predict(images_test)
        result = result/factor


        qcnn = result[:,:,:,0]
        ucnn = result[:,:,:,1]

        return qcnn, ucnn

    def prediction_bdato(self, dato, model, factorb, factore):

    #dato = np.concatenate((dato_q.reshape(1,nx,nx,1), dato_u.reshape(1,nx,nx,1), mask.reshape(1,nx,nx,1), variance_map.reshape(1,nx,nx,1)), axis=-1)
        images_test = self.periodic_padding(dato)
        images_test[:,:,:,0] *= factorb
        images_test[:,:,:,1] *= factore

        result = model.predict(images_test)
        result = result/factorb

        bcnn = result[:,:,:,0]

        return bcnn#, ucnn

    def get_nn_outputs(self, data_red, model, map_rescale_factor_q, dx, tfac):

        qcnn, ucnn = self.prediction(data_red, model, map_rescale_factor_q)  
        efft_cnn, bfft_cnn = utilities.transf_eb(qcnn, ucnn, self.npixels, dx)
        ecnn = np.fft.irfft2(efft_cnn)/tfac
        bcnn = np.fft.irfft2(bfft_cnn)/tfac
        qecnn, uecnn = utilities.transf_qu_eb(ecnn, bcnn, self.npixels, dx, only_e=True, only_b=False)

        return qecnn, uecnn, qcnn, ucnn, ecnn


    def WF_red_bdato(self, qobs_fid, uobs_fid, factors, models):

        map_rescale_factor_q1 = factors["factor1"]
        map_rescale_factor_q2 = factors["factor2"]
        map_rescale_factor_q3 = factors["factor3"]
        map_rescale_factor_b = factors["factorb"]
        map_rescale_factor_e = factors["factore"]

        modelb = models[0]
        model1 = models[1]
        model2 = models[2]
        if self.mask_mode == 1: 
            model3 = models[3]

        dx,_ = self.get_res()
        tfac = np.sqrt((dx*dx)/(self.npixels*self.npixels))
        
        data_red = np.zeros((1,self.npixels,self.npixels,4))
        data_red[:,:,:,0] = qobs_fid
        data_red[:,:,:,1] = uobs_fid
        data_red[:,:,:,2] = self.mask
        data_red[:,:,:,3] = self.variance_map

        qecnn1, uecnn1, qcnn1, ucnn1, ecnn1 = self.get_nn_outputs(data_red, model1, map_rescale_factor_q1, dx, tfac)

        qobs_fid2 = self.mask*(qobs_fid - qecnn1)  #1er residuo, tiene leakage, hago una iteracion 
        uobs_fid2 = self.mask*(uobs_fid - uecnn1) 

        data_red2 = np.zeros((1,self.npixels,self.npixels,4))
        data_red2[:,:,:,0] = qobs_fid2
        data_red2[:,:,:,1] = uobs_fid2
        data_red2[:,:,:,2] = self.mask
        data_red2[:,:,:,3] = self.variance_map

        qecnn2, uecnn2, qcnn2, ucnn2, ecnn2 = self.get_nn_outputs(data_red2, model2, map_rescale_factor_q2, dx, tfac)

        qobs_fid3 = self.mask*(qobs_fid - qecnn1 - qecnn2) 
        uobs_fid3 = self.mask*(uobs_fid - uecnn1 - uecnn2)

        if self.mask_mode == 1:

            data_red3 = np.zeros((1,self.npixels,self.npixels,4))
            data_red3[:,:,:,0] = qobs_fid3
            data_red3[:,:,:,1] = uobs_fid3
            data_red3[:,:,:,2] = self.mask
            data_red3[:,:,:,3] = self.variance_map

            qecnn3, uecnn3, qcnn3, ucnn3, ecnn3 = self.get_nn_outputs(data_red3, model3, map_rescale_factor_q3, dx, tfac)

            qobs_fid4 = self.mask*(qobs_fid - qecnn1 - qecnn2 - qecnn3) 
            uobs_fid4 = self.mask*(uobs_fid - uecnn1 - uecnn2 - uecnn3)

            qtotal = qcnn1 + qcnn2 + qcnn3
            utotal = ucnn1 + ucnn2 + ucnn3

            efft_tot, bfft_tot = utilities.transf_eb(qtotal, utotal, self.npixels, dx)
            ecnn_tot = np.fft.irfft2(efft_tot)/tfac

            efft_cnn4, bfft_cnn4 = utilities.transf_eb(qobs_fid4, uobs_fid4, self.npixels, dx)
            bcnn4 = np.fft.irfft2(bfft_cnn4)/tfac  
            ecnn4 = ecnn1 + ecnn2 + ecnn3

            data_red4 = np.zeros((1,self.npixels,self.npixels,4))
            data_red4[:,:,:,0] = bcnn4
            data_red4[:,:,:,1] = ecnn4
            data_red4[:,:,:,2] = self.mask
            data_red4[:,:,:,3] = self.variance_bmap

            bcnn5 = self.prediction_bdato(data_red4, modelb, map_rescale_factor_b, map_rescale_factor_e)

            prediction_b = bcnn5    
            prediction_e = ecnn_tot

            return prediction_e, prediction_b

        elif self.mask_mode == 0:

            qtotal = qcnn1 + qcnn2
            utotal = ucnn1 + ucnn2 

            efft_tot, bfft_tot = utilities.transf_eb(qtotal, utotal, self.npixels, dx)
            ecnn_tot = np.fft.irfft2(efft_tot)/tfac

            efft_cnn3, bfft_cnn3 = utilities.transf_eb(qobs_fid3, uobs_fid3, self.npixels, dx)
            bcnn3 = np.fft.irfft2(bfft_cnn3)/tfac  #este es el Bcnn final
            ecnn3 = ecnn1 + ecnn2 

            data_red3 = np.zeros((1,self.npixels,self.npixels,4))
            data_red3[:,:,:,0] = bcnn3
            data_red3[:,:,:,1] = ecnn3
            data_red3[:,:,:,2] = self.mask
            data_red3[:,:,:,3] = self.variance_bmap

            bcnn4 = self.prediction_bdato(data_red3, modelb, map_rescale_factor_b, map_rescale_factor_e)

            prediction_b = bcnn4    
            prediction_e = ecnn_tot

            return prediction_e, prediction_b


    def WF_red(self, qobs_fid, uobs_fid, factors, models):

        map_rescale_factor_q1 = factors["factor1"]
        map_rescale_factor_q2 = factors["factor2"]
        map_rescale_factor_q3 = factors["factor3"]
        map_rescale_factor_q4 = factors["factor4"]

        model1 = models[0]
        model2 = models[1]
        model3 = models[2]
        model4 = models[3]
        if self.mask_mode == 1: 
            model5 = models[4]
            map_rescale_factor_q5 = factors["factor5"]
    
        data_red = np.zeros((1,self.npixels,self.npixels,4))
        data_red[:,:,:,0] = qobs_fid
        data_red[:,:,:,1] = uobs_fid
        data_red[:,:,:,2] = self.mask
        data_red[:,:,:,3] = self.variance_map

        dx,_ = self.get_res()
        tfac = np.sqrt((dx*dx)/(self.npixels*self.npixels))

        qecnn1, uecnn1, qcnn1, ucnn1, ecnn1 = self.get_nn_outputs(data_red, model1, map_rescale_factor_q1, dx, tfac)

        qobs_fid2 = self.mask*(qobs_fid - qecnn1)   
        uobs_fid2 = self.mask*(uobs_fid - uecnn1) 

        data_red2 = np.zeros((1,self.npixels,self.npixels,4))
        data_red2[:,:,:,0] = qobs_fid2
        data_red2[:,:,:,1] = uobs_fid2
        data_red2[:,:,:,2] = self.mask
        data_red2[:,:,:,3] = self.variance_map

        qecnn2, uecnn2, qcnn2, ucnn2, ecnn2 = self.get_nn_outputs(data_red2, model2, map_rescale_factor_q2, dx, tfac)
        
        qobs_fid3 = self.mask*(qobs_fid - qecnn1 - qecnn2) 
        uobs_fid3 = self.mask*(uobs_fid - uecnn1 - uecnn2)

        data_red3 = np.zeros((1,self.npixels,self.npixels,4))
        data_red3[:,:,:,0] = qobs_fid3
        data_red3[:,:,:,1] = uobs_fid3
        data_red3[:,:,:,2] = self.mask
        data_red3[:,:,:,3] = self.variance_map

        qecnn3, uecnn3, qcnn3, ucnn3, ecnn3 = self.get_nn_outputs(data_red3, model3, map_rescale_factor_q3, dx, tfac)

        qobs_fid4 = self.mask*(qobs_fid - qecnn1 - qecnn2 - qecnn3) 
        uobs_fid4 = self.mask*(uobs_fid - uecnn1 - uecnn2 - uecnn3)

        data_red4 = np.zeros((1,self.npixels,self.npixels,4))
        data_red4[:,:,:,0] = qobs_fid4
        data_red4[:,:,:,1] = uobs_fid4
        data_red4[:,:,:,2] = self.mask
        data_red4[:,:,:,3] = self.variance_map

        qcnn4, ucnn4 = self.prediction(data_red4, model4, map_rescale_factor_q4) 

        if self.mask_mode == 1:

            efft_cnn4, bfft_cnn4 = utilities.transf_eb(qcnn4, ucnn4, self.npixels, dx)
            ecnn4 = np.fft.irfft2(efft_cnn4)/tfac
            bcnn4 = np.fft.irfft2(bfft_cnn4)/tfac
            qecnn4, uecnn4 = utilities.transf_qu_eb(ecnn4, bcnn4, self.npixels, dx, only_e=True, only_b=False) 

            qobs_fid5 = self.mask*(qobs_fid - qecnn1 - qecnn2 - qecnn3 - qecnn4) #este es el nuevo dato
            uobs_fid5 = self.mask*(uobs_fid - uecnn1 - uecnn2 - uecnn3 - uecnn4)

            data_red5 = np.zeros((1,self.npixels,self.npixels,4))
            data_red5[:,:,:,0] = qobs_fid5
            data_red5[:,:,:,1] = uobs_fid5
            data_red5[:,:,:,2] = self.mask
            data_red5[:,:,:,3] = self.variance_map

            qcnn5, ucnn5 = self.prediction(data_red5, model5, map_rescale_factor_q5)

            efft_cnn5, bfft_cnn5 = utilities.transf_eb(qcnn5, ucnn5, self.npixels, dx)
            ecnn5 = np.fft.irfft2(efft_cnn5)/tfac
            bcnn5 = np.fft.irfft2(bfft_cnn5)/tfac  #este es el Bcnn final

            prediction_b = bcnn5

            qtotal = qcnn1 + qcnn2 + qcnn3 + qcnn4  
            utotal = ucnn1 + ucnn2 + ucnn3 + ucnn4

            efft_tot, bfft_tot = utilities.transf_eb(qtotal, utotal, self.npixels, dx)
            ecnn_tot = np.fft.irfft2(efft_tot)/tfac

            prediction_e = ecnn_tot

            return prediction_e, prediction_b

        elif self.mask_mode == 0: 

            efft_cnn4, bfft_cnn4 = utilities.transf_eb(qcnn4, ucnn4, self.npixels, dx)
            ecnn4 = np.fft.irfft2(efft_cnn4)/tfac
            bcnn4 = np.fft.irfft2(bfft_cnn4)/tfac  #este es el Bcnn final

            prediction_b = bcnn4

            qtotal = qcnn1 + qcnn2 + qcnn3 #+ qcnn4 
            utotal = ucnn1 + ucnn2 + ucnn3 #+ ucnn4

            efft_tot, bfft_tot = utilities.transf_eb(qtotal, utotal, self.npixels, dx)
            ecnn_tot = np.fft.irfft2(efft_tot)/tfac

            prediction_e = ecnn_tot

            return prediction_e, prediction_b


    def El_fid(self, ell_flat, cl_plane, power, cl_bin): 

        bin_indices = np.digitize(ell_flat, self.bins) - 1

        El_sum = np.zeros(len(self.bins) - 1)
        El_count = np.zeros(len(self.bins) - 1)

        for bin_index in range(len(self.bins) - 1):
            mask = (bin_indices == bin_index)
            El_sum[bin_index] = np.sum(power[mask] / cl_plane[mask])
            El_count[bin_index] = np.sum(mask)

        El = 0.5 * El_sum / cl_bin

        return El

    #constant perturbation on angular spectrum in each mode of the bin
    def perturbation(self, cl_ang, k):   
        cl_ang_pert = np.zeros((len(self.bins)-1,len(cl_ang)))
    
        for j in range(len(self.bins) -1):    
            cl_ang_pert[j,:] =  cl_ang[:]
            cl_ang_pert[j,self.bins[j]:self.bins[j+1]] = cl_ang[self.bins[j]:self.bins[j+1]] + k*cl_ang[self.bins[j]:self.bins[j+1]]

        return cl_ang_pert, k


    # perturbation proportional to cl 
    def perturbation_plane(self, ell_flat, cl_plane, k):   
    
        cl_flat_pert_prop = np.zeros((len(self.bins)-1, len(cl_plane)))
        for j in range(len(self.bins)-1):
            cl_flat_pert_prop[j] = cl_plane[:]
            for i in range(len(cl_plane)):
                el = ell_flat[i]
                if el>=self.bins[j] and el<self.bins[j+1]:
                    cl_flat_pert_prop[j,i]=cl_plane[i] + k*cl_plane[i]

        return cl_flat_pert_prop, k


    #random perturbation
    def mode_perturbation(self, rfft_fiducial, ell_flat, cl_flat_pert):

        dx,_ = self.get_res()
        tfac = np.sqrt((dx*dx)/(self.npixels*self.npixels))

        #rfft_fiducial is a real_rfft
        rfft_shape = rfft_fiducial.shape
        rfft_reshape = np.reshape(rfft_fiducial, [rfft_shape[0]*rfft_shape[1]])
 
        rfft_pert = np.zeros((len(self.bins)-1,len(rfft_reshape)), dtype=complex) 
        rfft_pert[:,:] = rfft_reshape.copy()
    
        for i in range(len(ell_flat)):
            el = ell_flat[i]    
            for j in range(len(self.bins) -1):
                    if el>=self.bins[j] and el<self.bins[j+1]:
                        ran1 = np.random.normal(scale=np.sqrt(cl_flat_pert[j,i]/2))
                        ran2 = np.random.normal(scale=np.sqrt(cl_flat_pert[j,i]/2))
                        rfft_pert[j,i] = ran1 + ran2*1j
    
        signal_pert = np.zeros((self.nbins, self.npixels, self.npixels))
        for i in range(len(self.bins)-1):
            rfft_pert_reshape = np.reshape(rfft_pert[i,:], [rfft_shape[0], rfft_shape[1]])
            signal_pert[i,:,:] = np.fft.irfft2(rfft_pert_reshape/tfac)#tfac

        return rfft_pert, signal_pert

    def mode_perturbation_phase(self, rfft_fiducial, ell_flat, cl_flat_pert):

        #dx,_ = self.get_res()
        #tfac = np.sqrt((dx*dx)/(self.npixels*self.npixels))
    
        rfft_shape = rfft_fiducial.shape
        rfft_reshape = np.reshape(rfft_fiducial, [rfft_shape[0]*rfft_shape[1]])
 
        rfft_pert = np.zeros((len(self.bins)-1,len(rfft_reshape)), dtype=complex) #shape (25,len(ell_flat))
        rfft_pert[:,:] = rfft_reshape.copy()
    
        for i in range(len(ell_flat)):
            el = ell_flat[i]    
            for j in range(len(self.bins) -1):
                    if el>=self.bins[j] and el<self.bins[j+1]:
                        phase = np.arctan2(rfft_reshape[i].imag, rfft_reshape[i].real)
                        modulo = np.random.normal(scale=np.sqrt(cl_flat_pert[j,i]))
                        x = modulo*np.cos(phase)
                        y = modulo*np.sin(phase)
                        rfft_pert[j,i] = x + y*1j

        rfft_pert = np.reshape(rfft_pert, (len(self.bins)-1, rfft_shape[0], rfft_shape[1]))

        return rfft_pert#, seÃ±al_pert


#function that calculates noise bias term
    def noise_bias(self, nsamples_bias, factors, models, ell_flat, cl_ang_e, cl_ang_b, cl_plane_e, cl_plane_b, cl_bin_flat_e, cl_bin_flat_b):

        prediction_fid_e = np.zeros((nsamples_bias,self.npixels,self.npixels))
        prediction_fid_b = np.zeros((nsamples_bias,self.npixels,self.npixels))

        for i in range(nsamples_bias):
            qobs_fid, uobs_fid, qsky_fid, usky_fid, esky_fid, bsky_fid =self.make_map_noseed(cl_ang_e, cl_ang_b)
            if self.modelb == 1: 
                prediction_e, prediction_b = self.WF_red_bdato(qobs_fid, uobs_fid, factors, models)
            else:
                prediction_e, prediction_b = self.WF_red(qobs_fid, uobs_fid, factors, models)

            prediction_fid_e[i] = prediction_e[0,:,:]
            prediction_fid_b[i] = prediction_b[0,:,:]
    
        El_muchas_e = np.zeros((nsamples_bias,len(self.bins)-1))
        El_muchas_b = np.zeros((nsamples_bias,len(self.bins)-1))

        for i in range(nsamples_bias):
            mapa_pred_e = prediction_fid_e[i,:,:]
            mapa_pred_b = prediction_fid_b[i,:,:]
            power_map_e = self.power(mapa_pred_e)
            power_map_b = self.power(mapa_pred_b)
            El_muchas_e[i] = self.El_fid(ell_flat, cl_plane_e, power_map_e, cl_bin_flat_e) 
            El_muchas_b[i] = self.El_fid(ell_flat, cl_plane_b, power_map_b, cl_bin_flat_b) 
    
        bl_e = np.mean(El_muchas_e, axis=0)
        bl_b = np.mean(El_muchas_b, axis=0)

        return bl_e, bl_b


#function that calculates the fisher matrix
    def fisher(self, qdata_pert, udata_pert, Elfid_e, Elfid_b, c, ell_flat, cl_plane_e, cl_plane_b, cl_bin_flat_e, cl_bin_flat_b, factors, models):

        fisher_matrix_e = np.zeros((len(self.bins)-1,len(self.bins)-1))
        fisher_matrix_b = np.zeros((len(self.bins)-1,len(self.bins)-1))

        prediction_e_pert = np.zeros((len(self.bins)-1,self.npixels,self.npixels))  
        prediction_b_pert = np.zeros((len(self.bins)-1,self.npixels,self.npixels))

        for j in range(len(self.bins)-1):

            if self.modelb == 1: 
                prediction_e_pert[j], prediction_b_pert[j] = self.WF_red_bdato(qdata_pert[j], udata_pert[j], factors, models)
            else: 
                prediction_e_pert[j], prediction_b_pert[j] = self.WF_red(qdata_pert[j], udata_pert[j], factors, models)

            power_map_e_pert = self.power(prediction_e_pert[j,:,:]) 
            power_map_b_pert = self.power(prediction_b_pert[j,:,:]) 

            El_e_pert = self.El_fid(ell_flat, cl_plane_e, power_map_e_pert, cl_bin_flat_e)  
            El_b_pert = self.El_fid(ell_flat, cl_plane_b, power_map_b_pert, cl_bin_flat_b)

            fisher_matrix_e[:,j] = (El_e_pert - Elfid_e)/(c*cl_bin_flat_e[j])
            fisher_matrix_b[:,j] = (El_b_pert - Elfid_b)/(c*cl_bin_flat_b[j])

        
        return fisher_matrix_e, fisher_matrix_b



    

















 
            
















