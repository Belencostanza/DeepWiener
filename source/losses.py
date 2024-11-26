#author: Belén Costanza

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import scipy.interpolate

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import camb
from camb import model, initialpower

import utilities


tf.keras.backend.set_floatx('float32')



class loss_functions:

    def __init__(self, Lsize, npixels, mask, variance_map, variance_bmap, factor_q, factor_b):

        """ Class for the implemented loss functions 

        Params:

        Lsize = size map in deg (20)
        npixels = number of pixels (256)
        mask = it can be Mask1 or Mask2 defined in the paper 
        variance_map = variance in each pixel for QU noise
        variance_bmap = variance in each pixel for B noise
        factor_q = rescale factor for QU maps
        factor_b = rescale factor for B maps

        """

        self.Lsize = Lsize
        self.npixels = npixels
        self.mask = mask
        self.variance_map = variance_map
        self.variance_bmap = variance_bmap

        self.map_rescale_factor_q = factor_q
        self.map_rescale_factor_b = factor_b

    def get_res(self):

        #angular_resolution_deg = self.Lsize/self.npixels
        #angular_resolution_arcmin = angular_resolution_deg*60
        rad2arcmin = 180.*60./np.pi
        d2r = np.pi/180.
        dx = self.Lsize*d2r / float(self.npixels)

        return dx 


    def bl(self):
        lmax = 7000
        fwhm_arcmin = 0.001
        ls = np.arange(0,lmax+1)
        return np.exp( -(fwhm_arcmin * np.pi/180./60.)**2 / (16.*np.log(2.)) * ls*(ls+1.) )


    def get_power_spectrum(self, modelb=0):

        lmax = 7000

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.1)
        pars.set_for_lmax(7000, lens_potential_accuracy=1)
        pars.WantTensors = True

        results = camb.get_results(pars)

        powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

        totCL          = powers['total']
        unlensedCL     = powers['unlensed_scalar']
        unlensed_Tot   = powers['unlensed_total']
        lensed_SC      = powers['lensed_scalar']
        tensorCL       = powers['tensor']
        lens_potential = powers['lens_potential']

        ls = np.arange(unlensedCL.shape[0])
        factor = ls*(ls+1)/2./np.pi

        cl_EE_unlensed = np.copy(totCL[:,1])
        cl_BB_unlensed = np.copy(totCL[:,2])

        cl_EE_unlensed[2:] = cl_EE_unlensed[2:] / factor[2:]
        cl_BB_unlensed[2:] = cl_BB_unlensed[2:] / factor[2:]

        cl_array_e = np.copy(cl_EE_unlensed)
        cl_array_b = np.copy(cl_BB_unlensed)


        cl_ee = cl_array_e * (self.map_rescale_factor_q**2)  ## * (beam**2)
        cl_bb = cl_array_b * (self.map_rescale_factor_q**2)  ## * (beam**2)

        if modelb == 1:
            print('factor_b')
            cl_bb = cl_array_b * (self.map_rescale_factor_b**2)  
            
        dx = self.get_res()

        lx,ly=np.meshgrid( np.fft.fftfreq( self.npixels, dx )[0:int(self.npixels/2+1)]*2.*np.pi,np.fft.fftfreq( self.npixels, dx )*2.*np.pi )
        l = np.sqrt(lx**2 + ly**2)
        ell_flat = l.flatten()
        ellmask_nonzero = np.logical_and(ell_flat >= 2, ell_flat <= lmax)
        ell_ql = np.arange(0,cl_ee.shape[0])
    
        cl_ee[0:2] = cl_ee[2]
        interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(cl_ee[1:]), kind='linear',fill_value=0,bounds_error=False)
        cl_ee = np.zeros(ell_flat.shape[0])
        cl_ee[1:] = np.exp(interp(np.log(ell_flat[1:])))
        cl_ee[0] = cl_ee[1]
    
        cl_bb[0:2] = cl_bb[2]
        interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(cl_bb[1:]), kind='linear',fill_value=0,bounds_error=False)
        cl_bb = np.zeros(ell_flat.shape[0])
        cl_bb[1:] = np.exp(interp(np.log(ell_flat[1:])))
        cl_bb[0] = cl_bb[1]

        return cl_ee, cl_bb

    def noise_spectrum(self, nlev_p): 
        #nlev_p in uk.arcmin
        lmax = 7000

        nl = (nlev_p*np.pi/180./60.)**2 #/ beam**2
        nl_spec = np.ones_like(nl, shape=(7001))
        nl_spec[:] = nl

        dx = self.get_res()

        lx,ly=np.meshgrid( np.fft.fftfreq( self.npixels, dx )[0:int(self.npixels/2+1)]*2.*np.pi,np.fft.fftfreq( self.npixels, dx )*2.*np.pi )
    #lx,ly=np.meshgrid( np.fft.fftfreq( nx, 1/float(nx) ),np.fft.fftfreq( nx, 1/float(nx) ) )
        l = np.sqrt(lx**2 + ly**2)
        ell_flat = l.flatten()#*72.

        ell_ql = np.arange(0,nl_spec.shape[0])

        nl_flat = np.copy(nl_spec)
        nl_flat[0:2] = nl_spec[2]
        interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(nl_flat[1:]), kind='linear',fill_value=0,bounds_error=False)
        nl_flat = np.zeros(ell_flat.shape[0])
        nl_flat[1:] = np.exp(interp(np.log(ell_flat[1:])))
        nl_flat[0] = nl_flat[1]

        return nl_flat

    def inverse_cl(self, data, modes_e, modelb):

        if modes_e == True:
            cl_ee,_ = self.get_power_spectrum()
            data = tf.math.divide_no_nan(data, cl_ee)
            return data
        if modes_e == False:
            _,cl_bb = self.get_power_spectrum(modelb)
            #nl = noise_spectrum()
            norm = cl_bb #+ nl
            data = tf.math.divide_no_nan(data,norm)
            return data

    def transf_eb(self, qmap, umap, nx, dx):
        
        lx,ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi, np.fft.fftfreq( nx, dx )*2.*np.pi )
        tpi  = 2.*np.arctan2(lx, -ly)
        tfac = np.sqrt((dx * dx) / (nx * nx))
        qfft = tf.signal.rfft2d(qmap)*tfac
        ufft = tf.signal.rfft2d(umap)*tfac
        
        efft = (+np.cos(tpi) * qfft + np.sin(tpi) * ufft)
        bfft = (-np.sin(tpi) * qfft + np.cos(tpi) * ufft)
        
        return efft, bfft

    def transf_onlye_onlyb(self, efft, bfft, only_e = True, only_b = False):

        dx = self.get_res()

        lx,ly = np.meshgrid( np.fft.fftfreq( self.npixels, dx )[0:int(self.npixels/2+1)]*2.*np.pi, np.fft.fftfreq( self.npixels, dx )*2.*np.pi )  
        tpi  = 2.*np.arctan2(lx, -ly)
        tfac = np.sqrt((dx * dx) / (self.npixels * self.npixels))

        if only_e == True: 
            qmap = tf.signal.irfft2d(np.cos(tpi)*efft)/tfac
            umap = tf.signal.irfft2d(np.sin(tpi)*efft)/tfac
        elif only_b == True: 
            qmap = tf.signal.irfft2d(-np.sin(tpi)*bfft)/tfac
            umap = tf.signal.irfft2d(np.cos(tpi)*bfft)/tfac

        return qmap, umap


    def realspace_loss(self, y_true, y_pred):
    #square map in real space and multiply by mask to remove masked pixels. weight the other by pixel noise variance.
        y_true_Q = y_true[:,:,:,0]
        y_pred_Q = y_pred[:,:,:,0]
        y_true_U = y_true[:,:,:,1]
        y_pred_U = y_pred[:,:,:,1]

        loss_Q = (y_pred_Q - y_true_Q)*(y_pred_Q - y_true_Q) / (self.variance_map*self.map_rescale_factor_q**2)
        loss_Q = loss_Q*self.mask
        loss_U = (y_pred_U - y_true_U)*(y_pred_U - y_true_U) / (self.variance_map*self.map_rescale_factor_q**2)
        loss_U = loss_U*self.mask

        loss = tf.reduce_mean(loss_Q) + tf.reduce_mean(loss_U)

        return loss


    def fourier_loss(self, y_true, y_pred):

        rmap_Q = y_pred[:,:,:,0]
        rmap_U = y_pred[:,:,:,1]
        #ecnn = y_true[:,:,:,2]

        dx = self.get_res()

        tfac = np.sqrt((dx*dx)/(self.npixels*self.npixels))

        efft_res, bfft_res = self.transf_eb(rmap_Q, rmap_U, self.npixels, dx)    
        efft_shape = efft_res.get_shape().as_list()
        bfft_shape = bfft_res.get_shape().as_list()
        #efft_tot = tf.signal.rfft2d(ecnn)*tfac
        #print('efft_tot', efft_tot)
        #print('efft_res', efft_res)
        #print(np.shape(efft_tot))
        #print(np.shape(efft_res))
        efft = efft_res
        power_e = tf.math.real((efft * tf.math.conj(efft)))
        power_b = tf.math.real((bfft_res * tf.math.conj(bfft_res)))
        power_e = tf.reshape(power_e,[-1,efft_shape[1]*efft_shape[2]])
        power_b = tf.reshape(power_b,[-1,bfft_shape[1]*bfft_shape[2]])

        #weight by signal power spectrum
        power_e = self.inverse_cl(power_e, modes_e=True, modelb=0)
        power_b = self.inverse_cl(power_b, modes_e=False, modelb=0)

        loss = tf.reduce_mean(power_e) + tf.reduce_mean(power_b)
#    loss= tf.reduce_mean(power_b) #solo pongo termino de B

        return loss


    def fourier_loss_iter(self, y_true, y_pred):

        rmap_Q = y_pred[:,:,:,0]
        rmap_U = y_pred[:,:,:,1]
        ecnn = y_true[:,:,:,2]

        dx = self.get_res()

        tfac = np.sqrt((dx*dx)/(self.npixels*self.npixels))

        efft_res, bfft_res = self.transf_eb(rmap_Q, rmap_U, self.npixels, dx)    
        efft_shape = efft_res.get_shape().as_list()
        bfft_shape = bfft_res.get_shape().as_list()
        efft_tot = tf.signal.rfft2d(ecnn)*tfac
        #print('efft_tot', efft_tot)
        #print('efft_res', efft_res)
        #print(np.shape(efft_tot))
        #print(np.shape(efft_res))
        efft = efft_tot + efft_res
        power_e = tf.math.real((efft * tf.math.conj(efft)))
        power_b = tf.math.real((bfft_res * tf.math.conj(bfft_res)))
        power_e = tf.reshape(power_e,[-1,efft_shape[1]*efft_shape[2]])
        power_b = tf.reshape(power_b,[-1,bfft_shape[1]*bfft_shape[2]])

        #weight by signal power spectrum
        power_e = self.inverse_cl(power_e, modes_e=True, modelb=0)
        power_b = self.inverse_cl(power_b, modes_e=False, modelb=0)

        loss = tf.reduce_mean(power_e) + tf.reduce_mean(power_b)
#    loss= tf.reduce_mean(power_b) #solo pongo termino de B

        return loss
    

    def loss_wiener_j3(self, y_true, y_pred):
        #real space noise weighted difference
        term1 = self.realspace_loss(y_true, y_pred)
        #fourier space on the input map
        term2 = self.fourier_loss(y_true, y_pred)

        loss = term1 + term2
        return loss

    def loss_wiener_j3_iter(self, y_true, y_pred):
        #real space noise weighted difference
        term1 = self.realspace_loss(y_true, y_pred)
        #fourier space on the input map
        term2 = self.fourier_loss_iter(y_true, y_pred)

        loss = term1 + term2

        return loss
        

    def loss_b(self, y_true, y_pred):
        y_true_B = y_true[:,:,:,0]
        y_pred_B = y_pred[:,:,:,0]
        loss = tf.reduce_mean((y_pred_B - y_true_B)*(y_pred_B - y_true_B))
        return loss

    def loss_b_j3(self, y_true, y_pred): 

    #(Bcnn-Bobs)^{2}/sigma + BcnnBcnn*/ClB
        y_true_B = y_true[:,:,:,0]  #Bobs
        y_pred_B = y_pred[:,:,:,0]  #Bcnn

        loss_B = (y_pred_B - y_true_B)*(y_pred_B - y_true_B) / (self.variance_bmap*self.map_rescale_factor_b**2)
        #loss_B = loss_B*self.mask
        loss1 = tf.reduce_mean(loss_B)

        dx = self.get_res()

        tfac = np.sqrt((dx * dx) / (self.npixels * self.npixels))
        bfft_pred = tf.signal.rfft2d(y_pred_B)*tfac

        bfft_shape = bfft_pred.get_shape().as_list()
        power_b = tf.math.real((bfft_pred * tf.math.conj(bfft_pred)))
        power_b = tf.reshape(power_b,[-1,bfft_shape[1]*bfft_shape[2]])

    #weight by signal power spectrum
        power_b = self.inverse_cl(power_b, modes_e=False, modelb=1)

        loss2 = tf.reduce_mean(power_b)

        loss = loss1 + loss2

        return loss


    def loss_eb(self, y_true, y_pred):


        y_true_B = y_true[:,:,:,0]
    #print(y_true_B)
    
        y_pred_Q = y_pred[:,:,:,0]
        y_pred_U = y_pred[:,:,:,1]

        dx = self.get_res()
        tfac = 1./(np.sqrt((dx * dx) / (self.npixels * self.npixels)))

        efft, bfft = self.transf_eb(y_pred_Q, y_pred_U, self.npixels, dx) #toma las predicciones y calcula 
        emap = tf.signal.irfft2d(efft)*tfac
        y_pred_B = tf.signal.irfft2d(bfft)*tfac

        loss = tf.reduce_mean((y_pred_B - y_true_B)*(y_pred_B - y_true_B))  

        return loss














