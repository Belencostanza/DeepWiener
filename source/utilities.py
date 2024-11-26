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


####### necessary functions   #########

def signal_spectrum():

    #use CMB to simulate the power spectrum

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

    lmax=7000
    ls = np.arange(lmax+1)
    factor = ls*(ls+1)/2./np.pi
    #lmax=7000

    cl_EE_unlensed = np.copy(totCL[:lmax+1,1])
    cl_BB_unlensed = np.copy(totCL[:lmax+1,2])

    cl_EE_unlensed[2:] = cl_EE_unlensed[2:] / factor[2:]
    cl_BB_unlensed[2:] = cl_BB_unlensed[2:] / factor[2:]

    cl_EE_for_map = np.copy(cl_EE_unlensed)
    cl_BB_for_map = np.copy(cl_BB_unlensed)

    #cl_TT_for_map[lmax+1:len(cl_TT_unlensed)] = 0.
    
    return cl_EE_for_map, cl_BB_for_map


def power(map, dx, nx):

    tfac = np.sqrt((dx*dx)/(nx*nx))
    fft = np.fft.rfft2(map[:,:])*tfac
    fft_shape = fft.shape 

    return fft

def transf_qu(power_e, power_b, npixels, dx):
    
    lx, ly = np.meshgrid( np.fft.fftfreq( npixels, dx )[0:int(npixels/2+1)]*2.*np.pi, np.fft.fftfreq(npixels, dx)*2.*np.pi )
    tpi  = 2.*np.arctan2(lx, -ly)#*180/np.pi
    tfac = 1/(np.sqrt((dx * dx) / (npixels * npixels)))
    power_q = np.cos(tpi)*power_e - np.sin(tpi)*power_b
    power_u = np.sin(tpi)*power_e + np.cos(tpi)*power_b
    
    return power_q, power_u

def transf_eb(qmap, umap, nx, dx):
       
    tfac = np.sqrt((dx*dx)/(nx*nx))
    qfft = np.fft.rfft2(qmap[:,:])*tfac
    ufft = np.fft.rfft2(umap[:,:])*tfac
    lx,ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi, np.fft.fftfreq( nx, dx )*2.*np.pi ) 
    #lx,ly=np.meshgrid( np.fft.fftfreq( nx, 1/float(nx) ),np.fft.fftfreq( nx, 1/float(nx) ) )
    tpi  = 2.*np.arctan2(lx, -ly) 

    efft = (+np.cos(tpi) * qfft + np.sin(tpi) * ufft)
    bfft = (-np.sin(tpi) * qfft + np.cos(tpi) * ufft)

    return efft, bfft


def transf_qu_eb(emap, bmap, nx, dx, only_e = False, only_b = False):
    
    lx,ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi, np.fft.fftfreq( nx, dx )*2.*np.pi )  
    tpi  = 2.*np.arctan2(lx, -ly)
    tfac = np.sqrt((dx * dx) / (nx * nx))
    efft = np.fft.rfft2(emap)*tfac
    bfft = np.fft.rfft2(bmap)*tfac
    
    if only_e == True: 
        qmap = np.fft.irfft2(np.cos(tpi)*efft)/tfac
        umap = np.fft.irfft2(np.sin(tpi)*efft)/tfac
    elif only_b == True: 
        qmap = np.fft.irfft2(-np.sin(tpi)*bfft)/tfac
        umap = np.fft.irfft2(np.cos(tpi)*bfft)/tfac
    else:
        qmap = np.fft.irfft2(np.cos(tpi)*efft - np.sin(tpi)*bfft)/tfac
        umap = np.fft.irfft2(np.sin(tpi)*efft + np.cos(tpi)*bfft)/tfac

    return qmap, umap


def periodic_padding(images,npad):
    if len(images.shape)==4:
        images = np.pad(images,pad_width=((0,0),(npad,npad),(npad,npad),(0,0)),mode='wrap')
    if len(images.shape)==3:
        images = np.pad(images,pad_width=((npad,npad),(npad,npad),(0,0)),mode='wrap')
    return images


def prediction(data_q, data_u, mask, model, factor, variance_map, npad):

    data = np.concatenate((data_q.reshape(1,256,256,1), data_u.reshape(1,256,256,1), mask.reshape(1,256,256,1), variance_map.reshape(1,256,256,1)), axis=-1)
    images_test = periodic_padding(data, npad)
    images_test[:,:,:,[0,1]] *= factor

    result = model.predict(images_test)
    result = result/factor

    qcnn = result[:,:,:,0]
    ucnn = result[:,:,:,1]

    return qcnn, ucnn


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   #,array[idx]

def compute_bins(lmin, lmax, Nbins):

    ls = np.arange(lmax+1)

    num_modes = np.zeros(lmax+1)
    cumulative_num_modes = np.zeros(lmax+1)

    bin_edges = np.zeros(Nbins+1)
    bin_edges[0] = lmin
    
    cumulative = 0
    for i in range(lmin,lmax+1):
        num_modes[i] = 2*i +1
        
        cumulative += num_modes[i]
        cumulative_num_modes[i] = cumulative

            
    Num_modes_total = num_modes.sum()
    print("Total number of modes in (l_min,l_max) = ", Num_modes_total)   
    Num_modes_per_bin = Num_modes_total / Nbins
    print("Number of modes in each bin = ", Num_modes_per_bin)

    for i in range(1,Nbins+1):
        
        #Num_modes_per_bin*i #cumulative modes up to bin "i"
        
        bin_edges[i] = find_nearest(cumulative_num_modes, Num_modes_per_bin*i)
    
    bin_edges = np.asarray(bin_edges,int)
   
    return Num_modes_per_bin, cumulative_num_modes, bin_edges


def powerauto(map, nx, dx):
    tfac = np.sqrt((dx*dx)/(nx*nx))
    fft = np.fft.rfft2(map[:,:])*tfac
    fft_shape = fft.shape
    power = np.real(fft*np.conj(fft))
    power_reshape = np.reshape(power, [fft_shape[0]*fft_shape[1]]) #coef de fourier mapa
    return power_reshape    

def power_cross(map1, map2, nx, dx):
    tfac = np.sqrt((dx*dx)/(nx*nx))
    fft1 = np.fft.rfft2(map1[:,:])*tfac
    fft2 = np.fft.rfft2(map2[:,:])*tfac
    fft_shape = fft1.shape 
    power = np.real(fft1*np.conj(fft2))
    power_reshape = np.reshape(power, [fft_shape[0]*fft_shape[1]]) #coef de fourier mapa
    return power_reshape

def cross_correlation(ell_flat,power_cross, power_wf, power_cnn,bins):
    
    bin_indices = np.digitize(ell_flat, bins, right=False)
        
    count = np.zeros(len(bins) -1)
    coef_corr = np.zeros((len(bins)-1))
    coef_wf = np.zeros((len(bins))-1)
    coef_cnn = np.zeros((len(bins))-1)
    
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        count[i - 1] = np.sum(mask)
        coef_corr[i - 1] = np.sum(power_cross[mask])
        coef_wf[i-1] = np.sum(power_wf[mask])
        coef_cnn[i-1] = np.sum(power_cnn[mask])
    
    coef_corr = coef_corr/count
    coef_wf = coef_wf/count
    coef_cnn = coef_cnn/count
    
    rl = coef_corr/(coef_wf*coef_cnn)**(1/2.)
    return rl



def bin_power(ell, cl, bins): 
 
    bin_indices = np.digitize(ell, bins, right=False)

    # Initialize arrays to store results
    count = np.zeros(len(bins) - 1)
    cl_bin_sum = np.zeros(len(bins) - 1)
    el_med_sum = np.zeros(len(bins) - 1)


    # Calculate sum of ell values and cl values for each bin
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        count[i - 1] = np.sum(mask)
        cl_bin_sum[i - 1] = np.sum(cl[mask])
        el_med_sum[i - 1] = np.sum(ell[mask] * cl[mask])

    # Calculate the binned results
    el_med = (el_med_sum / cl_bin_sum).astype(int)
    cl_bin = cl_bin_sum / count

    return el_med, cl_bin, count


def power_spectrum_flat(cl_ee_camb, cl_bb_camb, nx, dx):


    lx,ly=np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( nx, dx )*2.*np.pi )
    #lx,ly=np.meshgrid( np.fft.fftfreq( nx, 1/float(nx) ),np.fft.fftfreq( nx, 1/float(nx) ) )
    l = np.sqrt(lx**2 + ly**2)
    ell_flat = l.flatten()#*72.

    ell_ql = np.arange(0,cl_ee_camb.shape[0])

    cl_ee = np.copy(cl_ee_camb)
    cl_ee[0:2] = cl_ee_camb[2]
    interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(cl_ee[1:]), kind='linear',fill_value=0,bounds_error=False)
    cl_ee = np.zeros(ell_flat.shape[0])
    cl_ee[1:] = np.exp(interp(np.log(ell_flat[1:])))
    cl_ee[0] = cl_ee_camb[1]
    
    cl_bb = np.copy(cl_bb_camb)
    cl_bb[0:2] = cl_bb_camb[2]
    interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(cl_bb[1:]), kind='linear',fill_value=0,bounds_error=False)
    cl_bb = np.zeros(ell_flat.shape[0])
    cl_bb[1:] = np.exp(interp(np.log(ell_flat[1:])))
    cl_bb[0] = cl_bb[1]

    return ell_flat, cl_ee, cl_bb

