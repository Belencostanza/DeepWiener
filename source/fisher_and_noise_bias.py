#author: Belén Costanza

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import scipy 
from scipy import interpolate
import time

import nifty7 as ift
import utilities

from read_input import load_config
import PowerSpectrum as PP


####################### read-config #####################

if len(sys.argv) != 2:
        print("Usage: python your_script.py input_file")
        sys.exit(1)

input_file = sys.argv[1]

input_data = load_config(input_file)

data_folder = input_data["name_data_folder"]
result_folder = input_data["name_result_folder"]

npad = int(input_data["npad"])
npixels = int(input_data["npixels"])
Lsize = int(input_data["Lsize"])
    
nsamples_bias = int(input_data["nsamples_bias"])  ## 100
nsamples_fisher = int(input_data["nsamples_fisher"])## 100
mask_mode = int(input_data["mask"])
modelb = int(input_data["modelb"])

if modelb == 0:
    print('qu models')

    name_archivo_bl_e = input_data["namequ_file_bl_e"]
    name_archivo_bl_b = input_data["namequ_file_bl_b"]

    name_mean_e  = input_data["namequ_fisher_mean_e"]
    name_mean_b  = input_data["namequ_fisher_mean_b"]

    name_array_e = input_data["namequ_fisher_array_e"]
    name_array_b = input_data["namequ_fisher_array_b"]

else: #modelb == 1
    print('b models')

    name_archivo_bl_e = input_data["nameb_file_bl_e"]
    name_archivo_bl_b = input_data["nameb_file_bl_b"]

    name_mean_e  = input_data["nameb_fisher_mean_e"]
    name_mean_b  = input_data["nameb_fisher_mean_b"]

    name_array_e = input_data["nameb_fisher_array_e"]
    name_array_b = input_data["nameb_fisher_array_b"]

variance_map = np.load(data_folder + 'variance_QU_planck.npy')
variance_bmap = np.load(data_folder + 'variance_mapB_planck.npy')

masks = np.load(data_folder + 'masks_QU.npy')
if mask_mode == 0: 
    mask = masks[0]
else:
    mask = masks[1]

lmin = int(input_data["lmin"])
lmax = int(input_data["lmax"])
nbins = int(input_data["nbins"])


factor1 = float(input_data['factor1'])
factor2 = float(input_data['factor2'])
factor3 = float(input_data['factor3'])
factor4 = float(input_data['factor4'])
factorb = float(input_data['factorb'])
factore = float(input_data['factore'])

factors = {"factor1": factor1, "factor2": factor2, "factor3": factor3, "factor4": factor4, "factorb": factorb, "factore": factore}

if mask_mode == 1:
    factor5 = float(input_data['factor5'])
    factors = {"factor1": factor1, "factor2": factor2, "factor3": factor3, "factor4": factor4, "factor5": factor5, "factorb": factorb, "factore": factore}

name_model1 = input_data['name_model1']
name_model2 = input_data['name_model2']
name_model3 = input_data['name_model3']
name_model4 = input_data['name_model4']
name_model5 = input_data['name_model5']
name_modelb = input_data['name_modelb']

name_models = {"name_model1": result_folder + name_model1, "name_model2": result_folder + name_model2, "name_model3": result_folder + name_model3, "name_model4": result_folder + name_model4, "name_model5": result_folder + name_model5, "name_modelb": result_folder + name_modelb}

cl_ang_e, cl_ang_b = utilities.signal_spectrum() #fiducial angular power spectrum
_,_, bin_edges = utilities.compute_bins(lmin,lmax,nbins) 


######################### call class ##################################

ps = PP.PowerSpectrum(Lsize, npixels, npad, nbins, bin_edges, modelb, mask_mode, mask, variance_map, variance_bmap)

ell_flat, cl_plane_e = ps.flat_spectrum(cl_ang_e) #fiducial flat power spectrum
_, cl_plane_b = ps.flat_spectrum(cl_ang_b)
ell_flat_real, cl_plane_real_e = ps.flat_spectrum_real(cl_ang_e)
_, cl_plane_real_b = ps.flat_spectrum_real(cl_ang_b)

el_bin_flat_e, cl_bin_flat_e,_ = ps.bineado(ell_flat, cl_plane_e)  #binned flat power spectrum
el_bin_flat_b, cl_bin_flat_b,_ = ps.bineado(ell_flat, cl_plane_b)


#load models once
print('loading models')
models = ps.get_models(factors, name_models)

#################### necessary functions ##############################


def perturbation(cte):
    cl_flat_pert_prop_e, c = ps.perturbation_plane(ell_flat_real, cl_plane_real_e, k=cte)
    cl_flat_pert_prop_b, _ = ps.perturbation_plane(ell_flat_real, cl_plane_real_b, k=cte)
    return cl_flat_pert_prop_e, cl_flat_pert_prop_b, c

    
def fiducial(nsamples_fisher, noise_array, cl_bin_e, cl_bin_b):

    esky_fid = np.zeros((nsamples_fisher, npixels, npixels))
    bsky_fid = np.zeros((nsamples_fisher, npixels, npixels))

    prediction_e = np.zeros((nsamples_fisher, npixels, npixels))
    prediction_b = np.zeros((nsamples_fisher, npixels, npixels))

    El_fiducial_e = np.zeros((nsamples_fisher,len(bin_edges)-1))
    El_fiducial_b = np.zeros((nsamples_fisher,len(bin_edges)-1))


    for i in range(nsamples_fisher):
        _,_, qsky_fid, usky_fid, esky_fid[i], bsky_fid[i] =ps.make_map_noseed(cl_ang_e, cl_ang_b)

        qobs_fid = mask*(qsky_fid + noise_array)
        uobs_fid = mask*(usky_fid + noise_array)

        if modelb == 1:
            prediction_e[i], prediction_b[i] = ps.WF_red_bdato(qobs_fid, uobs_fid, factors, models) #EWF, BWF
        else:
            #print('entre')
            prediction_e[i], prediction_b[i] = ps.WF_red(qobs_fid, uobs_fid, factors, models) #EWF, BWF
    
        power_map_fid_e = ps.power(prediction_e[i,:,:]) #El_fid(ell_flat, cl_plane, power_map, bin_edges,cl_bin)
        power_map_fid_b = ps.power(prediction_b[i,:,:]) 
        El_fiducial_e[i] = ps.El_fid(ell_flat, cl_plane_e, power_map_fid_e, cl_bin_e) #aca el El_fid se calcula sobre los fiduciales utilizando todos los modos (256,256)
        El_fiducial_b[i] = ps.El_fid(ell_flat, cl_plane_b, power_map_fid_b, cl_bin_b) #aca el El_fid se calcula sobre los fiduciales utilizando todos los modos (256,256)

    return esky_fid, bsky_fid, El_fiducial_e, El_fiducial_b
   

def fisher_parallel(esky_fid, bsky_fid, El_fiducial_e, El_fiducial_b, noise_array, cl_flat_pert_prop_e, cl_flat_pert_prop_b, c):

    _,rfft_fiducial_e = ps.power_real(esky_fid) #necesito unicamente para hacer la perturbacion que el rfft fiducial sea real
    _,rfft_fiducial_b = ps.power_real(bsky_fid)
    rfft_pert_e = ps.mode_perturbation_phase(rfft_fiducial_e, ell_flat_real, cl_flat_pert_prop_e)
    rfft_pert_b = ps.mode_perturbation_phase(rfft_fiducial_b, ell_flat_real, cl_flat_pert_prop_b)

    #print(np.shape(rfft_pert_e))
    dx,_ = ps.get_res()
    tfac = np.sqrt((dx*dx)/(npixels*npixels))

    rfft_pert_q, rfft_pert_u = utilities.transf_qu(rfft_pert_e, rfft_pert_b, npixels, dx)
    qsky_pert = np.fft.irfft2(rfft_pert_q/tfac)
    usky_pert = np.fft.irfft2(rfft_pert_u/tfac)

    qdata_pert = mask*(qsky_pert+noise_array) #shape (25,256,256)
    udata_pert = mask*(usky_pert+noise_array)

    Elfid_e = El_fiducial_e
    Elfid_b = El_fiducial_b
    
    fisher_value_e, fisher_value_b = ps.fisher(qdata_pert, udata_pert, Elfid_e, Elfid_b, c, ell_flat, cl_plane_e, cl_plane_b, cl_bin_flat_e, cl_bin_flat_b, factors, models)
    
    return fisher_value_e, fisher_value_b
    


################################# calculate noise bias term and fisher matrix ###################################


print('start bias calculation')
t0 = time.time()
bl_e, bl_b = ps.noise_bias(nsamples_bias, factors, models, ell_flat, cl_ang_e, cl_ang_b, cl_plane_e, cl_plane_b, cl_bin_flat_e, cl_bin_flat_b)
t1 = time.time()
np.save(result_folder + name_archivo_bl_e, bl_e)
np.save(result_folder + name_archivo_bl_b, bl_b)
print('finish, time to calculate noise bias with 100 maps:', t1-t0)
        

print('start fisher calculation')

k = np.array([-0.1, 0.1])
        
noise_array = ps.map_noise(cl_ang_e)

esky_fid, bsky_fid, El_fiducial_e, El_fiducial_b = fiducial(nsamples_fisher, noise_array, cl_bin_flat_e, cl_bin_flat_b)
        
for cte in k:
    print('fisher calculation:', cte)

    t2 = time.time()
            
    cl_flat_pert_prop_e, cl_flat_pert_prop_b, c = perturbation(cte)
                        
    fisher_array_e = np.zeros((nsamples_fisher, len(bin_edges)-1, len(bin_edges)-1))
    fisher_array_b = np.zeros((nsamples_fisher, len(bin_edges)-1, len(bin_edges)-1))

    for j in range(nsamples_fisher):
        fisher_value_e, fisher_value_b = fisher_parallel(esky_fid[j], bsky_fid[j], El_fiducial_e[j], El_fiducial_b[j], noise_array, cl_flat_pert_prop_e, cl_flat_pert_prop_b, cte)
        fisher_array_e[j] = fisher_value_e
        fisher_array_b[j] = fisher_value_b
                
    fisher_mean_e = np.sum(fisher_array_e, axis=0)/(nsamples_fisher)
    fisher_mean_b = np.sum(fisher_array_b, axis=0)/(nsamples_fisher)
    
    t3 = time.time()
    print('time to calculate fisher matrix with 2000 maps:', t3-t2)
   
    np.save(result_folder + name_mean_e+str(cte)+'.npy', fisher_mean_e)
    np.save(result_folder + name_array_e+str(cte)+'.npy', fisher_array_e)
    np.save(result_folder + name_mean_b+str(cte)+'.npy', fisher_mean_b)
    np.save(result_folder + name_array_b+str(cte)+'.npy', fisher_array_b)
    

