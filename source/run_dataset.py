#author: Belén Costanza 

import sys, platform, os
import numpy as np
import matplotlib.pyplot as plt

from read_input import load_config
import make_dataset as dd
import time
#import utilities


if len(sys.argv) != 2:
    print("Usage: python your_script.py input_file")
    sys.exit(1)

input_file = sys.argv[1]

#Dictionary with the parameters of the input file:
input_data = load_config(input_file)

for key, value in input_data.items():
    print(f'{key} = {value}')


data_folder = input_data["name_data_folder"]
result_folder = input_data["name_result_folder"]

iteration = int(input_data["iter"])
modelb = int(input_data['modelb'])

if iteration == 1: 
    filename_train = input_data["name_train1"] 
    filename_valid = input_data["name_valid1"]
    name_test_noise = input_data["name_test_noise1"]
elif iteration == 2: 
    filename_train = input_data["name_train2"] 
    filename_valid = input_data["name_valid2"]
    name_test_noise = input_data["name_test_noise2"]
elif iteration == 3: 
    filename_train = input_data["name_train3"] 
    filename_valid = input_data["name_valid3"]
    name_test_noise = input_data["name_test_noise3"]
elif iteration == 4: 
    filename_train = input_data["name_train4"] 
    filename_valid = input_data["name_valid4"]
    name_test_noise = input_data["name_test_noise4"]
elif iteration == 5: 
    filename_train = input_data["name_train5"] 
    filename_valid = input_data["name_valid5"]
    name_test_noise = input_data["name_test_noise5"]

if modelb == 1 and iteration == 0: 
    filename_train = input_data["name_trainb"] 
    filename_valid = input_data["name_validb"]
    name_test_noise = input_data["name_test_noiseb"]

name_test_noiseless = input_data["name_test_noiseless"]

nsims_train = int(input_data["nsims_train"])
nsims_valid = int(input_data["nsims_valid"])
nsims_test = int(input_data["nsims_test"])

npad = int(input_data["npad"])
npixels = int(input_data["npixels"])
Lsize = int(input_data["Lsize"])
mask_mode = int(input_data["mask"])

variance_map = np.load(data_folder + 'variance_QU_planck.npy')
variance_bmap = np.load(data_folder + 'variance_mapB_planck.npy')

masks = np.load(data_folder + 'masks_QU.npy')
if mask_mode == 0: 
    mask = masks[0]
else:
    mask = masks[1]

#make dictionary 
name_model1 = input_data['name_model1']
name_model2 = input_data['name_model2']
name_model3 = input_data['name_model3']
name_model4 = input_data['name_model4']
name_modelb = input_data['name_modelb']

name_models = {"name_model1": result_folder + name_model1, "name_model2": result_folder + name_model2, "name_model3": result_folder + name_model3, "name_model4": result_folder + name_model4, "name_modelb": result_folder + name_modelb}
print(name_models)

factor1 = float(input_data['factor1'])
factor2 = float(input_data['factor2'])
factor3 = float(input_data['factor3'])
factor4 = float(input_data['factor4'])
factorb = float(input_data['factorb'])

factors = {"factor1": factor1, "factor2": factor2, "factor3": factor3, "factor4": factor4, "factorb": factorb}

make_data = dd.make_dataset(nsims_train, nsims_valid, nsims_test, data_folder + filename_train, data_folder + filename_valid, variance_map, variance_bmap, mask, npad, npixels, Lsize)

#start to create dataset
t0 = time.time()
make_data.get_maps_records(iteration, name_models, factors, modelb, mask_mode)
make_data.get_maps_test(data_folder + name_test_noise, data_folder + name_test_noiseless, iteration, name_models, factors, modelb, mask_mode)
t1 = time.time()
print('Time to create dataset:', t1-t0)
