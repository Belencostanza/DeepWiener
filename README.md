# DeepWiener
Neural network to simulate the Wiener Filter (WF) for Cosmic Microwave Background (CMB) polarization maps with inhomogeneous noise applied. In addition, the power spectrum estimation algorithm is included 


# Description of the codes: 

Brief overview of the codes inclued in ``source``:
- ``CG_inho.py``: Conjugate Gradient algorithm for Wiener Filter calculation.
- ``wf_noise.py``: Module used to generate flat sky maps with Nifty7.
- ``make_dataset.py``: Module that creates the dataset for each iteration.
- ``losses.py``: Module with the loss functions used to train the neural network. 
- ``train_functions.py``: DeepWiener neural network and training.
- ``PowerSpectrum.py``: Module to estimate the noise bias term and fisher matrix of the power spectrum.

# Requirements: 

- Tensorflow 2.X
- CAMB (https://camb.readthedocs.io/en/latest/index.html)
- Nifty7 (https://ift.pages.mpcdf.de/nifty/user/index.html)


# Usage: 

The implemented codes here perform the WF reconstruction of CMB polarization maps with inhomogeneous noise applied. The inhomogeneous noise is presented by a variance map extracted from Planck. Two masks are presented with different sky fraction. 

The dictionary input include these parameters:

- ``modelb``: put equal to 0 if only the loss function J(Q,U) is used, put equal to 1 if the loss function J(B) is used.
- ``iter``: iteration number. Until 4 for the Mask1 case and until 5 for the Mask2 case. Put equal to 0 if ``modelb`` is equal to 1. If the iteration number is higher than 1 be aware of having the results of the previous iterations.    
- ``name_data_folder``: path where the dataset is stored.
- ``name_train``, ``name_valid``, ``name_test``: filenames of the training, validation and test set that will be saved in the data folder.
- ``nsims_train``, ``nsims_valid``, ``nsims_test``: number of maps for each set.
- ``mask``: put equal to 0 if the Mask1 is used, put equal to 1 if the Mask2 is used.
- ``name_result_folder``: path where the results are stored.
- ``factor``: normalization factor for the dataset in each iteration.
- ``name_model``: name of the models that will be saved in the results folder in (.h5 format).
- ``name_path``: checkpoints name.
- ``filters``: number of filters used for the convolution in each layer.
- ``namequ_file_bl``, ``nameb_file_bl``: name of the noise bias calculated with ``PowerSpectrum.py``, for models trained with J(Q,U) and J(B), that will be saved in the results folder.
- ``namequ_fisher_mean``, ``nameb_fisher_mean``: name of the fisher matrix calculated with ``PowerSpectrum.py``, for models trained with J(Q,U) and J(B), that will be saved in the results folder.
- ``nsamples_bias``, ``nsamples_fisher``: number of simulations used to estimate the noise bias term and the fisher matrix.
- ``nbins``: number of bins.

To run the software: 

1. Edit the input dictionary ``input_inho_mask.dict`` with the desired example.
2. Run ``run_dataset.py /path/input_inho_mask.dict`` to create the dataset.
3. Run ``train_functions.py /path/input_inho_mask.dict`` to train the neural network.
4. Run ``eval_model.py /path/input_inho_mask.dict`` to evaluate the trained models and obtain the WF predictions.
5. Run ``fisher_and_noise_bias.py /path/input_inho_mask.dict`` to calculate the noise bias and the fisher matrix for the power spectrum estimation.

# Contact 

Feel free to contact me at belen@fcaglp.unlp.edu.ar for comments, questions and suggestions.










