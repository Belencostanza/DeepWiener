import sys, os

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import nifty7 as ift

#import healpy as hp

class wfilter_TT:
    def __init__(self, npix,resolution,signal_dic, noise_dic, mask_dic=None):
        """    
        :param signal_dic: signal dictionary containing power spectrum starting at l=0
        :param noise_dic: noise dictionary power spectrum starting at l=0 if method specified in type is harmonic
        :if method is from var needs to input variance. 
        :param npix: number of pixles in each dimension of the map
        :param resoluton: pixel size in arcminutes
        """
 
        ## Parameters of the map
        self.resolution=resolution/60.0*np.pi/180.
        self.npix=npix
        self.size_map=npix*self.resolution
        self.kfun=2.0*np.pi/self.size_map
        
        ## Initialize nifty variables
        # Defines spaces and transformations
        self.initialize_nifty()
        
        ## Define mask
        if mask_dic== None:
            self.Mask = ift.ScalingOperator(self.s_space, 1.)
            self.R = self.HT # @ ift.create_harmonic_smoothing_operator((h_space,), 0, 0.02)
        else:
            self.mask_dic=mask_dic
            self.init_mask()

        
        # signal quantities
        self.signal_dic=signal_dic
        self.init_signal()

        # noise quantities
        self.noise_dic=noise_dic
        self.init_noise()        
    
        ## Create WF 
        self.create_wf()
        
    
        return

    def initialize_nifty(self):
        # Define spaces and transformations
        self.s_space = ift.RGSpace([self.npix,self.npix])
        self.h_space = self.s_space.get_default_codomain()
        self.HT = ift.HartleyOperator(self.h_space,self.s_space)     
        self.kvals=self.h_space.get_unique_k_lengths()
        
        return
    
    def init_mask(self):
        
        mask=self.mask_dic['mask_array']
        mask = ift.Field.from_raw(self.s_space, mask)

        self.Mask= ift.DiagonalOperator(mask)
        
        ## Response
        self.R = self.Mask @ self.HT # @ ift.create_harmonic_smoothing_operator((h_space,), 0, 0.02)
        #

    
    def init_signal(self):
        
        ## Interpolation function for power spectra
        cls_tt=self.signal_dic['cls']
        ls=np.arange(len(cls_tt))

        ## This needs fixing. 
        ## Does not like zero power in some modes. 
        ## It falis when trying to do the weiner filter if noise was set to using the varaince in real space.
        ## Probably related to the zero mode?
        ## In the case of harmonic noise I am also putting non-zero noise in the zero mode. 
        self.angpower_signal = interpolate.interp1d(ls[2:], cls_tt[2:],bounds_error=False,fill_value=np.min(cls_tt[2:])) 
                
        ## Covariance matrices
        self.Sh = ift.create_power_operator(self.h_space, power_spectrum=self.signal_power)
        ## Contribution to Weiner filter curvature
        self.D_inv_S=self.Sh.inverse
        
        return
        
    def init_noise(self):
        
        noise_method_dic={'harmonic':self.init_noise_harmonic, 'from var':self.init_noise_fromvar,
                         'from varmap':self.init_noise_fromvarmap}
        
        init_func=noise_method_dic[self.noise_dic['type']]
        init_func()
        
        return

    def init_noise_harmonic(self):
        
        ## Interpolation function for power spectra
        nls_tt=self.noise_dic['cls']
        ls=np.arange(len(nls_tt))
        
        
        ## Note that I am putting noise in the zero mode as well
        ## Otherwise I was running into problems at the time of the Weiner filter.  
        
        self.angpower_noise = interpolate.interp1d(ls[2:], nls_tt[2:],
                                                   bounds_error=False,fill_value=np.mean(nls_tt[2:]))        
        ## Covariance matrices
        self.Nh = ift.create_power_operator(self.h_space, power_spectrum=self.noise_power)
        self.N = ift.SandwichOperator.make(self.R.adjoint, self.Nh)
        #self.N=self.HT@self.Nh@self.HT.adjoint
        ## Contribution to Weiner filter curvature
        self.D_inv_N=self.Nh.inverse  
        
        ## Define map making routine
        self.make_noise_map=self.make_noise_map_harmonic

        return

    def init_noise_fromvar(self):
        
        ## Interpolation function for power spectra
        sigma2=self.noise_dic['sigma2_noise']
        print('Noise variance',sigma2)
        ## Covariance matrices
        self.N = ift.ScalingOperator(self.s_space, sigma2)
        ## Contribution to Weiner filter curvature
        self.D_inv_N=self.R.adjoint @ self.N.inverse @ self.R 
        
        ## Define map making routine
        self.make_noise_map=self.make_noise_map_from_var

        return
    
    def init_noise_fromvarmap(self):
        
        ## Map with variance
        varmap=self.noise_dic['varmap']
        #print('Average Noise variance',np.mean(varmap))
        self.varmap=self.array_pixelsp2field(varmap)
        ## Covariance matrices
        self.N = ift.DiagonalOperator(self.varmap,self.s_space)
        ## Contribution to Weiner filter curvature
        self.D_inv_N=self.R.adjoint @ self.N.inverse @ self.R 
        
        ## Define map making routine
        self.make_noise_map=self.make_noise_map_from_varmap

        return
       
    def create_wf(self):
            
        ## Inversion method
        self.IC = ift.GradientNormController(iteration_limit=50000,
                                    tol_abs_gradnorm=0.1)
        #Curvature
        #self.D_inv = self.R.adjoint @ self.N.inverse @ self.R + self.S_inv ## If noise is defined in real space
        self.D_inv = self.D_inv_N + self.D_inv_S
        
        self.D=ift.InversionEnabler(self.D_inv, self.IC, self.D_inv_S).inverse
        return

    def signal_power(self,k):
        l=k*self.kfun
        return self.angpower_signal(l)/self.size_map**2
    
    def noise_power(self,k):
        l=k*self.kfun
        return self.angpower_noise(l)/self.size_map**2
    
    def WF(self,map):
        j=self.R.adjoint_times(self.N.inverse_times(map))
        return self.HT(self.D(j))
    
    def make_signal_map(self):
        sh = self.Sh.draw_sample_with_dtype(dtype=np.float64)
        return self.R(sh)
        
    def make_noise_map_harmonic(self):
        nh = self.Nh.draw_sample_with_dtype(dtype=np.float64)
        return self.R(nh)

    def make_noise_map_from_var(self):
        sigma=self.noise_dic['sigma2_noise']**0.5
        n = ift.Field.from_random(domain=self.s_space, random_type='normal',
                          std=sigma, mean=0)        
        return n

    def make_noise_map_from_varmap(self):
        n=self.N.draw_sample_with_dtype(dtype=float, from_inverse=False)
        
        return n

    def make_map(self):
        signal = self.make_signal_map()
        noise = self.make_noise_map()
        return signal+noise
    
    def measure_cl(self,map,binbounds=None):
        
        ## Measures power spectrum of map
        ## Can also provide the bounds for the bins in which to measure the power spectrum in binbounds

        
        power_data = ift.power_analyze(self.HT.inverse(map),binbounds=binbounds).val
        kvals=ift.PowerSpace(self.h_space,binbounds=binbounds).k_lengths
        
        cls=power_data*(self.size_map)**2
        ls=kvals*self.kfun
        
        return ls,cls
    
    def array_pixelsp2field(self,array):
        return ift.Field.from_raw(self.s_space,np.asarray(array))

        
    
    
    
