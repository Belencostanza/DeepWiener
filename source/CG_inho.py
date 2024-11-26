import numpy as np
import matplotlib.pyplot as plt

import healpy as hp
import nifty7 as ift

import scipy
import wf_noise as wf

import os
import sys
import camb
from camb import model, initialpower

#import nifty8 as ift
from scipy import interpolate

import copy



##### code based on quicklens https://github.com/dhanson/quicklens/tree/843c308dfce8f99d860ba9212aecc79f68ed963a  


class spec_camb():
    def __init__(self, cls):
        
        #self.cls = cls
        [self.cle, self.clb] = cls 
    def copy(self):
        
        return spec_camb([self.cle.copy(), self.clb.copy()])


class cov_s():
    def __init__(self, cl, lmax):

        self.lmax = lmax
        zs = np.zeros(self.lmax+1)
        clmat = np.zeros( (self.lmax+1,2,2))
        clmat[:,0,0] = getattr(cl, 'cle', zs.copy())
        clmat[:,1,1] = getattr(cl, 'clb', zs.copy())
        
        self.clmat = clmat
    
    def inverse(self):
        """ return a new cov_s object, containing the 3x3 matrix pseudo-inverse of this one, multipole-by-multipole. """
        ret = copy.deepcopy(self)
        for l in range(0, self.lmax+1):
            ret.clmat[l,:,:] = np.linalg.pinv( self.clmat[l] )
        return ret
    
    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret.clmat += other.clmat
        return ret
    
    #que soporte la opracion con un teb (efft, bfft)
    def __mul__(self, other):
        if np.isscalar(other):
            ret = copy.deepcopy(self)
            ret.clmat *= other
            return ret
        
        elif ( hasattr(other, 'efft') and hasattr(other, 'bfft') and hasattr(other, 'get_ell') ):
            teb = other 
            ret = teb.copy()
            ell = teb.get_ell()
        
            def fftxcl(fft, cl):
            #np.inter(x, xp, fp)
            # x-coordinates at which to evaluate the interpolated values, xp x-coordinates of data points(cl)
            # fp y-coordinates of data points, same length as xp
                return fft * np.interp( ell.flatten(), np.arange(0, len(cl)), cl, right=0 ).reshape(fft.shape)
        
        #ret.efft[:,:]  = fftxcl( teb.tfft, self.clmat[:,1,0] ) + fftxcl( teb.efft, self.clmat[:,1,1] ) + fftxcl( teb.bfft, self.clmat[:,1,2] )
        #ret.bfft[:,:]  = fftxcl( teb.tfft, self.clmat[:,2,0] ) + fftxcl( teb.efft, self.clmat[:,2,1] ) + fftxcl( teb.bfft, self.clmat[:,2,2] )
        
            ret.efft[:,:]  = fftxcl( teb.efft, self.clmat[:,0,0] ) + fftxcl( teb.bfft, self.clmat[:,0,1] )
            ret.bfft[:,:]  = fftxcl( teb.efft, self.clmat[:,1,0] ) + fftxcl( teb.bfft, self.clmat[:,1,1] )        
        
            return ret


class sinv_class(cov_s):
    def __init__(self, cl, lmax):
        
        super(sinv_class, self).__init__(cl, lmax)
        self.clmat = self.inverse().clmat
        
    def hashdict(self):
        return { 'clmat' : self.clmat }


class tebfft():
    def __init__(self, nx, dx, ffts=None):
        """ class which contains the FFT of a tqumap. E- and B-mode polarization. """
        #super( tebfft, self ).__init__(nx, dx, ny=ny, dy=dy)
        self.nx = nx
        self.dx = dx
        
        if ffts is None:
            self.efft = np.zeros( (self.nx, int(self.nx/2+1)), dtype=complex )
            self.bfft = np.zeros( (self.nx, int(self.nx/2+1)), dtype=complex )
        else:
            [self.efft, self.bfft] = ffts

        #assert( (self.nx, self.nx/2+1) == self.tfft.shape )
        assert( (self.nx, int(self.nx/2+1)) == self.efft.shape )
        assert( (self.nx, int(self.nx/2+1)) == self.bfft.shape )
        
    def copy(self):
        return tebfft( self.nx, self.dx,
                       [self.efft.copy(), self.bfft.copy()])
        
    def get_lxly(self):
        """ returns the (lx, ly) pair associated with each Fourier mode in E, B. """
        return np.meshgrid( np.fft.fftfreq( self.nx, self.dx )[0:int(self.nx/2+1)]*2.*np.pi,
                            np.fft.fftfreq( self.nx, self.dx )*2.*np.pi )
    
    def get_ell(self):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode in E, B. """
        lx, ly = self.get_lxly()
        return np.sqrt(lx**2 + ly**2)
        
        
    def get_tqu(self):
        """ returns the tqumap given by the inverse Fourier transform of this object. """
        lx, ly = self.get_lxly()
        tpi  = 2.*np.arctan2(lx, -ly)

        tfac = np.sqrt((self.nx * self.nx) / (self.dx * self.dx))
    
            #tmap = np.fft.irfft2(self.tfft) * tfac
        qmap = np.fft.irfft2(np.cos(tpi)*self.efft - np.sin(tpi)*self.bfft) * tfac
        umap = np.fft.irfft2(np.sin(tpi)*self.efft + np.cos(tpi)*self.bfft) * tfac

        return qumap( self.nx, self.dx, [qmap, umap])
    
    def __mul__(self, other):
        if ( np.isscalar(other) or ( (type(other) == np.ndarray) and
                                     (getattr(other, 'shape', None) == self.tfft.shape) ) ):
            return tebfft( self.nx, self.dx,
                           ffts=[self.efft * other,
                                 self.bfft * other])
    
    
    def __imul__(self, other):
        if ( np.isscalar(other) or ( (type(other) == np.ndarray) and
                                     (getattr(other, 'shape', None) == self.efft.shape) ) ):
            #self.tfft *= other
            self.efft *= other
            self.bfft *= other
        return self
    
    def __add__(self, other):
        return tebfft( self.nx, self.dx,
                       [self.efft + other.efft, self.bfft + other.bfft])
    
    def __iadd__(self, other):
        #assert( self.compatible(other) )
        self.efft += other.efft; self.bfft += other.bfft
        return self
    
    def __sub__(self, other):
        return tebfft( self.nx, self.dx,
                       [self.efft - other.efft, self.bfft - other.bfft])
   


class qumap():
    def __init__(self, nx, dx, maps=None):
        """ class which contains polarization (Q, U) maps. """
        
        self.nx = nx
        self.dx = dx
        #super( qumap, self ).__init__(nx, dx)
        if maps is None:
            self.qmap = np.zeros( (self.nx, self.nx) )
            self.umap = np.zeros( (self.nx, self.nx) )
        else:
            [self.qmap, self.umap] = maps

        assert( (self.nx, self.nx) == self.qmap.shape )
        assert( (self.nx, self.nx) == self.umap.shape )

    def copy(self):
        return qumap( self.nx, self.dx,
                       [self.qmap.copy(), self.umap.copy()])
    
    def __mul__(self, other):
        ret = self.copy()
        ret.qmap *= other.qmap
        ret.umap *= other.umap
        return ret
    
    def get_teb(self):
        """ return a tebfft object containing the fourier transform of the Q,U maps. """
        ret = tebfft( self.nx, self.dx)
        
        lx, ly = ret.get_lxly()
        tpi  = 2.*np.arctan2(lx, -ly)

        tfac = np.sqrt((self.dx * self.dx) / (self.nx * self.nx))
        qfft = np.fft.rfft2(self.qmap) * tfac
        ufft = np.fft.rfft2(self.umap) * tfac
        
        #ret.tfft[:] = np.fft.rfft2(self.tmap) * tfac
        ret.efft[:] = (+np.cos(tpi) * qfft + np.sin(tpi) * ufft)
        ret.bfft[:] = (-np.sin(tpi) * qfft + np.cos(tpi) * ufft)
        return ret

    #def degrade(self, fac):
    #    ret = qumap( int(self.nx/fac), self.dx*fac)

    #    for i in range(0,fac):
    #        for j in range(0, fac):
    #            ret.map += self.map[i::fac,j::fac]

    #    return ret 


class ninv_class():
    
    def __init__(self, ninv, noise_pix, lmax):
        self.ninv = ninv
        self.noise_pix = noise_pix
        self.lmax = lmax
        
    def hashdict(self):
        ret = {}

        ret['ninv'] = self.ninv.hashdict()

        return ret
        
    def mult_tqu(self, tqu):
        """ returns Y' * N^{-1} * tqu, where tqu is a maps.tqumap object. """

        ret  = (tqu * self.ninv).get_teb() #* self.transf

        ret *= (1. / (ret.dx * ret.dx))
        return ret
    
    def mult_teb(self, teb):
        """ returns (Y' * N^{-1} * Y) * teb, where teb is a maps.tebfft object. """
        
        #ret = ( (teb * self.transf).get_tqu() * self.ninv ).get_teb() * self.transf
        ret = (teb.get_tqu()*self.ninv).get_teb() 
        ret *= (1. / (ret.dx * ret.dx))
        return ret
    
    def get_fl(self):
        """ return an approximation of [Y' N^{-1} Y] which is diagonal in Fourier space. """
        nee = np.mean(1./self.noise_pix)
        clee = nee * np.ones(self.lmax+1)
        clbb = nee * np.ones(self.lmax+1)
        nls = [clee, clbb]
        ninv = cov_s(spec_camb(nls), self.lmax)
        #ninv = cov_s( util.dictobj( { 'clee' : nee * np.ones(lmax+1), 'clbb' : nee * np.ones(lmax+1) } ) )  
        
        return ninv * (1./(self.ninv.dx*self.ninv.dx))

    #def degrade(self, tfac):
    #    """ returns a copy of this filter object appropriate for a map with half the resolution / number of this pixels. """
    #    return ninv_class(self.ninv.degrade(tfac), self.noise_pix, self.lmax)

            
    
class fw_op():
    """ returns [S^{-1} + (Y' * N^{-1} * Y)] * teb. """ 
    def __init__(self, sinv, ninv):
        self.sinv = sinv
        self.ninv = ninv
        
    def __call__(self, teb):
        return self.calc(teb)
    
    def calc(self, teb):
        return (self.sinv * teb + self.ninv.mult_teb(teb))
  

class pre_op_diag():
    """ returns an approximation of the operation [Y^{t} N^{-1} Y + S^{-1}]^{-1} which is diagonal in Fourier space, represented by a maps.tebfft object. """
    def __init__(self, sinv, ninv):
        self.filt = (sinv + ninv.get_fl()).inverse()

    def __call__(self, talm):
        return self.calc(talm)
        
    def calc(self, teb):
        return self.filt * teb
    
class dot_op():
    """ defines a dot product for two maps.tebfft objects using their cross-spectra up to a specified lmax. """
    def __init__(self, lmax=None):
        self.lmax = lmax
        
    def __call__(self, teb1, teb2):
        return np.sum( (teb1.efft * np.conj(teb2.efft) + teb1.bfft * np.conj(teb2.bfft) ).flatten().real )
        

class monitor_basic():
    """ a simple class for monitoring a conjugate descent iteration. """
    def __init__(self, dot_op, iter_max=np.inf, eps_min=1.0e-10):
        self.dot_op   = dot_op
        self.iter_max = iter_max
        self.eps_min  = eps_min
        #self.logger   = logger

        #self.watch = util.stopwatch()

    def criterion(self, it, soltn, resid):
        delta = self.dot_op( resid, resid )
        
        if (it == 0):
            self.d0 = delta

        #if (self.logger is not None): self.logger( it, np.sqrt(delta/self.d0), watch=self.watch,
                                                 #  soltn=soltn, resid=resid )

        if (it >= self.iter_max) or (delta <= self.eps_min**2 * self.d0):
            return True

        return False

    def __call__(self, *args):
        return self.criterion(*args)

class cache_mem(dict):
    def __init__(self):
        pass

    def store(self, key, data):
        [dTAd_inv, searchdirs, searchfwds] = data
        self[key] = [dTAd_inv, searchdirs, searchfwds]

    def restore(self, key):
        return self[key]

    def remove(self, key):
        del self[key]

    def trim(self, keys):
        assert( set(keys).issubset(self.keys()) )
        for key in (set(self.keys()) - set(keys)):
            del self[key]


def tr_cg(i):
    """ truncation and restart function for conjugate gradients. """
    return i-1


#Ax es fwd_op
def cd_solve(x, b, Ax, pre_ops, dot_op, criterion, tr=tr_cg, cache=cache_mem(), roundoff=25):

    it = 0
    residual = b - Ax(x)
    n_pre_ops = len(pre_ops)
    searchdirs = [op(residual) for op in pre_ops]

    while criterion(it, x, residual) == False:
        searchfwds = [Ax(searchdir) for searchdir in searchdirs]
        deltas     = [dot_op(searchdir, residual) for searchdir in searchdirs]
    
        dTAd = np.zeros( (n_pre_ops, n_pre_ops) )
        for ip1 in range(0, n_pre_ops):
            for ip2 in range(0, ip1+1):
                dTAd[ip1, ip2] = dTAd[ip2, ip1] = dot_op(searchdirs[ip1], searchfwds[ip2])
        dTAd_inv = np.linalg.inv(dTAd)
    
        alphas = np.dot( dTAd_inv, deltas )
        for (searchdir, alpha) in zip( searchdirs, alphas ):
            x += searchdir * alpha
     
        cache.store( it, [dTAd_inv, searchdirs, searchfwds] )

        # update residual
        it += 1
        if ( np.mod(it, roundoff) == 0 ):
            residual = b - Ax(x)
        else:
            for (searchfwd, alpha) in zip( searchfwds, alphas ):
                residual -= searchfwd * alpha
            
        searchdirs = [pre_op(residual) for pre_op in pre_ops]
    

        # orthogonalize w.r.t. previous searches.
        prev_iters = range( tr(it), it )
        
        for titer in prev_iters:
            [prev_dTAd_inv, prev_searchdirs, prev_searchfwds] = cache.restore(titer)

            for searchdir in searchdirs:
                proj  = [ dot_op(searchdir, prev_searchfwd) for prev_searchfwd in prev_searchfwds ]
                betas = np.dot( prev_dTAd_inv, proj )

                for (beta, prev_searchdir) in zip( betas, prev_searchdirs):
                    searchdir -= prev_searchdir * beta

        # clear old keys from cache
        cache.trim( range( tr(it+1), it ) )

    return it



