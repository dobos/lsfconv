import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import TruncatedSVD

from .psf import Psf

class PcaPsf(Psf):
    """
    Computes the convolution of a data vector with a kernel expressed as
    a linear combination of basis function (derived from PCA) and the expansion
    coefficients are tabulated and interpolated as a function of wavelength.

    This kind of PSF representation uses a fixed kernel size with a fixed
    wavelength grid and fixed kernel wavelength steps.
    """

    __SUPPRESS_WARNING_DWAVE = False
    __SUPPRESS_WARNING_NORM = False
    __SUPPRESS_WARNING_SIZE = False

    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, PcaPsf):
            self.wave = None            # wavelength grid
            self.wave_edges = None
            self.dwave = None
            self.mean = None            # mean kernel
            self.eigs = None            # eigenvalues
            self.eigv = None            # eigenvectors
            self.pc = None              # coeffs as functions of wavelength
        else:
            self.wave = orig.wave
            self.wave_edges = orig.wave_edges
            self.dwave = orig.dwave
            self.mean = orig.mean
            self.eigs = orig.eigs
            self.eigv = orig.eigv
            self.pc = orig.pc

        if self.wave is not None and self.pc is not None:
            self.init_ip()
        else:
            self.pc_ip = None

    def init_ip(self):
        # Interpolate PCs
        self.pc_ip = interp1d(self.wave, self.pc.T, bounds_error=False, fill_value=(self.pc[0], self.pc[-1]))

    @staticmethod
    def from_psf(source_psf, wave, dwave=None, size=None, s=slice(None), normalize=True, truncate=None):
        """
        Precomputes the kernel as a function of wavelength on the provided grid
        using truncated PCA.
        """
        
        tr = np.s_[:truncate] if truncate is not None else np.s_[:]

        w, dw, k, idx, shift = source_psf.eval_kernel(wave, dwave=dwave, size=size, s=s, normalize=normalize)
        
        M = np.mean(k, axis=0)
        X = k - M

        if X.shape[0] > 1000:
            C = np.matmul(X.T, X)
            U, S, Vt = np.linalg.svd(C)
            S = np.sqrt(S)
        elif truncate is not None and truncate > 100 and truncate < size // 4:
            svd = TruncatedSVD(n_components=truncate * 2)
            svd.fit(X)
            Vt = svd.components_
            S = svd.singular_values_
        else:
            U, S, Vt = np.linalg.svd(X)
            
        PC = np.matmul(X, Vt[tr].T)

        pca_psf = PcaPsf()
        pca_psf.wave = w
        pca_psf.dwave = dw
        pca_psf.mean = M
        pca_psf.eigs = S[tr]
        pca_psf.eigv = Vt[tr]
        pca_psf.pc = PC

        pca_psf.init_ip()
        
        return pca_psf

    def eval_kernel_impl(self, wave, dwave=None, size=None, s=slice(None), normalize=True):
        """
        Return the matrix necessary to calculate the convolution with a varying kernel 
        using the direct matrix product method. This function is for compatibility with the
        base class and not used by the `convolve` implementation itself.
        """

        if not np.array_equal(wave, self.wave):
            raise ValueError("Wave grid doesn't match tabulated grid.")

        if dwave is not None:
            self.warn('DWAVE')

        if normalize is not None:
            self.warn('NORM')

        shift = self.eigv.shape[-1] // 2

        # idx will hold negative values because the tabulated kernels are usually
        # computed beyond the wave grid.
        idx = (np.arange(self.eigv.shape[-1]) - shift) + np.arange(wave.size)[:, np.newaxis]
        
        w = self.wave[s]
        dw = None if self.dwave is None else self.dwave[s]
        pc = self.pc_ip(w).T
        k = self.mean + np.matmul(pc, self.eigv)
        
        if normalize:
            k = self.normalize(k)

        # Return 0 for shift since we have the kernel for the entire wavelength range
        return w, dw, k, idx[s], 0

    def get_optimal_size(self, wave, tol=1e-5):
        raise NotImplementedError()

    def convolve(self, wave, values, errors=None, size=None, normalize=None, mode='valid'):
        if size is not None:
            self.warn('SIZE')

        if normalize is not None:
            self.warn('NORM')

        # Convolve value vector with each eigenfunction, than take linear combination with
        # the wave-dependent principal components

        if isinstance(values, np.ndarray):
            vv = [ values ]
        else:
            vv = values
        
        if isinstance(errors, np.ndarray):
            ee = [ errors ]
        else:
            ee = errors

        if mode == 'valid':
            shift = (self.eigv.shape[1] // 2)
            w = wave[shift:-shift]
        elif mode == 'same':
            shift = 0
            w = wave
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        pc = self.pc_ip(w).T

        rv = []
        for v in vv:
            m = np.convolve(v, self.mean, mode=mode)
            c = np.empty((m.shape[0], self.eigv.shape[0]))
            for i in range(self.eigv.shape[0]):
                c[..., i] = np.convolve(v, self.eigv[i], mode=mode)
            rv.append(m + np.sum(pc * c, axis=-1))

        if ee is not None:
            re = []
            for e in ee:
                m = np.convolve(e**2, self.mean**2, mode=mode)
                c = np.empty((m.shape[0], self.eigv.shape[0]))
                for i in range(self.eigv.shape[0]):
                    c[..., i] = np.convolve(e**2, self.eigv[i]**2, mode='valid')
                re.append(np.sqrt(m + np.sum(pc**2 * c, axis=-1)))
        else:
            re = None

        if isinstance(values, np.ndarray):
            rv = rv[0]

        if isinstance(errors, np.ndarray):
            re = re[0]

        return w, rv, re, shift

    def warn(self, id):
        if id == 'DWAVE' and not PcaPsf.__SUPPRESS_WARNING_DWAVE:
            logging.warning('PCA PSF does not support overriding dwave.')
            PcaPsf.__SUPPRESS_WARNING_DWAVE = True
        elif id == 'NORM' and not PcaPsf.__SUPPRESS_WARNING_NORM:
            logging.warning('PCA PSF does not support renormalizing the precomputed kernel.')
            PcaPsf.__SUPPRESS_WARNING_NORM = True
        elif id == 'SIZE' and not PcaPsf.__SUPPRESS_WARNING_SIZE:
            logging.warning('PCA PSF does not support overriding kernel size.')
            PcaPsf.__SUPPRESS_WARNING_SIZE = True