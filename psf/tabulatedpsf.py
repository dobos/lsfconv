import logging
import numpy as np

from .psf import Psf

class TabulatedPsf(Psf):
    """
    Computes the convolution of a data vector with a kernel tabulated as
    a function of wavelength.

    This kind of PSF representation uses a fixed kernel size with a fixed
    wavelength grid and fixed kernel wavelength steps.
    """

    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, TabulatedPsf):
            self.wave = None            # wavelength grid
            self.wave_edges = None
            self.dwave = None
            self.kernel = None
        else:
            self.wave = orig.wave
            self.wave_edges = orig.wave_edges
            self.dwave = orig.dwave
            self.kernel = orig.kernel

    def eval_kernel_impl(self, wave, dwave=None, size=None, s=slice(None), normalize=True):
        """
        Return the tabulated kernel, regardless of the value of `wave` and `size`,
        these are for compatibility only.
        """

        # TODO: do we want to allow interpolation?

        if not np.array_equal(wave, self.wave):
            raise ValueError("Wave grid doesn't match tabulated grid.")

        if dwave is not None:
            logging.warning('Tabulated PSF does not support overriding dwave.')

        if size is not None:
            logging.warning('Tabulated PSF does not support overriding kernel size.')

        shift = self.kernel.shape[-1] // 2

        # idx will hold negative values because the tabulated kernels are usually
        # computed beyond the wave grid.
        idx = (np.arange(self.kernel.shape[-1]) - shift) + np.arange(wave.size)[:, np.newaxis]

        w = self.wave[s]
        dw = None if self.dwave is None else self.dwave[s]
        k = self.kernel[s]

        if normalize:
            k = self.normalize(k)

        # Return 0 for shift since we have the kernel for the entire wavelength range
        return w, dw, k, idx[s], 0

    def get_optimal_size(self, wave, tol=1e-5):
        raise NotImplementedError()