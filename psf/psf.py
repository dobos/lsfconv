from collections import Iterable
import numpy as np

class Psf():
    """
    When implemented in derived classes, it calculates the point spread
    function kernel and computes the convolution.
    """

    def __init__(self, reuse_kernel=False, orig=None):

        if not isinstance(orig, Psf):
            self.reuse_kernel = reuse_kernel

            self.cached_wave = None
            self.cached_dwave = None
            self.cached_kernel = None
            self.cached_idx = None
            self.cached_shift = None
        else:
            self.reuse_kernel = reuse_kernel if reuse_kernel is not None else orig.reuse_kernel

            self.cached_wave = orig.cached_wave
            self.cached_dwave = orig.cached_dwave
            self.cached_kernel = orig.cached_kernel
            self.cached_idx = orig.cached_idx
            self.cached_shift = orig.cached_shift

    def eval_kernel_at(self, lam, dwave, normalize=True):
        raise NotImplementedError()

    def eval_kernel_impl(self, wave, dwave=None, size=None, s=slice(None), normalize=None):
        """
        Calculate the kernel at each value of the wave vector. If dwave is provided,
        the kernels are evaluated at those offsets, otherwise dwave is calculated
        from wave using `size`, assuming an odd value for kernel size. When `dwave`
        is calculated from `wave`, the beginning and end of the wave grid will be
        avoided.
        """

        normalize = normalize if normalize is not None else False

        if dwave is None:
            size = size if size is not None else 5
            shift = size // 2
            idx = (np.arange(size) - shift) + np.arange(shift, wave.size - shift)[:, np.newaxis]
            w = wave[shift:-shift]
            dw = wave[idx] - w[:, np.newaxis]
        else:
            size = dwave.shape[-1]
            shift = 0
            idx = None
            w = wave
            dw = dwave
        
        # TODO: apply s earlier for speed
        k = self.eval_kernel_at(w[s], dw[s], normalize=normalize)
        
        return w[s], dw[s], k, None if idx is None else idx[s], shift

    def eval_kernel(self, wave, dwave=None, size=None, s=slice(None), normalize=None):

        if self.reuse_kernel and self.cached_kernel is not None:
            return self.cached_wave, self.cached_dwave, self.cached_kernel, self.cached_idx, self.cached_shift

        w, dw, k, idx, shift = self.eval_kernel_impl(wave, dwave=dwave, size=size, s=s, normalize=normalize)

        if self.reuse_kernel:
            self.cached_wave = w
            self.cached_dwave = dw
            self.cached_kernel = k
            self.cached_idx = idx
            self.cached_shift = shift
                
        return w, dw, k, idx, shift

    def normalize(self, k):
        return k / np.sum(k, axis=-1, keepdims=True)

    def get_optimal_size(self, wave, tol=1e-5):
        """
        Given a tolerance and a wave grid, return the optimal kernel size.
        """

        # Evaluate the kernel at the most extreme values of wave and find
        # where the kernel goes below `tol`. Assumes a kernel that decreases
        # monotonically in both directions from the center.

        k = self.eval_kernel_at(wave[0], wave - wave[0], normalize=True)
        s1 = np.max(np.where(k > tol)[1]) * 2 + 1

        k = self.eval_kernel_at(wave[-1], wave - wave[-1], normalize=True)
        s2 = np.max(wave.size - np.where(k > tol)[1]) * 2 + 1

        return max(s1, s2)

    def convolve(self, wave, values, errors=None, size=None, normalize=None):
        """
        Convolve the vectors of the `values` list with a kernel returned by `eval_kernel`.
        Works as numpy convolve with the option `valid`, i.e. the results will be shorter.
        """
        
        if isinstance(values, np.ndarray):
            vv = [ values ]
        else:
            vv = values
        
        if isinstance(errors, np.ndarray):
            ee = [ errors ]
        else:
            ee = errors
        
        w, _, k, idx, shift = self.eval_kernel(wave, size=size, normalize=normalize)

        rv = []
        for v in vv:
            rv.append(np.sum(v[idx] * k, axis=-1))

        # Also convolve the error vector assuming it contains uncorrelated sigmas
        if ee is not None:
            re = []
            for e in ee:
                re.append(np.sqrt(np.sum((e[idx] * k)**2, axis=-1)))
        else:
            re = None

        if isinstance(values, np.ndarray):
            rv = rv[0]

        if isinstance(errors, np.ndarray):
            re = re[0]

        return w, rv, re, shift