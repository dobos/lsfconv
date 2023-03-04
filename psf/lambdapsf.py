import numpy as np

from .psf import Psf

class LambdaPsf(Psf):
    """
    Implements a point spread function with an arbitrary kernel function provided
    as a callable.
    """

    def __init__(self, func, reuse_kernel=False, orig=None):
        super().__init__(reuse_kernel=reuse_kernel, orig=orig)

        if isinstance(orig, LambdaPsf):
            self.func = func
        else:
            self.func = func if func is not None else orig.func

    def eval_kernel_at(self, lam, dwave, normalize=True):
        """
        Calculate the kernel around `lam` at `dwave` offsets.
        """
        if not isinstance(lam, np.ndarray):
            lam = np.array([lam])

        k = self.func(lam[:, np.newaxis], dwave)

        if normalize:
            k = self.normalize(k)

        return k