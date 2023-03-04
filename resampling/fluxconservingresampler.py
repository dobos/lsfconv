import numpy as np
from scipy.interpolate import interp1d

from .resampler import Resampler

class FluxConservingResampler(Resampler):
    def __init__(self, kind=None, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, FluxConservingResampler):
            self.kind = kind if kind is not None else 'linear'
        else:
            self.kind = orig.kind

    def resample_value(self, wave, wave_edges, value, error=None, target_wave=None, target_wave_edges=None):
        target_wave = target_wave if target_wave is not None else self.target_wave
        target_wave_edges = target_wave_edges if target_wave_edges is not None else self.target_wave_edges

        wave_edges = wave_edges if wave_edges is not None else self.find_wave_edges(wave)
        target_wave_edges = target_wave_edges if target_wave_edges is not None else self.find_wave_edges(target_wave)

        if value is None:
            ip_value = None
        else:
            # (Numerically) integrate the flux density as a function of wave at the upper
            # edges of the wavelength bins
            cs = np.empty(wave_edges.shape, dtype=value.dtype)
            cs[0] = 0.0
            cs[1:] = np.cumsum(value * np.diff(wave_edges))
            ip = interp1d(wave_edges, cs, bounds_error=False, fill_value=(0, cs[-1]), kind=self.kind)
            
            # Interpolate the integral and take the numerical differential
            if target_wave_edges.ndim == 1:
                ip_value = np.diff(ip(target_wave_edges)) / np.diff(target_wave_edges)
            elif target_wave_edges.ndim == 2:
                ip_value = (ip(target_wave_edges[1]) - ip(target_wave_edges[0])) / (target_wave_edges[1] - target_wave_edges[0])
            else:
                raise NotImplementedError()

        if error is None:
            ip_error = None
        else:
            # For the error vector, use nearest-neighbor interpolations
            # later we can figure out how to do this correctly and add correlated noise, etc.

            # TODO: do this with correct propagation of error
            ip = interp1d(wave, error, kind='nearest', assume_sorted=True)
            ip_error = ip(target_wave)

        return ip_value, ip_error