import numpy as np
from scipy.interpolate import interp1d

class Resampler():
    def __init__(self, orig=None):
        
        if not isinstance(orig, Resampler):
            self.target_wave = None
            self.target_wave_edges = None
        else:
            self.target_wave = np.copy(orig.target_wave)
            self.target_wave_edges = np.copy(orig.target_wave_edges)

    def init(self, target_wave, target_wave_edges=None):
        self.target_wave = target_wave
        self.target_wave_edges = target_wave_edges if target_wave_edges is not None else self.find_wave_edges(target_wave)

    def reset(self):
        self.target_wave = None
        self.target_wave_edges = None

    def find_wave_edges(self, wave):
        # TODO: Do not automatically assume linear binning

        wave_edges = np.empty((wave.shape[0] + 1,), dtype=wave.dtype)
        wave_edges[1:-1] = 0.5 * (wave[1:] + wave[:-1])
        wave_edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
        wave_edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])

        return wave_edges

    def find_centers(self, wave_edges):
        # TODO: Do not automatically assume linear binning
        return 0.5 * (wave_edges[1:] + wave_edges[:-1])

    def resample_value(self, wave, wave_edges, value, error=None, target_wave=None, target_wave_edges=None):
        raise NotImplementedError()
   
    def resample_mask(self, wave, wave_edges, mask, target_wave=None, target_wave_edges=None):
        target_wave = target_wave if target_wave is not None else self.target_wave

        # TODO: we only take the closest bin here which is incorrect
        #       mask values should be combined using bitwise or across bins

        if mask is None:
            return None
        else:
            wl_idx = np.digitize(target_wave, wave)
            return mask[wl_idx]

    def resample_weight(self, wave, wave_edges, weight, target_wave=None, target_wave_edges=None):
        target_wave = target_wave if target_wave is not None else self.target_wave

        # Weights are resampled using 1d interpolation.
        if weight is None:
            return None
        else:
            ip = interp1d(wave, weight)
            return ip(target_wave)