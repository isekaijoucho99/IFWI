"""
Generator for wavelet or segements.

@author: jiansun

--Update:

The difference between generator0.py and generator.py is the wGenerator.ricker(), whether it is minimum-phase or not.

-Jian 2022.2.20
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from math import exp
from PIL import Image
from scipy.special import binom

#############################################################################################
# ##                                       Wavelet Generator                              ###
#############################################################################################
class wGenerator(object):
    def __init__(self, t, freq=None):
        self.tvec = t
        self.dtype = t.dtype
        self.device = self.tvec.device
        if freq:
            self.freq = freq
        else:
            self.freq = 20  # default frequency for ricker
            self.freqOrmsby = [5, 10, 20, 30]  # default frequency for Ormsby
        
    def ricker(self):
        """
        Return a Ricker wavelet with the specified dominant self.frequency (default: 20Hz).
        """
        tmp = (np.pi * self.freq * (self.tvec - 1.0 / self.freq))**2
        # tmp = (np.pi * self.freq * self.tvec)**2
        wavelet = (1. - 2. * tmp) * torch.exp(-tmp)
        # return wavelet.type(self.dtype).to(self.device)
        return wavelet.type(self.dtype)
        
    def ricker_reform(self):
        """
        Return a reformed Ricker wavelet with the specified dominant self.frequency (default: 20Hz).
        """
        tmp = (np.pi * self.freq * (self.tvec - 1.0 / self.freq))**2
        wavelet = (1. - 2. * tmp) * torch.exp(-tmp * 2)
        # wavelet = (1. - 2. * tmp) * torch.exp(-tmp, dtype=self.dtype)
        # wavelet[0:20] = 0
        # wavelet[48:] = 0
        return wavelet.type(self.dtype).to(self.device)
    
    def gaussian(self):
        """
        Return a wavelet with a gaussian function 
        default: 
        x = t_vec
        xnot = dt
        xwid = dt
        yht = 1
        """
        x = self.tvec
        yht = 1
        xnot = self.tvec[1] - self.tvec[0]
        xwid = xnot
        sigma = xwid / 4
        wavelet = yht * torch.exp(-.5 * ((x - xnot) / sigma)**2)
        return wavelet.type(self.dtype).to(self.device)
    
    def ormsby(self):
        """
        Return a Ormsby wavelet with the specified list self.frequency (default: [5, 10, 20, 30]Hz).
        """
        if isinstance(self.freq, list):
            freqOrmsby = self.freq
        else:
            freqOrmsby = self.freqOrmsby
        wavelet = (np.pi * freqOrmsby[3]**2 / (freqOrmsby[3] - freqOrmsby[2]) 
                   * (np.sinc(np.pi * freqOrmsby[3] * self.tvec)**2) 
                   - np.pi * freqOrmsby[2]**2 / (freqOrmsby[3] - freqOrmsby[2]) 
                   * (np.sinc(np.pi * freqOrmsby[2] * self.tvec)**2)) 
        - (np.pi * freqOrmsby[1]**2 / (freqOrmsby[1] - freqOrmsby[0]) 
           * (np.sinc(np.pi * freqOrmsby[1] * self.tvec)**2) - np.pi 
           * freqOrmsby[0]**2 / (freqOrmsby[1] - freqOrmsby[0]) 
           * (np.sinc(np.pi * freqOrmsby[0] * self.tvec)**2))
        return torch.from_numpy(wavelet, device=self.device, dtype=self.dtype)


"""
For gen_Segment1d & gen_Segment2d:
input_data:     wavelet, is 1D tensor (shape: [num_vels, nt]) in time, 
                    which represents the total "RNN time steps".
                shot_records, represent N shot_records, 
                    in shape of [num_vels, num_shots, nt].
segment_size:   each segment include segment_size time_step, 
                    i.e., segment_size rnn units at each "time (RNN)" step.
option:         =0 (default), averaging partitioning the input with segement_size.
                =1, starting point for segments moving forward by step:
                    for even number segment_size: segment_size//2 step.
                    for odd number segment_size: segment_size//2+1 step.
                =2, starting point for segments at always index=0.
                    For example, segments are:
                    [0->segment_size, 0->2*segment_size, 0->3*segment_size, ...]
"""


#############################################################################################
# ##                                    Segment data Generator                            ###
# ##    Segment data/wavelet generator is for preparing truncated inputs for RNN          ###
#############################################################################################


def gen_Segment2d(wavelet=None, shot_records=None, segment_size=None, option=0):
    if shot_records is not None:                
        num_vels, num_shots, nt, nx = shot_records.shape
    else:
        nt = len(wavelet)
    if segment_size is None:
        segment_size = nt

    x = None
    y = None
    if option == 1:
        num_segments = (nt - (segment_size + 1) // 2) // (segment_size // 2)
        # for even segment_size: num_segments = (nt - segment_size/2) // (segment_size/2)
        # for odd segment_size:  num_segments = (nt - segment_size//2-1) // (segment_size//2)
        for i in range(num_segments):
            if wavelet is not None:
                # prepare the input of wavelet 
                x = wavelet[i * segment_size // 2:i * segment_size // 2 + segment_size]
            if shot_records is not None:
                # partition of shot records
                y = shot_records[:, :, i * segment_size // 2:i * segment_size // 2 + segment_size, :]
            yield (x, y)
    elif option == 2:
        num_segments = nt // segment_size
        if num_segments * segment_size < nt:
            num_segments += 1
        for i in range(num_segments):
            if wavelet is not None:
                # prepare the input of wavelet 
                x = wavelet[0:min((i + 1) * segment_size, nt)]
            if shot_records is not None:
                # partition of shot records
                y = shot_records[:, :, 0:min((i + 1) * segment_size, nt), :]
            yield (x, y)
    else:  # option==0
        num_segments = nt // segment_size
        for i in range(num_segments):
            if wavelet is not None:
                # prepare the input of wavelet 
                x = wavelet[i * segment_size:(i + 1) * segment_size]
            if shot_records is not None:
                # partition of shot records
                y = shot_records[:, :, i * segment_size:(i + 1) * segment_size, :]       
            yield (x, y)

