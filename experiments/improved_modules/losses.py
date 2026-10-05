"""
Improved Loss Functions for IFWI
---------------------------------
Focus on improving deep layer inversion quality.

Includes:
1. Depth-weighted loss
2. Multi-scale loss
3. Multiple-wave aware loss
4. Physics-informed regularization
"""

import torch
import torch.nn.functional as F
import numpy as np


def make_depth_weights(nz, config, device, dtype):
    """Positive spatial preconditioner with unit mean; ratios retain their meaning."""
    if nz < 1: raise ValueError('nz must be positive')
    kind=config.get('weight_type','piecewise')
    deep=float(config.get('deep_weight',2.)); bottom=float(config.get('bottom_weight',5.))
    start=float(config.get('deep_start',.5)); scale=float(config.get('scale',.3))
    if not all(np.isfinite(x) for x in (deep,bottom,start,scale)) or min(deep,bottom,scale)<=0 or not 0<=start<1:
        raise ValueError('Invalid depth weights')
    z=torch.arange(nz,device=device,dtype=dtype)
    if kind=='identity': w=torch.ones_like(z)
    elif kind=='piecewise':
        w=torch.ones_like(z); w[int(nz*start):]=deep; w[int(nz*.75):]=bottom
    elif kind=='linear': w=torch.linspace(1,deep,nz,device=device,dtype=dtype)
    elif kind=='exponential':
        # Subtract maximum before exponentiation to avoid overflow.
        w=torch.exp((z-z.max())/(nz*scale))
    else: raise ValueError('Unknown weight_type: '+kind)
    if not torch.isfinite(w).all() or not (w>0).all(): raise ValueError('Nonfinite/underflowed weights')
    return w/w.mean()


def attach_velocity_preconditioner(v_wave, weights):
    if v_wave.ndim!=3 or weights.ndim!=1 or v_wave.shape[1]!=len(weights):
        raise ValueError('Expected velocity [batch,nz,nx] and weights [nz]')
    weights=weights.to(device=v_wave.device,dtype=v_wave.dtype)
    return v_wave.register_hook(lambda grad: grad*weights[None,:,None])


class DepthWeightedLoss:
    """Removed misleading API: depth is not an axis of shot-gather MSE."""
    def __init__(self,*args,**kwargs):
        raise ValueError('Use gradient_preconditioner on the wave velocity branch; data loss remains MSE')


class DeepLayerPriorLoss:
    """Deep priors on physical velocity, with explicit units and smoothing kernel.

    ``velocity_scale=1000`` expresses penalties in km/s for m/s input.
    Charbonnier smooths the horizontal TV term only; range and monotonic
    violations retain their one-sided hinge interpretation.
    """
    def __init__(self,nz,nx,deep_start=.5,monotonic_weight=0.,horizontal_weight=.1,range_weight=1.,
                 velocity_scale=1.,prior_form='tv',charbonnier_eps=1e-3):
        if not 0<=deep_start<1: raise ValueError('Invalid deep_start')
        self.deep_idx=int(nz*deep_start)
        self.monotonic_weight=monotonic_weight
        self.horizontal_weight=horizontal_weight
        self.range_weight=range_weight
        self.velocity_scale=float(velocity_scale)
        self.prior_form=prior_form
        self.charbonnier_eps=float(charbonnier_eps)
        if any(not np.isfinite(x) or x<0 for x in (monotonic_weight,horizontal_weight,range_weight)):
            raise ValueError('Prior coefficients must be nonnegative and finite')
        if any(not np.isfinite(x) or x<=0 for x in (self.velocity_scale,self.charbonnier_eps)):
            raise ValueError('velocity_scale and charbonnier_eps must be positive and finite')
        if prior_form not in ('tv','charbonnier'):
            raise ValueError('Unknown prior_form: '+str(prior_form))
    def components(self,velocity_model,prior_info=None):
        deep=velocity_model[:,self.deep_idx:,:]/self.velocity_scale
        terms={k:velocity_model.new_zeros(()) for k in ('prior_monotonic','prior_horizontal','prior_range')}
        if self.monotonic_weight and deep.shape[1]>1:
            terms['prior_monotonic']=self.monotonic_weight*torch.relu(-torch.diff(deep,dim=1)).mean()
        if self.horizontal_weight and deep.shape[-1]>1:
            diff=torch.diff(deep,dim=-1)
            horizontal=diff.abs() if self.prior_form=='tv' else (diff.square()+self.charbonnier_eps**2).sqrt()-self.charbonnier_eps
            terms['prior_horizontal']=self.horizontal_weight*horizontal.mean()
        if prior_info and 'deep_velocity_range' in prior_info:
            lo,hi=prior_info['deep_velocity_range']
            if not np.isfinite([lo,hi]).all() or lo>=hi: raise ValueError('Invalid velocity range')
            lo,hi=lo/self.velocity_scale,hi/self.velocity_scale
            terms['prior_range']=self.range_weight*(torch.relu(lo-deep).mean()+torch.relu(deep-hi).mean())
        return terms
    def __call__(self,velocity_model,prior_info=None):
        return sum(self.components(velocity_model,prior_info).values())


class MultiScaleLoss:
    """
    Multi-scale loss combining different frequency bands.
    """
    def __init__(self, frequency_bands=[(2, 5), (5, 10), (10, 20)],
                 weights=[1.0, 1.0, 1.0]):
        """
        Args:
            frequency_bands: list of (f_low, f_high) in Hz
            weights: relative weights for each band
        """
        self.frequency_bands = frequency_bands
        self.weights = weights

    def bandpass_filter(self, data, f_low, f_high, dt):
        """
        Simple bandpass filter using FFT.

        Args:
            data: shot gather [num_vels, ns, nt, nx]
            f_low, f_high: frequency band in Hz
            dt: time sampling in seconds

        Returns:
            filtered_data: bandpass filtered data
        """
        # FFT along time axis
        data_fft = torch.fft.rfft(data, dim=2)
        freq = torch.fft.rfftfreq(data.shape[2], d=dt).to(data.device)

        # Create bandpass mask
        mask = ((freq >= f_low) & (freq <= f_high)).float()

        # Apply filter
        data_fft_filtered = data_fft * mask[None, None, :, None]
        filtered_data = torch.fft.irfft(data_fft_filtered, n=data.shape[2], dim=2)

        return filtered_data

    def __call__(self, data_pred, data_obs, dt=0.001):
        """
        Compute multi-scale loss.

        Args:
            data_pred, data_obs: [num_vels, ns, nt, nx]
            dt: time sampling interval

        Returns:
            total_loss: weighted sum of band losses
        """
        total_loss = 0.0
        for (f_low, f_high), weight in zip(self.frequency_bands, self.weights):
            pred_filtered = self.bandpass_filter(data_pred, f_low, f_high, dt)
            obs_filtered = self.bandpass_filter(data_obs, f_low, f_high, dt)

            band_loss = ((pred_filtered - obs_filtered) ** 2).mean()
            total_loss += weight * band_loss

        return total_loss / sum(self.weights)


class CombinedLoss:
    """Explicit data objective and priors; raw waveform MSE is always reported."""
    def __init__(self,nz,nx,config):
        if config.get('use_depth_weight'):
            raise ValueError('Move use_depth_weight to gradient_preconditioner.enabled')
        if config.get('use_multiscale'):
            raise ValueError('Multiscale objective is outside the repaired comparison protocol')
        self.config=config
        self.data_objective=config.get('data_objective','mse')
        if self.data_objective not in ('mse','huber'):
            raise ValueError('Unknown data_objective: '+str(self.data_objective))
        self.huber_beta=float(config.get('huber_beta',1.))
        if self.data_objective=='huber' and (not np.isfinite(self.huber_beta) or self.huber_beta<=0):
            raise ValueError('huber_beta must be positive and finite in shot amplitude units')
        self.prior_loss=DeepLayerPriorLoss(nz,nx,**config.get('prior_params',{})) if config.get('use_prior') else None
        self.lambda_prior=float(config.get('lambda_prior',.1))
        if not np.isfinite(self.lambda_prior) or self.lambda_prior<0: raise ValueError('Invalid lambda_prior')
    def __call__(self,data_pred,data_obs,velocity_model,prior_info=None,dt=.001):
        residual=data_pred-data_obs
        data_mse=residual.square().mean()
        if self.data_objective=='huber':
            # This convention matches r^2 (and its derivative) in the core.
            beta=self.huber_beta
            data_loss=torch.where(residual.abs()<=beta,residual.square(),2*beta*residual.abs()-beta**2).mean()
        else:
            data_loss=data_mse
        info=prior_info or {'deep_velocity_range': self.config.get('velocity_range',[1500.,5500.])}
        terms=self.prior_loss.components(velocity_model,info) if self.prior_loss else {k:data_loss.new_zeros(()) for k in ('prior_monotonic','prior_horizontal','prior_range')}
        prior=sum(terms.values())
        total=data_loss+self.lambda_prior*prior
        return total,{'data_loss':data_loss,'data_mse':data_mse,'prior_loss':prior,'total_loss':total,**terms}
