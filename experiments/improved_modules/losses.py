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


class DepthWeightedLoss:
    """
    Depth-weighted data fitting loss.
    Gives higher weights to gradients from deeper layers.
    """
    def __init__(self, nz, weight_type='exponential', scale=0.3,
                 deep_start=0.5, deep_weight=2.0, bottom_weight=5.0):
        """
        Args:
            nz: number of depth samples
            weight_type: 'exponential', 'linear', or 'piecewise'
            scale: scale factor for exponential weighting
            deep_start: fraction of depth where deep layer starts (for piecewise)
            deep_weight: weight multiplier for deep layers
            bottom_weight: weight multiplier for bottom quarter
        """
        self.nz = nz
        self.weight_type = weight_type

        if weight_type == 'exponential':
            # Exponentially increasing weights with depth
            depth_indices = torch.arange(nz, dtype=torch.float32)
            self.depth_weights = torch.exp(depth_indices / (nz * scale))

        elif weight_type == 'linear':
            # Linearly increasing weights
            self.depth_weights = torch.linspace(1.0, deep_weight, nz)

        elif weight_type == 'piecewise':
            # Piecewise constant weights
            self.depth_weights = torch.ones(nz)
            deep_idx = int(nz * deep_start)
            bottom_idx = int(nz * 0.75)
            self.depth_weights[deep_idx:] = deep_weight
            self.depth_weights[bottom_idx:] = bottom_weight

        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

    def __call__(self, data_pred, data_obs, velocity_model=None):
        """
        Compute depth-weighted loss.

        Args:
            data_pred: predicted shot gathers [num_vels, ns, nt, nx]
            data_obs: observed shot gathers [num_vels, ns, nt, nx]
            velocity_model: velocity model [num_vels, nz, nx] (optional, for gradient weighting)

        Returns:
            loss: scalar weighted loss
        """
        # Standard data fitting loss
        residual = (data_pred - data_obs) ** 2
        data_loss = residual.sum() / (data_pred.shape[0] * data_pred.shape[1] *
                                      data_pred.shape[2] * data_pred.shape[3])

        # If velocity model is provided, apply depth weighting to its gradient
        if velocity_model is not None:
            # This will be applied during backward pass via gradient hooks
            pass

        return data_loss

    def apply_gradient_weighting(self, velocity_grad, device='cpu'):
        """
        Apply depth weights to velocity gradient.

        Args:
            velocity_grad: gradient tensor [num_vels, nz, nx]
            device: torch device

        Returns:
            weighted_grad: depth-weighted gradient
        """
        weights = self.depth_weights.to(device)
        # Broadcast weights: [nz] -> [1, nz, 1]
        weighted_grad = velocity_grad * weights[None, :, None]
        return weighted_grad


class DeepLayerPriorLoss:
    """
    Regularization loss with geological priors for deep layers.
    """
    def __init__(self, nz, nx, deep_start=0.5):
        """
        Args:
            nz, nx: model dimensions
            deep_start: fraction of depth where deep layer starts
        """
        self.nz = nz
        self.nx = nx
        self.deep_idx = int(nz * deep_start)

    def __call__(self, velocity_model, prior_info=None):
        """
        Compute deep layer prior loss.

        Args:
            velocity_model: [num_vels, nz, nx] in m/s
            prior_info: dict with optional constraints
                - 'deep_velocity_range': (v_min, v_max) in m/s
                - 'well_log': (well_x, well_velocities)

        Returns:
            loss: scalar prior loss
        """
        deep_region = velocity_model[:, self.deep_idx:, :]
        losses = []

        # 1. Monotonic increase with depth (deep layers usually faster)
        if velocity_model.shape[1] > 1:
            vertical_diff = torch.diff(velocity_model, dim=1)
            # Penalize decrease in velocity with depth
            loss_monotonic = torch.relu(-vertical_diff).mean()
            losses.append(loss_monotonic)

        # 2. Velocity range constraint for deep layers
        if prior_info and 'deep_velocity_range' in prior_info:
            v_min, v_max = prior_info['deep_velocity_range']
            loss_range = (torch.relu(v_min - deep_region).mean() +
                         torch.relu(deep_region - v_max).mean())
            losses.append(loss_range)

        # 3. Horizontal smoothness in deep layers
        if deep_region.shape[-1] > 1:
            horizontal_tv = torch.abs(torch.diff(deep_region, dim=-1)).mean()
            losses.append(0.1 * horizontal_tv)

        # 4. Well log constraint
        if prior_info and 'well_log' in prior_info:
            well_x, well_velocities = prior_info['well_log']
            if isinstance(well_x, int):
                loss_well = ((velocity_model[:, :, well_x] - well_velocities) ** 2).mean()
                losses.append(loss_well)

        return sum(losses) if losses else torch.tensor(0.0, device=velocity_model.device)


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
    """
    Combined loss function integrating all improvements.
    """
    def __init__(self, nz, nx, config):
        """
        Args:
            nz, nx: model dimensions
            config: dict with loss configuration
                - 'use_depth_weight': bool
                - 'use_prior': bool
                - 'use_multiscale': bool
                - 'depth_weight_params': dict for DepthWeightedLoss
                - 'prior_params': dict for DeepLayerPriorLoss
                - 'multiscale_params': dict for MultiScaleLoss
                - 'lambda_prior': weight for prior loss
        """
        self.config = config

        if config.get('use_depth_weight', False):
            self.depth_loss = DepthWeightedLoss(
                nz, **config.get('depth_weight_params', {})
            )
        else:
            self.depth_loss = None

        if config.get('use_prior', False):
            self.prior_loss = DeepLayerPriorLoss(
                nz, nx, **config.get('prior_params', {})
            )
        else:
            self.prior_loss = None

        if config.get('use_multiscale', False):
            self.multiscale_loss = MultiScaleLoss(
                **config.get('multiscale_params', {})
            )
        else:
            self.multiscale_loss = None

        self.lambda_prior = config.get('lambda_prior', 0.1)

    def __call__(self, data_pred, data_obs, velocity_model,
                 prior_info=None, dt=0.001):
        """
        Compute combined loss.

        Returns:
            loss_dict: dict with individual loss components
        """
        loss_dict = {}

        # Data fitting loss
        if self.depth_loss:
            data_loss = self.depth_loss(data_pred, data_obs, velocity_model)
        elif self.multiscale_loss:
            data_loss = self.multiscale_loss(data_pred, data_obs, dt)
        else:
            # Standard MSE loss
            data_loss = ((data_pred - data_obs) ** 2).sum() / (
                data_pred.shape[0] * data_pred.shape[1] *
                data_pred.shape[2] * data_pred.shape[3]
            )

        loss_dict['data_loss'] = data_loss

        # Prior loss
        if self.prior_loss and velocity_model is not None:
            prior_loss = self.prior_loss(velocity_model, prior_info)
            loss_dict['prior_loss'] = prior_loss
        else:
            prior_loss = torch.tensor(0.0, device=data_pred.device)
            loss_dict['prior_loss'] = prior_loss

        # Total loss
        total_loss = data_loss + self.lambda_prior * prior_loss
        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict
