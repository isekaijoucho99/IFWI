"""
Adaptive Optimizers for IFWI
----------------------------
Depth-adaptive learning rates and advanced optimization strategies.

Includes:
1. Depth-adaptive learning rate scheduler
2. Layer-wise learning rate adjustment
3. Cosine annealing with warm restarts for deep layers
"""

import torch
import torch.optim as optim
import numpy as np


class DepthAdaptiveOptimizer:
    """
    Optimizer with different learning rates for shallow and deep layers.
    Deep layers get higher learning rates to compensate for weak gradients.
    """
    def __init__(self, velocity_net, lr_shallow=1e-4, lr_deep=5e-4,
                 deep_threshold=0.5, optimizer_type='Adam'):
        """
        Args:
            velocity_net: the IRN/AttentionIRN network
            lr_shallow: learning rate for shallow layers
            lr_deep: learning rate for deep layers
            deep_threshold: depth fraction where "deep" starts
            optimizer_type: 'Adam', 'SGD', or 'AdamW'
        """
        self.velocity_net = velocity_net
        self.lr_shallow = lr_shallow
        self.lr_deep = lr_deep
        self.deep_threshold = deep_threshold

        # Group parameters by depth sensitivity
        # For IRN, we can't directly separate by depth, but we can use layer depth
        param_groups = self._create_param_groups()

        # Create optimizer
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(param_groups)
        elif optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(param_groups, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _create_param_groups(self):
        """
        Create parameter groups with different learning rates.
        Later layers (closer to output) affect deep regions more.
        """
        param_groups = []

        # Get all linear layers
        linear_layers = [m for m in self.velocity_net.modules()
                        if isinstance(m, torch.nn.Linear)]

        n_layers = len(linear_layers)
        deep_layer_idx = int(n_layers * self.deep_threshold)

        # Shallow layers (early layers)
        shallow_params = []
        for i, layer in enumerate(linear_layers[:deep_layer_idx]):
            shallow_params.extend(layer.parameters())

        if shallow_params:
            param_groups.append({
                'params': shallow_params,
                'lr': self.lr_shallow,
                'name': 'shallow'
            })

        # Deep layers (later layers)
        deep_params = []
        for i, layer in enumerate(linear_layers[deep_layer_idx:]):
            deep_params.extend(layer.parameters())

        if deep_params:
            param_groups.append({
                'params': deep_params,
                'lr': self.lr_deep,
                'name': 'deep'
            })

        return param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class CosineAnnealingWithWarmup:
    """
    Cosine annealing learning rate scheduler with warm-up.
    Helps with convergence for deep layer features.
    """
    def __init__(self, optimizer, warmup_epochs=100, max_epochs=5000,
                 eta_min=1e-6, last_epoch=-1):
        """
        Args:
            optimizer: torch optimizer
            warmup_epochs: number of warmup epochs
            max_epochs: total training epochs
            eta_min: minimum learning rate
            last_epoch: index of last epoch
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr_scale = self.eta_min + (1 - self.eta_min) * \
                       (1 + np.cos(np.pi * progress)) / 2

        # Update learning rates
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale

    def get_lr(self):
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]


class GradientModifier:
    """
    Modifies gradients to enhance deep layer updates.
    Can be used as a hook on velocity model or network parameters.
    """
    def __init__(self, nz, modification_type='depth_weight',
                 weight_scale=0.3, deep_boost=2.0):
        """
        Args:
            nz: number of depth samples
            modification_type: 'depth_weight' or 'adaptive_clip'
            weight_scale: scale for depth weighting
            deep_boost: boost factor for deep layers
        """
        self.nz = nz
        self.modification_type = modification_type

        if modification_type == 'depth_weight':
            # Create depth-dependent weights
            depth_indices = torch.arange(nz, dtype=torch.float32)
            self.weights = torch.exp(depth_indices / (nz * weight_scale))
            self.weights = self.weights / self.weights[0]  # Normalize to start at 1.0

        elif modification_type == 'adaptive_clip':
            # Adaptive clipping: clip less for deep layers
            self.clip_values = torch.linspace(0.5, 0.1, nz)  # Less clipping for deep

        self.deep_boost = deep_boost

    def __call__(self, grad):
        """
        Modify gradient tensor.

        Args:
            grad: gradient tensor [num_vels, nz, nx] or [nz, nx]

        Returns:
            modified_grad: modified gradient
        """
        if grad is None:
            return None

        if self.modification_type == 'depth_weight':
            # Apply depth weights
            weights = self.weights.to(grad.device)

            if grad.ndim == 3:  # [num_vels, nz, nx]
                modified = grad * weights[None, :, None]
            elif grad.ndim == 2:  # [nz, nx]
                modified = grad * weights[:, None]
            else:
                modified = grad

        elif self.modification_type == 'adaptive_clip':
            # Adaptive gradient clipping
            clip_vals = self.clip_values.to(grad.device)

            if grad.ndim == 3:
                for i in range(grad.shape[1]):  # For each depth
                    grad[:, i, :] = torch.clamp(grad[:, i, :],
                                               -clip_vals[i], clip_vals[i])
            elif grad.ndim == 2:
                for i in range(grad.shape[0]):
                    grad[i, :] = torch.clamp(grad[i, :],
                                            -clip_vals[i], clip_vals[i])
            modified = grad

        else:
            modified = grad

        return modified


def create_improved_optimizer(params, config):
    """
    Factory function to create optimizers based on config.

    Args:
        params: model parameters or velocity_net
        config: dict with optimizer configuration
            - 'optimizer_type': 'adam', 'depth_adaptive', or 'adamw'
            - 'learning_rate': base learning rate
            - 'use_scheduler': bool
            - 'scheduler_params': dict for scheduler

    Returns:
        optimizer: torch optimizer
        scheduler: learning rate scheduler (or None)
        grad_modifier: gradient modifier (or None)
    """
    opt_type = config.get('optimizer_type', 'adam').lower()
    lr = config.get('learning_rate', 1e-4)

    if opt_type == 'depth_adaptive':
        # Depth-adaptive optimizer
        optimizer = DepthAdaptiveOptimizer(
            velocity_net=params,  # Assumes params is the network
            lr_shallow=lr,
            lr_deep=config.get('lr_deep', lr * 5),
            deep_threshold=config.get('deep_threshold', 0.5),
            optimizer_type=config.get('base_optimizer', 'Adam')
        ).optimizer

    elif opt_type == 'adamw':
        optimizer = optim.AdamW(
            params if not hasattr(params, 'parameters') else params.parameters(),
            lr=lr,
            weight_decay=config.get('weight_decay', 1e-4)
        )

    else:  # adam (default)
        optimizer = optim.Adam(
            params if not hasattr(params, 'parameters') else params.parameters(),
            lr=lr
        )

    # Create scheduler if requested
    scheduler = None
    if config.get('use_scheduler', False):
        sched_params = config.get('scheduler_params', {})
        scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=sched_params.get('warmup_epochs', 100),
            max_epochs=sched_params.get('max_epochs', 5000),
            eta_min=sched_params.get('eta_min', 1e-6)
        )

    # Create gradient modifier if requested
    grad_modifier = None
    if config.get('use_grad_modifier', False):
        nz = config.get('nz', 94)
        grad_modifier = GradientModifier(
            nz=nz,
            modification_type=config.get('grad_mod_type', 'depth_weight'),
            weight_scale=config.get('grad_weight_scale', 0.3),
            deep_boost=config.get('grad_deep_boost', 2.0)
        )

    return optimizer, scheduler, grad_modifier
