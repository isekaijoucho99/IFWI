"""
Experimental IFWI networks, including coordinate-local depth attention.

Legacy output attention is retained for reproducibility. Residual output
attention and feature FiLM start at identity for controlled comparisons.
SpatialAttention and MultiScaleIRN are legacy, separately exposed helpers.
"""

import torch
import torch.nn as nn
import numpy as np


class DepthAttention(nn.Module):
    """
    Depth-aware attention module.
    Gives higher attention weights to deep layer features.
    """
    def __init__(self, hidden_dim=64, deep_bias=2.0, depth_min=0., depth_max=1.4):
        """
        Args:
            hidden_dim: dimension for attention network
            deep_bias: bias factor for deep layers (higher = more attention)
        """
        super().__init__()
        self.deep_bias = deep_bias
        if not np.isfinite(depth_max-depth_min) or depth_max<=depth_min:
            raise ValueError("Invalid depth range")
        self.depth_min, self.depth_max = depth_min, depth_max

        # Attention network: depth -> attention weight
        self.attention_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z_coord):
        """
        Args:
            z_coord: depth coordinate [batch, ..., 1]

        Returns:
            attention_weight: [batch, ..., 1], range [1, 1+deep_bias]
        """
        # Normalize depth to [0, 1]
        z_normalized = (z_coord-self.depth_min)/(self.depth_max-self.depth_min)

        # Compute attention weight
        attn = self.attention_net(z_normalized)

        # Scale to [1, 1+deep_bias]
        attn_scaled = 1.0 + self.deep_bias * attn

        return attn_scaled


class _IdentityDepthConditioner(nn.Module):
    """Coordinate-local MLP with an initially zero final affine layer."""
    def __init__(self, out_features, hidden_dim, depth_min, depth_max, strength):
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim < 2:
            raise ValueError('attention_hidden must be an integer >= 2')
        if not np.isfinite(depth_max-depth_min) or depth_max <= depth_min:
            raise ValueError('Invalid depth range')
        if not np.isfinite(strength) or not 0 < strength < 1:
            raise ValueError('attention_strength must be finite and in (0, 1)')
        self.depth_min, self.depth_max = depth_min, depth_max
        self.strength = float(strength)
        self.attention_net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, out_features))
        nn.init.zeros_(self.attention_net[-1].weight)
        nn.init.zeros_(self.attention_net[-1].bias)

    def logits(self, z_coord):
        # Fixed physical bounds preserve predictions when coordinates are cropped.
        depth = (z_coord-self.depth_min)/(self.depth_max-self.depth_min)
        return self.attention_net(depth)


class DepthResidualAttention(_IdentityDepthConditioner):
    """Positive bounded output gain, initialized to exactly one.

    This multiplies the IRN output before physical velocity denormalization.
    It does not promise a positive physical velocity.
    """
    def __init__(self, hidden_dim=64, depth_min=0., depth_max=1.4, strength=.5):
        super().__init__(1, hidden_dim, depth_min, depth_max, strength)

    def forward(self, z_coord):
        return 1. + self.strength * torch.tanh(self.logits(z_coord))


class DepthFeatureFiLM(_IdentityDepthConditioner):
    """Depth-conditioned gain and shift for the last hidden feature vector.

    Both paths are bounded and start at identity. Parameter count grows with
    feature width; no pairwise spatial attention matrix is constructed.
    """
    def __init__(self, features, hidden_dim=64, depth_min=0., depth_max=1.4,
                 strength=.5, shift_strength=.1):
        if not np.isfinite(shift_strength) or shift_strength <= 0:
            raise ValueError('attention_shift_strength must be finite and positive')
        super().__init__(2*features, hidden_dim, depth_min, depth_max, strength)
        self.shift_strength = float(shift_strength)

    def modulation(self, z_coord):
        gain_logits, shift_logits = self.logits(z_coord).chunk(2, dim=-1)
        gain = 1. + self.strength * torch.tanh(gain_logits)
        shift = self.shift_strength * torch.tanh(shift_logits)
        return gain, shift

    def forward(self, features, z_coord):
        gain, shift = self.modulation(z_coord)
        return gain * features + shift


class AttentionIRN(nn.Module):
    """
    Implicit Representation Network with Depth Attention.
    Enhanced version of the original IRN.
    """
    def __init__(self,
                 neuron=[2, 256, 256, 256, 256, 1],
                 omega_0=30,
                 prob=0.2,
                 bias=True,
                 dropout=False,
                 outermost_linear=False,
                 activation='sine',
                 use_attention=True,
                 attention_hidden=64,
                 deep_bias=2.0, depth_min=0., depth_max=1.4,
                 attention_type='legacy_output', attention_strength=.5,
                 attention_shift_strength=.1):
        """
        Args:
            neuron: list defining network architecture
            omega_0: frequency for sine activation
            prob: dropout probability
            bias: use bias in linear layers
            dropout: enable dropout
            outermost_linear: no activation on last layer
            activation: 'sine', 'relu', or 'tanh'
            use_attention: enable depth attention
            attention_hidden: hidden dim for attention network
            deep_bias: attention bias for deep layers
            attention_type: legacy_output, depth_residual, or feature_film
            attention_strength: bounded gain departure from one for new variants
            attention_shift_strength: bounded feature shift for feature_film
        """
        super().__init__()
        self.omega_0 = omega_0
        self.neuron = neuron
        self.d_flag = dropout
        self.outermost_linear = outermost_linear
        self.use_attention = use_attention
        if attention_type not in ('legacy_output', 'depth_residual', 'feature_film'):
            raise ValueError('Unknown attention_type: '+str(attention_type))
        if attention_type == 'feature_film' and len(neuron) < 3:
            raise ValueError('feature_film requires at least one hidden layer')
        self.attention_type = attention_type

        # Main network layers
        self.linear = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for idx in range(len(neuron) - 1):
            self.linear.append(nn.Linear(neuron[idx], neuron[idx + 1], bias=bias))
            if self.d_flag:
                self.dropout_layers.append(nn.Dropout(prob, inplace=False))

        # Activation function
        if activation == 'relu':
            self.omega_0 = 1
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.omega_0 = 1
            self.activation = nn.Tanh()
        else:
            self.activation = torch.sin
            self.init_weights()

        # Depth attention module
        if self.use_attention:
            if attention_type == 'legacy_output':
                self.depth_attention = DepthAttention(attention_hidden, deep_bias, depth_min, depth_max)
            elif attention_type == 'depth_residual':
                self.depth_attention = DepthResidualAttention(
                    attention_hidden, depth_min, depth_max, attention_strength)
            else:
                self.depth_attention = DepthFeatureFiLM(
                    neuron[-2], attention_hidden, depth_min, depth_max,
                    attention_strength, attention_shift_strength)

    def init_weights(self):
        """SIREN initialization"""
        with torch.no_grad():
            self.linear[0].weight.uniform_(-1 / self.neuron[0], 1 / self.neuron[0])
            for ix in range(1, len(self.linear)):
                self.linear[ix].weight.uniform_(
                    -np.sqrt(6 / self.neuron[ix]) / self.omega_0,
                    np.sqrt(6 / self.neuron[ix]) / self.omega_0
                )

    def forward(self, coords):
        """
        Args:
            coords: [batch, nz, nx, 2] with (x, z) coordinates

        Returns:
            feature: [batch, nz, nx, 1] normalized velocity-network output
            coords: input coordinates (with gradient tracking)
        """
        coords = coords.clone().detach().requires_grad_(True)

        # Extract depth coordinate for attention
        z_coord = coords[..., 1:2]  # [..., 1]

        # Main network forward pass
        feature = self.activation(self.omega_0 * self.linear[0](coords))

        for ilayer, layer in enumerate(self.linear[1:]):
            if self.d_flag:
                feature = self.dropout_layers[ilayer](feature)

            if (self.use_attention and self.attention_type == 'feature_film'
                    and ilayer + 2 == len(self.linear)):
                feature = self.depth_attention(feature, z_coord)
            feature = layer(feature)

            if not self.outermost_linear or ilayer + 2 < len(self.linear):
                feature = self.activation(self.omega_0 * feature)

        # Apply depth attention
        if self.use_attention and self.attention_type != 'feature_film':
            attention_weight = self.depth_attention(z_coord)
            feature = feature * attention_weight

        return feature, coords


class SpatialAttention(nn.Module):
    """
    Spatial self-attention for 2D velocity fields.
    Helps capture long-range dependencies, especially for deep layers.
    """
    def __init__(self, in_channels=1, reduction=8):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: [batch, channels, nz, nx]

        Returns:
            out: attention-weighted features
        """
        batch, C, nz, nx = x.size()

        # Query, Key, Value projections
        proj_query = self.query_conv(x).view(batch, -1, nz * nx).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, nz * nx)

        # Attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)

        # Apply attention to value
        proj_value = self.value_conv(x).view(batch, -1, nz * nx)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, nz, nx)

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out


class MultiScaleIRN(nn.Module):
    """
    Multi-scale implicit representation network.
    Processes features at different scales for better deep layer modeling.
    """
    def __init__(self,
                 neuron=[2, 256, 256, 256, 256, 1],
                 omega_0=30,
                 scales=[1.0, 2.0, 4.0],
                 activation='sine'):
        super().__init__()
        self.scales = scales
        self.omega_0 = omega_0
        self.activation = torch.sin if activation == 'sine' else nn.ReLU()

        # Separate network for each scale
        self.scale_nets = nn.ModuleList()
        for _ in scales:
            layers = []
            for idx in range(len(neuron) - 1):
                layers.append(nn.Linear(neuron[idx], neuron[idx + 1]))
            self.scale_nets.append(nn.ModuleList(layers))

        # Fusion layer
        self.fusion = nn.Linear(len(scales), 1)

        self.init_weights()

    def init_weights(self):
        for net in self.scale_nets:
            with torch.no_grad():
                net[0].weight.uniform_(-1 / 2, 1 / 2)
                for layer in net[1:]:
                    n_in = layer.weight.shape[1]
                    layer.weight.uniform_(
                        -np.sqrt(6 / n_in) / self.omega_0,
                        np.sqrt(6 / n_in) / self.omega_0
                    )

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)

        scale_outputs = []

        for scale, net in zip(self.scales, self.scale_nets):
            # Scale coordinates
            scaled_coords = coords * scale

            # Forward pass for this scale
            feature = self.activation(self.omega_0 * net[0](scaled_coords))
            for layer in net[1:-1]:
                feature = self.activation(self.omega_0 * layer(feature))
            feature = net[-1](feature)

            scale_outputs.append(feature)

        # Fuse multi-scale features
        stacked = torch.stack(scale_outputs, dim=-1)
        output = self.fusion(stacked)

        return output, coords


def create_improved_network(config, device='cpu'):
    """
    Factory function to create improved networks based on config.

    Args:
        config: dict with network configuration
            - 'network_type': 'attention', 'multiscale', or 'vanilla'
            - other params specific to each network type
        device: torch device

    Returns:
        network: instantiated network module
    """
    net_type = config.get('network_type', 'vanilla')

    if net_type == 'attention':
        network = AttentionIRN(
            neuron=config.get('neuron', [2, 256, 256, 256, 256, 1]),
            omega_0=config.get('omega_0', 30),
            prob=config.get('prob', 0.2),
            dropout=config.get('dropout', False),
            activation=config.get('activation', 'sine'),
            use_attention=config.get('use_attention',True),
            outermost_linear=config.get('outermost_linear',True),
            bias=config.get('bias',True),
            depth_min=config.get('depth_min',0.),
            depth_max=config.get('depth_max',1.4),
            attention_hidden=config.get('attention_hidden', 64),
            deep_bias=config.get('deep_bias', 2.0),
            attention_type=config.get('attention_type', 'legacy_output'),
            attention_strength=config.get('attention_strength', .5),
            attention_shift_strength=config.get('attention_shift_strength', .1)
        )

    elif net_type == 'multiscale':
        network = MultiScaleIRN(
            neuron=config.get('neuron', [2, 256, 256, 256, 256, 1]),
            omega_0=config.get('omega_0', 30),
            scales=config.get('scales', [1.0, 2.0, 4.0]),
            activation=config.get('activation', 'sine')
        )

    else:  # vanilla
        # Use original IRN from ifwi_modules
        from ifwi_modules import IRN
        network = IRN(
            neuron=config.get('neuron', [2, 256, 256, 256, 256, 1]),
            omega_0=config.get('omega_0', 30),
            prob=config.get('prob', 0.2),
            dropout=config.get('dropout', False),
            activation=config.get('activation', 'sine'),
            outermost_linear=config.get('outermost_linear',True),
            bias=config.get('bias',True)
        )

    return network.to(device)
