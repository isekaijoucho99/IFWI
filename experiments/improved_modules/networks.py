"""
Improved Neural Network Architectures for IFWI
----------------------------------------------
Focus on deep layer representation with attention mechanisms.

Includes:
1. Depth-aware attention network
2. Multi-scale feature extraction
3. Physics-informed layers
"""

import torch
import torch.nn as nn
import numpy as np


class DepthAttention(nn.Module):
    """
    Depth-aware attention module.
    Gives higher attention weights to deep layer features.
    """
    def __init__(self, hidden_dim=64, deep_bias=2.0):
        """
        Args:
            hidden_dim: dimension for attention network
            deep_bias: bias factor for deep layers (higher = more attention)
        """
        super().__init__()
        self.deep_bias = deep_bias

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
        z_normalized = z_coord / (z_coord.max() + 1e-8)

        # Compute attention weight
        attn = self.attention_net(z_normalized)

        # Scale to [1, 1+deep_bias]
        attn_scaled = 1.0 + self.deep_bias * attn

        return attn_scaled


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
                 deep_bias=2.0):
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
        """
        super().__init__()
        self.omega_0 = omega_0
        self.neuron = neuron
        self.d_flag = dropout
        self.outermost_linear = outermost_linear
        self.use_attention = use_attention

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
            self.depth_attention = DepthAttention(attention_hidden, deep_bias)

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
            feature: [batch, nz, nx, 1] velocity prediction
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

            feature = layer(feature)

            if not self.outermost_linear or ilayer + 2 < len(self.linear):
                feature = self.activation(self.omega_0 * feature)

        # Apply depth attention
        if self.use_attention:
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
            use_attention=True,
            attention_hidden=config.get('attention_hidden', 64),
            deep_bias=config.get('deep_bias', 2.0)
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
            activation=config.get('activation', 'sine')
        )

    return network.to(device)
