"""UNTRAINED masked conditional transport contract, not a scientific qualification.

No trained weights are provided. See train_conditional_dfm for the separate
training implementation. The training contract uses Gaussian t=0 and geological t=1 marginals, a shared condition/time
pair per group, and estimated anchors (never true known pixels). The loader checks
architecture compatibility and qualification *declarations*, not the scientific
validity of those declarations or a report. Existing smooth-condition checkpoints
are incompatible and rejected. Generated targets are soft proposals, not truth.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'ifwi_masked_conditional_dfm_v1'
ARCHITECTURE = 'masked_two_time_cnn_v1'
MULTISCALE_ARCHITECTURE = 'masked_two_time_unet_v1'
ARCHITECTURES = (ARCHITECTURE, MULTISCALE_ARCHITECTURE)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be integer in [{low}, {high}]')
    return value


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


class MaskedTwoTimeTransport(nn.Module):
    """Small NEW architecture: T=x+(r-t)u; six input channels, no attention.

    Raw architecture is untrained. A condition has one background, one normalized
    estimated-anchor-times-mask, and one binary mask channel, shared by all draws.
    It does not hard overwrite output anchors and does not mask transport velocity.
    """
    def __init__(self, width: int = 32):
        super().__init__()
        self.width = _integer(width, 'width', 4, 128)
        self.net = nn.Sequential(nn.Conv2d(6,width,3,padding=1), nn.SiLU(),
                                 nn.Conv2d(width,width,3,padding=1), nn.SiLU(),
                                 nn.Conv2d(width,1,3,padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t: float, r: float, condition):
        if not math.isfinite(t) or not math.isfinite(r) or not 0 <= t <= r <= 1:
            raise ValueError('Require shared scalar 0 <= t <= r <= 1')
        if x.ndim != 4 or x.shape[1] != 1 or condition.shape != (1,3,*x.shape[-2:]):
            raise ValueError('Require B1HW samples and one shared 13HW condition')
        if not torch.isfinite(x).all() or not torch.isfinite(condition).all():
            raise ValueError('Nonfinite transport input')
        if r == t:
            return x
        features = torch.cat((x, condition.expand(x.shape[0],-1,-1,-1),
                              torch.full_like(x,t), torch.full_like(x,r-t)), dim=1)
        return x + (r-t)*self.net(features)


class _ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming,outgoing,3,padding=1), nn.SiLU(),
                                    nn.Conv2d(outgoing,outgoing,3,padding=1))
        self.skip = nn.Conv2d(incoming,outgoing,1) if incoming != outgoing else nn.Identity()

    def forward(self, x):
        return F.silu(self.layers(x) + self.skip(x))


class MaskedMultiScaleTransport(nn.Module):
    """Optional compact three-resolution residual U-Net, still UNTRAINED.

    Two pooling levels expand spatial context relative to the baseline CNN.
    This is not global attention and does not guarantee global geology coherence.
    Bilinear resizing handles odd and singleton axes without cropping conditions.
    Memory/performance at production size (including 16 GB GPUs) is unverified.
    """
    def __init__(self, width: int = 32):
        super().__init__()
        self.width = _integer(width,'width',4,128)
        self.encoder0 = _ResidualBlock(6,width)
        self.encoder1 = _ResidualBlock(width,2*width)
        self.bottleneck = _ResidualBlock(2*width,4*width)
        self.decoder1 = _ResidualBlock(6*width,2*width)
        self.decoder0 = _ResidualBlock(3*width,width)
        self.output = nn.Conv2d(width,1,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t: float, r: float, condition):
        if not math.isfinite(t) or not math.isfinite(r) or not 0 <= t <= r <= 1:
            raise ValueError('Require shared scalar 0 <= t <= r <= 1')
        if x.ndim != 4 or x.shape[1] != 1 or condition.shape != (1,3,*x.shape[-2:]):
            raise ValueError('Require B1HW samples and one shared 13HW condition')
        if not torch.isfinite(x).all() or not torch.isfinite(condition).all():
            raise ValueError('Nonfinite transport input')
        if r == t:
            return x
        features = torch.cat((x,condition.expand(x.shape[0],-1,-1,-1),
                              torch.full_like(x,t),torch.full_like(x,r-t)),dim=1)
        e0 = self.encoder0(features)
        e1 = self.encoder1(F.avg_pool2d(e0,2,ceil_mode=True))
        low = self.bottleneck(F.avg_pool2d(e1,2,ceil_mode=True))
        d1 = self.decoder1(torch.cat((F.interpolate(low,size=e1.shape[-2:],mode='bilinear',
                                                   align_corners=False),e1),dim=1))
        d0 = self.decoder0(torch.cat((F.interpolate(d1,size=e0.shape[-2:],mode='bilinear',
                                                   align_corners=False),e0),dim=1))
        return x + (r-t)*self.output(d0)


def make_transport(architecture: str = ARCHITECTURE, width: int = 32):
    """Construct an UNTRAINED backbone. State dictionaries are architecture-specific."""
    if architecture == ARCHITECTURE:
        return MaskedTwoTimeTransport(width)
    if architecture == MULTISCALE_ARCHITECTURE:
        return MaskedMultiScaleTransport(width)
    raise ValueError('Unknown conditional transport architecture')


class FrozenConditionalDFM:
    """Fail-closed adapter for externally trained, declared qualified checkpoints.

    Qualification status/report hash is evidence metadata, not a certificate.
    No unqualified override exists. Caller supplies no truth to this interface.
    All inference conditions are detached copies; no gradient enters the target.
    """
    def __init__(self, checkpoint, spacing_m, device='cpu', *, nfe=1, seed=3, max_cells=4096):
        self.nfe = _integer(nfe,'nfe',1,64)
        self.seed = _integer(seed,'seed',0,2**63-1)
        self.max_cells = _integer(max_cells,'max_cells',1,1048576)
        self.spacing_m = _positive(spacing_m,'spacing_m')
        self.device = torch.device(device)
        path = Path(checkpoint)
        if not path.is_file() or path.stat().st_size > 64*1024*1024:
            raise ValueError('Checkpoint missing or exceeds 64 MiB resource limit')
        self.checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        data = torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(data,dict) or not isinstance(data.get('meta'),dict):
            raise ValueError('New conditional checkpoint metadata required')
        self.meta = data['meta']
        m = self.meta
        if m.get('schema') != SCHEMA or m.get('architecture') not in ARCHITECTURES:
            raise ValueError('Incompatible schema/architecture; legacy smooth-condition weights rejected')
        if m.get('units') != 'm/s':
            raise ValueError('Checkpoint velocity units must be m/s')
        width = _integer(m.get('width'),'width',4,128)
        self.mean_mps = _positive(m.get('mean_mps'),'mean_mps')
        self.scale_mps = _positive(m.get('scale_mps'),'scale_mps')
        if not 100 <= self.mean_mps <= 15000 or not 1 <= self.scale_mps <= 15000:
            raise ValueError('Normalization must be in physical m/s units')
        if not math.isclose(_positive(m.get('spacing_m'),'spacing_m'),self.spacing_m,rel_tol=1e-7):
            raise ValueError('Checkpoint spacing_m does not match inversion spacing')
        tr=m.get('training',{})
        if not isinstance(tr,dict): raise ValueError('Conditional training declaration required')
        _integer(tr.get('steps'),'training steps',1,2**63-1)
        required = dict(contract='shared_background_estimate_anchors_mask_v1',
                        time_semantics='gaussian_0_to_geology_1_two_time_transport',
                        anchor_semantics='estimated_not_truth')
        if tr.get('toy') is not False or any(tr.get(k)!=v for k,v in required.items()):
            raise ValueError('Non-toy conditional training/time/anchor contract required')
        train, val = tr.get('train_model_hashes'), tr.get('val_model_hashes')
        if any(not isinstance(v,list) or not v or not all(_sha(h) for h in v) for v in (train,val)):
            raise ValueError('Independent training and validation model SHA256 lists required')
        if set(train)&set(val): raise ValueError('Training and validation model hashes must be disjoint')
        q=m.get('qualification',{})
        if (not isinstance(q,dict) or q.get('status')!='qualified' or
            not isinstance(q.get('report_ref'),str) or not q['report_ref'].strip() or not _sha(q.get('report_sha256'))):
            raise ValueError('Qualification declaration with report reference and SHA256 required')
        # Constructing a network must not perturb the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.model = make_transport(m['architecture'],width)
        try:
            self.model.load_state_dict(data['state_dict'],strict=True)
        except (KeyError,RuntimeError) as exc:
            raise ValueError('Incompatible conditional model state') from exc
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('Nonfinite checkpoint parameters')
        self.model.to(self.device).eval().requires_grad_(False)
        self.calls = 0

    @property
    def inference_identity(self):
        """Bind cached targets to the exact declared sampler, not only weights.

        Native module parameters are immutable after loading. This property
        detects changed time grids/seeds/normalization before a cached refresh.
        """
        settings=dict(checkpoint=self.checkpoint_hash,nfe=self.nfe,seed=self.seed,
                      mean_mps=self.mean_mps,scale_mps=self.scale_mps,
                      spacing_m=self.spacing_m,max_cells=self.max_cells,
                      architecture=self.meta['architecture'],device=str(self.device))
        return hashlib.sha256(json.dumps(settings,sort_keys=True,allow_nan=False).encode()).hexdigest()

    @torch.no_grad()
    def generate(self, background_mps, anchor_mps, anchor_mask, *, sample_id):
        _integer(sample_id,'sample_id',0,2**63-1)
        inputs=(background_mps,anchor_mps,anchor_mask)
        if any(not isinstance(v,torch.Tensor) for v in inputs): raise ValueError('Tensor inputs required')
        shape=background_mps.shape
        if len(shape)!=3 or shape[0]!=1 or min(shape[-2:])<1 or any(v.shape!=shape for v in inputs):
            raise ValueError('Matching nonempty [1,H,W] inputs required')
        if background_mps.numel()>self.max_cells:
            raise ValueError('Grid cells exceed max_cells before inference allocation')
        if any(not torch.isfinite(v).all() for v in inputs): raise ValueError('Nonfinite condition')
        if any(not v.is_floating_point() for v in inputs[:2]): raise ValueError('Floating m/s inputs required')
        if not ((anchor_mask==0)|(anchor_mask==1)).all(): raise ValueError('Binary estimated-anchor mask required')
        for v in inputs[:2]:
            if not ((v>=100)&(v<=15000)).all(): raise ValueError('Velocity must be physical m/s in [100,15000]')
        bg, anchors, mask = (v.detach().to(device=self.device,dtype=torch.float32).clone()[:,None] for v in inputs)
        condition=torch.cat(((bg-self.mean_mps)/self.scale_mps,
                             ((anchors-self.mean_mps)/self.scale_mps)*mask,mask),dim=1)
        rng=torch.Generator(device='cpu').manual_seed((self.seed*1000003+sample_id)%2**63)
        state=torch.randn((1,1,*shape[-2:]),generator=rng,dtype=torch.float32).to(self.device)
        for i in range(self.nfe):
            self.calls += 1
            state=self.model(state,i/self.nfe,(i+1)/self.nfe,condition)
            if not torch.isfinite(state).all(): raise FloatingPointError('Nonfinite conditional DFM output')
        output=(state[:,0]*self.scale_mps+self.mean_mps).detach()
        if not ((output>=100)&(output<=15000)).all():
            raise ValueError('Generated proposal outside physical velocity range [100,15000] m/s')
        return output
