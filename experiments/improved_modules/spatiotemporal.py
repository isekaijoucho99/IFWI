"""Stateless frequency continuation and bounded data/gradient response weights.

Temporal weights use observed-window energy and the detached current residual.
Spatial weights use a smoothed magnitude of the *current local velocity gradient*.
This is a response proxy, not physical illumination, cross-shot consistency, a
fault detector, or an inference that late arrivals uniquely correspond to depth.
No ground-truth velocity enters this controller. All weights are diagnostics and
preconditioners, not learned parameters; their construction never backpropagates.
"""
import bisect
import copy
import math
import numbers

import torch
import torch.nn.functional as F


def _positive_number(value, label, maximum=None, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(label + ' must be a finite number')
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(label + ' is outside its valid range')
    if maximum is not None and value > maximum:
        raise ValueError(label + ' exceeds its upper bound')
    return value


def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(label + ' must be an integer >= ' + str(minimum))
    return int(value)


class SpatiotemporalController:
    """A deterministic controller indexed by absolute completed optimizer updates.

    Config defaults: schedule_steps=[30,70], cutoffs_hz=[5,8,None],
    time_attention=False, spatial_attention=False, time_window=64,
    time_strength=.3, spatial_strength=.25, ramp_steps=10,
    spatial_depth_bias=.25, residual_clip=3., energy_floor=1e-12.

    Both attention strengths are <= .5. Per-trace time weights and per-model
    spatial weights have mean 1 and lie in [1-strength, 1+strength]. There is no
    EMA: update_state is accepted for integration but has no effect on weights.
    """
    def __init__(self, config, dt):
        if not isinstance(config, dict):
            raise ValueError('config must be a mapping')
        self.config = copy.deepcopy(config)
        self.dt = _positive_number(dt, 'dt')
        steps = config.get('schedule_steps', [30, 70])
        cutoffs = config.get('cutoffs_hz', [5., 8., None])
        if not isinstance(steps, (list, tuple)) or not isinstance(cutoffs, (list, tuple)):
            raise ValueError('schedule_steps and cutoffs_hz must be sequences')
        self.schedule_steps = [_integer(value, 'schedule step') for value in steps]
        if any(right <= left for left, right in zip(self.schedule_steps, self.schedule_steps[1:])):
            raise ValueError('schedule_steps must be strictly increasing')
        if len(cutoffs) != len(steps) + 1:
            raise ValueError('cutoffs_hz must contain one more entry than schedule_steps')
        self.cutoffs_hz = [None if value is None else _positive_number(value, 'cutoff', .5 / self.dt)
                           for value in cutoffs]
        ordered = [float('inf') if value is None else value for value in self.cutoffs_hz]
        if any(right < left for left, right in zip(ordered, ordered[1:])):
            raise ValueError('Frequency continuation cutoffs must not decrease')
        for key in ('time_attention', 'spatial_attention'):
            value = config.get(key, False)
            if not isinstance(value, bool):
                raise ValueError(key + ' must be boolean')
            setattr(self, key, value)
        self.time_window = _integer(config.get('time_window', 64), 'time_window')
        self.ramp_steps = _integer(config.get('ramp_steps', 10), 'ramp_steps')
        self.time_strength = _positive_number(config.get('time_strength', .3), 'time_strength', .5, True)
        self.spatial_strength = _positive_number(config.get('spatial_strength', .25), 'spatial_strength', .5, True)
        self.spatial_depth_bias = _positive_number(config.get('spatial_depth_bias', .25), 'spatial_depth_bias', .5, True)
        self.residual_clip = _positive_number(config.get('residual_clip', 3.), 'residual_clip')
        self.energy_floor = _positive_number(config.get('energy_floor', 1e-12), 'energy_floor')
        self.completed_updates = 0
        self.last_time_weights = None
        self.last_spatial_weights = None

    @property
    def stage(self):
        return bisect.bisect_right(self.schedule_steps, self.completed_updates)

    @property
    def current_cutoff_hz(self):
        return self.cutoffs_hz[self.stage]

    @property
    def ramp(self):
        return min(self.completed_updates / self.ramp_steps, 1.)

    def set_step(self, completed):
        self.completed_updates = _integer(completed, 'completed', minimum=0)

    @staticmethod
    def _validate_tensor(value, ndim, label):
        if (not torch.is_tensor(value) or value.ndim != ndim or min(value.shape) < 1
                or value.dtype not in (torch.float32, torch.float64)):
            raise ValueError(label + ' must be a nonempty float32/float64 tensor of rank ' + str(ndim))
        if not bool(torch.isfinite(value).all()):
            raise ValueError(label + ' must be finite')

    def filter_data(self, data):
        """Zero-pad to twice the record length, low-pass along time axis 2, crop.

        A full-band stage returns the original tensor without FFT roundoff.
        The smooth response is exp(-.5*(frequency/cutoff)**8).
        """
        self._validate_tensor(data, 4, 'shot data')
        cutoff = self.current_cutoff_hz
        if cutoff is None:
            return data
        count = 2 * data.shape[2]
        frequencies = torch.fft.rfftfreq(count, d=self.dt, device=data.device, dtype=data.dtype)
        response = torch.exp(-.5 * (frequencies / cutoff).pow(8))
        transformed = torch.fft.rfft(data, n=count, dim=2)
        filtered = torch.fft.irfft(transformed * response[None, None, :, None], n=count, dim=2)
        return filtered[:, :, :data.shape[2], :]

    def _bounded_weights(self, score, strength, dimensions):
        # Subtract the minimum before centering so a constant score produces
        # exact zeros, even when float32 mean reduction has roundoff.
        shifted = score - score.amin(dim=dimensions, keepdim=True)
        centered = shifted - shifted.mean(dim=dimensions, keepdim=True)
        scale = centered.abs().amax(dim=dimensions, keepdim=True)
        normalized = centered / scale.clamp_min(self.energy_floor)
        normalized = torch.where(scale > self.energy_floor, normalized, torch.zeros_like(normalized))
        return (1. + strength * self.ramp * normalized).detach()

    def _window_energy(self, traces):
        """Energy per nonoverlapping window, counting only real samples at end."""
        count = traces.shape[-1]
        padding = (-count) % self.time_window
        windows = F.pad(traces.square(), (0, padding)).reshape(traces.shape[0], 1, -1, self.time_window)
        counts = traces.new_full((windows.shape[2],), self.time_window)
        if padding:
            counts[-1] -= padding
        return windows.sum(dim=-1) / counts[None, None, :]

    def _time_weights(self, residual, observed):
        if not self.time_attention or self.time_strength == 0 or self.completed_updates == 0:
            return torch.ones_like(residual).detach()
        with torch.no_grad():
            batch, shots, nt, receivers = residual.shape
            residual_traces = residual.detach().permute(0, 1, 3, 2).reshape(-1, 1, nt)
            observed_traces = observed.detach().permute(0, 1, 3, 2).reshape(-1, 1, nt)
            observed_energy = self._window_energy(observed_traces)
            residual_energy = self._window_energy(residual_traces)
            reference = observed_energy.mean(dim=-1, keepdim=True)
            denominator = observed_energy + .1 * reference + self.energy_floor
            reliability = observed_energy / denominator
            relative_residual = (residual_energy / denominator).clamp(0, self.residual_clip) / self.residual_clip
            scores = reliability * relative_residual
            # Only time is smoothed: neither shots nor receivers are pooled.
            scores = F.avg_pool1d(scores, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            scores = scores.repeat_interleave(self.time_window, dim=-1)[..., :nt]
            scores = F.avg_pool1d(scores, kernel_size=9, stride=1, padding=4, count_include_pad=False)
            # Gate again after smoothing. Silent raw-observation windows cannot
            # inherit positive weight from adjacent energetic windows/filter tails.
            scores = scores * reliability.repeat_interleave(self.time_window, dim=-1)[..., :nt]
            scores = scores.reshape(batch, shots, receivers, nt).permute(0, 1, 3, 2)
            return self._bounded_weights(scores, self.time_strength, dimensions=(2,))

    def data_loss(self, pred, obs, update_state=False):
        """Weighted filtered MSE; raw waveform MSE must be reported by the caller.

        update_state is intentionally unused: all weights are deterministic at a
        fixed absolute step and identical inputs. Diagnostic attributes are updated.
        """
        self._validate_tensor(pred, 4, 'predicted shots')
        self._validate_tensor(obs, 4, 'observed shots')
        if pred.shape != obs.shape or pred.device != obs.device or pred.dtype != obs.dtype:
            raise ValueError('predicted and observed shots must have the same shape, dtype and device')
        residual = self.filter_data(pred - obs)
        self.last_time_weights = self._time_weights(residual, obs)
        return (residual.square() * self.last_time_weights).mean()

    def spatial_gradient(self, grad):
        """Positive bounded preconditioning using detached local gradient response.

        The proxy emphasizes responsive regions; it never divides by local
        response to compensate weak/absent signals. The mild depth preference
        decays across continuation stages and vanishes in the full-band stage.
        """
        self._validate_tensor(grad, 3, 'velocity gradient')
        if not self.spatial_attention or self.spatial_strength == 0 or self.completed_updates == 0:
            self.last_spatial_weights = torch.ones_like(grad).detach()
            return grad
        with torch.no_grad():
            response = F.avg_pool2d(grad.detach().abs()[:, None], kernel_size=5, stride=1,
                                    padding=2, count_include_pad=False).squeeze(1)
            mean_response = response.mean(dim=(1, 2), keepdim=True)
            reliability = response / (response + mean_response + self.energy_floor)
            if self.current_cutoff_hz is None:
                depth_decay = 0.
            else:
                depth_decay = 1. - self.stage / max(len(self.schedule_steps), 1)
            z = torch.linspace(0., 1., grad.shape[1], dtype=grad.dtype, device=grad.device)[None, :, None]
            response_weights = self._bounded_weights(reliability, self.spatial_strength, dimensions=(1, 2))
            depth_weights = self._bounded_weights(reliability * z, self.spatial_strength, dimensions=(1, 2))
            # Mix *after* each normalization: normalizing a depth-biased score
            # afterwards would cancel its decay when the response is uniform.
            # A convex combination retains both bounds and unit spatial mean.
            mix = self.spatial_depth_bias * depth_decay
            self.last_spatial_weights = ((1. - mix) * response_weights + mix * depth_weights).detach()
        return grad * self.last_spatial_weights
