"""Batch source codes with the author's FD/PML arithmetic and adapted injection.

The FD/PML update below is copied without algebraic changes from Jian Sun's
``rnn_fd.py:Propagator2D.___step_rnncell`` (updated February 6, 2020), citing
Sun et al. (2020), GEOPHYSICS 85, R87-R99, "A theory-guided deep-learning
formulation and optimization of seismic waveform inversion." The audited
source file has SHA256 eeeebd821462fadb52bb9b12a06932f2c9f764e3e7482b316e12a94d26ec9fd0.
Only the source injection is adapted to weighted code channels with index_add_;
the original rnn_fd.py is retained unchanged.
"""
import torch
import torch.nn.functional as F

from rnn_fd import Propagator2D, rnn2D


class BatchedEncodedPropagator2D(Propagator2D):
    """Author FD/PML update with one vectorized weighted source injection."""

    def _Propagator2D___step_rnncell(self, source, u_prev, u_, vx_pre, vz_pre):
        # Preserve the author's operation ordering, padding, and PML coefficients.
        ux_x = F.conv2d(u_, self.kernel2Dx.repeat(u_.shape[1], 1, 1, 1), padding=1, groups=u_.shape[1])
        uz_z = F.conv2d(u_, self.kernel2Dz.repeat(u_.shape[1], 1, 1, 1), padding=1, groups=u_.shape[1])

        vx = self.alpha_x[None, None, ...] * vx_pre + self.dt * ux_x * self.belta_x[None, None, ...]
        vz = self.alpha_z[None, None, ...] * vz_pre + self.dt * uz_z * self.belta_z[None, None, ...]
        vx_x = F.conv2d(vx, self.kernel2Dx.repeat(u_.shape[1], 1, 1, 1), padding=1, groups=u_.shape[1])
        vz_z = F.conv2d(vz, self.kernel2Dz.repeat(u_.shape[1], 1, 1, 1), padding=1, groups=u_.shape[1])
        u_partial = F.conv2d(u_, self.kernel2D.repeat(u_.shape[1], 1, 1, 1), padding=(self.k_size - 1) // 2, groups=u_.shape[1])

        u_next = 1/(self.delta * self.dt+1)[None, None, ...] \
                 * ((u_partial * self.velocity**2 + 2*u_ - u_prev)
                    + self.dt * self.delta[None, None, ...] * u_
                    - (self.dt * self.velocity)**2 * (self.bx[None, None, ...] * vx_x
                                                      + self.cx[None, None, ...] * vx
                                                      + self.bz[None, None, ...] * vz_z
                                                      + self.cz[None, None, ...] * vz))

        # Replace only the author's nested model/shot injection loops.
        rows = self.source_z if self.freeSurface else self.source_z + self.npad
        indices = rows * self.nx_pad + self.source_x + self.npad
        injection = u_next.new_zeros(u_next.shape[0], u_next.shape[1],
                                     self.nz_pad * self.nx_pad)
        injection.index_add_(2, indices, source[0][:, None, None] * self.weights[None])
        return u_, u_next + injection.view_as(u_next), vx, vz


def build_code_solver(problem, count):
    geometry, cfg = problem['geometry_tensors'], problem['config']
    if geometry['xs'].shape[0] != 1:
        raise ValueError('Code batching requires one shared velocity model')
    fd = cfg['forward']
    device = problem['observed'].device
    vmax = problem['propagators'][0]['solver'].fd.vmax
    xs = geometry['xs'][:, :1].expand(-1, count).contiguous()
    zs = geometry['zs'][:, :1].expand(-1, count).contiguous()
    xr = geometry['xr'][:, :1].expand(-1, count, -1).contiguous()
    zr = geometry['zr'][:, :1].expand(-1, count, -1).contiguous()
    solver = rnn2D(nz=geometry['nz'], nx=geometry['nx'], xs=xs, zs=zs, xr=xr, zr=zr,
                   dz=cfg['data']['grid_spacing_m'], dt=fd['dt_s'], npad=fd['npad'],
                   order=2, vmax=vmax, log_para=fd['pml_reflection'],
                   freeSurface=fd['free_surface'], dtype=torch.float32, device=device)
    encoded = BatchedEncodedPropagator2D(
        geometry['nz'], geometry['nx'], cfg['data']['grid_spacing_m'], fd['dt_s'],
        fd['npad'], 2, fd['free_surface'], fd['pml_reflection'], vmax, torch.float32, device)
    encoded.source_z = geometry['zs'][0].to(device)
    encoded.source_x = geometry['xs'][0].to(device)
    solver.fd = encoded.to(device)
    return solver
