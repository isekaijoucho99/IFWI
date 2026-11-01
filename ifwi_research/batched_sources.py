"""Batch independent source codes through the original FD shot channels."""
import torch

from rnn_fd import Propagator2D, rnn2D


class BatchedEncodedPropagator2D(Propagator2D):
    """The unchanged FD step, followed by one weighted source injection per code."""

    def _Propagator2D___step_rnncell(self, source, u_prev, u_, vx_pre, vz_pre):
        zero = torch.zeros_like(source[0])
        u_, u_next, vx, vz = Propagator2D._Propagator2D___step_rnncell(
            self, [zero, source[1], source[2]], u_prev, u_, vx_pre, vz_pre)
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
