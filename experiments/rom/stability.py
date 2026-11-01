"""Finite-dimensional pole diagnostics for the reduced PML transfer function.

The auxiliary realization is algebraically equivalent to the reduced rational
operator. Numerical eigenvalues certify neither all materials nor all ranks.
"""
import numpy as np


def projected_recurrence(mass,damped_mass,stiffness,pml_terms,alphas,dt,source):
    mass,damped_mass,stiffness=map(np.asarray,[mass,damped_mass,stiffness])
    terms=np.asarray(pml_terms);alphas=np.asarray(alphas)
    rank=len(mass)
    if terms.shape!=(len(alphas),rank,rank):raise ValueError('PML term/profile mismatch')
    d=mass+dt*damped_mass
    count=rank*(2+len(alphas));matrix=np.zeros((count,count));rhs=np.zeros((count,source.shape[1]))
    matrix[:rank,:rank]=np.linalg.solve(d,2*mass+dt*damped_mass-stiffness-terms.sum(0))
    matrix[:rank,rank:2*rank]=np.linalg.solve(d,-mass)
    matrix[rank:2*rank,:rank]=np.eye(rank)
    rhs[:rank]=np.linalg.solve(d,source)
    for k,(alpha,term) in enumerate(zip(alphas,terms)):
        block=slice((k+2)*rank,(k+3)*rank)
        matrix[:rank,block]=np.linalg.solve(d,-alpha*term)
        matrix[block,:rank]=np.eye(rank);matrix[block,block]=alpha*np.eye(rank)
    return matrix,rhs


def pole_diagnostic(rom,velocity,system,max_dense_dimension=1800):
    mass,damped,source=[x.detach().cpu().numpy() for x in rom.projected_model(velocity)]
    dimension=len(mass)*(2+len(system.alphas))
    if dimension>max_dense_dimension:
        return dict(realization_dimension=dimension,status='not_computed',
            reason='Dense pole calculation exceeds declared diagnostic size; not a stability certificate')
    matrix,rhs=projected_recurrence(mass,damped,rom.stiffness_r.detach().cpu().numpy(),
        rom.pml_r.detach().cpu().numpy(),system.alphas,system.dt,source)
    eigenvalues=np.linalg.eigvals(matrix);radius=float(np.max(np.abs(eigenvalues)))
    return dict(realization_dimension=dimension,spectral_radius=radius,
        poles_outside_unit_disk=int(np.sum(np.abs(eigenvalues)>1+1e-7)),
        contour_radius=float(abs(rom.z[0].item())),
        status='no_exterior_poles_detected' if radius<=1+1e-7 else 'exterior_poles_detected',
        limit='One fixed matrix numerical spectrum; roundoff and other velocities are not certified')
