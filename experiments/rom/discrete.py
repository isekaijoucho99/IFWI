"""Exact pressure Schur complement of the project's ORIGINAL discrete PML step.

This is a discrete-frequency adaptation, not the SIAM elastic FEM operator.
No stability claim is transferred from that paper to this nonsymmetric PML.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def stencil_matrix(kernel,shape):
    """Zero-padded conv2d cross-correlation, including outer-edge coefficients."""
    kernel=np.asarray(kernel);nz,nx=shape
    rows=np.arange(nz*nx).reshape(nz,nx);all_i=[];all_j=[];all_v=[]
    for iz,ix in np.ndindex(kernel.shape):
        value=kernel[iz,ix]
        if value==0:continue
        dz,dx=iz-kernel.shape[0]//2,ix-kernel.shape[1]//2
        z0,z1=max(0,-dz),min(nz,nz-dz);x0,x1=max(0,-dx),min(nx,nx-dx)
        target=rows[z0:z1,x0:x1].ravel();source=rows[z0+dz:z1+dz,x0+dx:x1+dx].ravel()
        all_i.append(target);all_j.append(source);all_v.append(np.full(len(target),value))
    return sp.coo_matrix((np.concatenate(all_v),(np.concatenate(all_i),np.concatenate(all_j))),shape=(nz*nx,nz*nx)).tocsr()


class DiscretePMLSystem:
    @classmethod
    def from_solver(cls,solver):
        fd=solver.fd;self=cls()
        if fd.k_size!=3:raise ValueError('Only original order2 FD is supported')
        self.shape=(fd.nz_pad,fd.nx_pad);self.dimension=int(np.prod(self.shape))
        self.npad=fd.npad;self.free_surface=fd.freeSurface;self.dt=float(fd.dt)
        self.physical_shape=(fd.nz_pad-fd.npad*(1 if fd.freeSurface else 2),fd.nx)
        self.dx=float(fd.dx);self.reference_vmax=float(fd.vmax)
        as_numpy=lambda x:x.detach().cpu().numpy().astype('float64')
        self.delta=as_numpy(fd.delta).ravel()
        self.stiffness=-stencil_matrix(as_numpy(fd.kernel2D)[0,0],self.shape)
        self.alphas=[];self.pml_terms=[]
        for suffix,kernel in [('x',fd.kernel2Dx),('z',fd.kernel2Dz)]:
            derivative=stencil_matrix(as_numpy(kernel)[0,0],self.shape)
            b=as_numpy(getattr(fd,'b'+suffix)).ravel();c=as_numpy(getattr(fd,'c'+suffix)).ravel()
            alpha=as_numpy(getattr(fd,'alpha_'+suffix)).ravel()
            beta=as_numpy(getattr(fd,'belta_'+suffix)).ravel()
            g=sp.diags(b)@derivative+sp.diags(c)
            for value in np.unique(alpha):
                term=(self.dt**3)*(g@sp.diags(beta*(alpha==value))@derivative)
                term=term.tocsr();term.eliminate_zeros()
                if term.nnz:self.alphas.append(float(value));self.pml_terms.append(term)
        self.alphas=np.asarray(self.alphas)
        return self

    def pad_numpy(self,velocity):
        velocity=np.asarray(velocity)
        if velocity.shape!=self.physical_shape or not np.isfinite(velocity).all() or np.min(velocity)<=0:
            raise ValueError('Require finite positive physical velocity with matching shape')
        top=0 if self.free_surface else self.npad
        return np.pad(velocity,((top,self.npad),(self.npad,self.npad)),mode='edge')

    def physical_indices(self,zx):
        zx=np.asarray(zx)
        if zx.ndim!=2 or zx.shape[1]!=2 or not np.issubdtype(zx.dtype,np.integer):
            raise ValueError('Locations must be integer [count,2] z,x coordinates')
        if np.any(zx<0) or np.any(zx[:,0]>=self.physical_shape[0]) or np.any(zx[:,1]>=self.physical_shape[1]):
            raise ValueError('Location outside physical grid')
        return (zx[:,0]+(0 if self.free_surface else self.npad))*self.shape[1]+zx[:,1]+self.npad

    def pressure_matrix(self,velocity,z):
        if abs(z)<=1:raise ValueError('Use an exterior discrete Laplace contour |z|>1')
        mass=self.pad_numpy(velocity).ravel()**-2
        zm1=z-1
        matrix=self.stiffness+sp.diags((zm1*zm1/z+self.dt*zm1*self.delta)*mass)
        for alpha,term in zip(self.alphas,self.pml_terms):matrix=matrix+(z/(z-alpha))*term
        return matrix.tocsr()

    def source_rhs(self,velocity,padded_indices):
        indices=np.asarray(padded_indices,dtype=int)
        if indices.ndim!=1 or np.any(indices<0) or np.any(indices>=self.dimension):
            raise ValueError('Padded source indices outside domain')
        mass=self.pad_numpy(velocity).ravel()**-2
        rhs=np.zeros((self.dimension,len(indices)))
        rhs[indices,np.arange(len(indices))]=(1+self.dt*self.delta[indices])*mass[indices]
        return rhs
