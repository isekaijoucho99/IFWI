"""Exact sparse static projections on the chosen device, including PML edges."""
import numpy as np
import torch


def sparse_tensor(matrix,device):
    a=matrix.tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.vstack([a.row,a.col]),dtype=torch.long,device=device),
        torch.tensor(a.data,dtype=torch.float64,device=device),a.shape,device=device).coalesce()


class SparseProjector:
    def __init__(self,system,device):
        self.device=device;self.stiffness=sparse_tensor(system.stiffness,device);self.terms=[]
        for term in system.pml_terms:
            rows=np.unique(term.nonzero()[0])
            self.terms.append((torch.tensor(rows,dtype=torch.long,device=device),sparse_tensor(term[rows,:],device)))

    @torch.no_grad()
    def project(self,basis):
        q=torch.as_tensor(np.asarray(basis),dtype=torch.float64,device=self.device)
        stiffness=q.T@torch.sparse.mm(self.stiffness,q)
        pml=torch.stack([q[rows].T@torch.sparse.mm(term,q) for rows,term in self.terms])
        return stiffness,pml
