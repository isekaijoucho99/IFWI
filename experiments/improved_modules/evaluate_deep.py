"""
Deep Layer Evaluation Metrics for IFWI
--------------------------------------
Specialized metrics focusing on deep layer inversion quality.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def evaluate_deep_layers(v_true,v_pred,depth_threshold=.5,corner_size=.25,dz=15):
    """Spatial benchmark errors. Undefined metrics use JSON null, never invented scores."""
    if torch.is_tensor(v_true): v_true=v_true.detach().cpu().numpy()
    if torch.is_tensor(v_pred): v_pred=v_pred.detach().cpu().numpy()
    v_true=np.asarray(v_true,dtype=np.float64); v_pred=np.asarray(v_pred,dtype=np.float64)
    if v_true.ndim!=2 or v_true.shape!=v_pred.shape or min(v_true.shape)<3:
        raise ValueError('Expected matching 2D grids with dimensions >=3')
    if not np.isfinite(v_true).all() or not np.isfinite(v_pred).all() or (v_true<=0).any():
        raise ValueError('Nonfinite model or nonpositive truth')
    if not 0<depth_threshold<1 or not 0<corner_size<=.5 or not np.isfinite(dz) or dz<=0:
        raise ValueError('Invalid evaluation region or dz')
    nz,nx=v_true.shape; start=int(nz*depth_threshold)
    cz,cx=int(nz*corner_size),int(nx*corner_size)
    if nz-start<3 or min(cz,cx)<1: raise ValueError('Evaluation region too small')
    a,b=v_true[start:],v_pred[start:]; error=v_pred-v_true
    grad_true=np.gradient(a,dz,axis=0); grad_pred=np.gradient(b,dz,axis=0)
    norm=float(np.linalg.norm(grad_true))
    span=float(np.ptp(a)); window=min(7,*a.shape); window-=1-window%2
    m={'full_rmse':float(np.sqrt(np.mean(error**2))),
       'full_relative_error':float(np.mean(np.abs(error)/v_true)),
       'deep_rmse':float(np.sqrt(np.mean((b-a)**2))),
       'deep_relative_error':float(np.mean(np.abs(b-a)/a)),
       'deep_ssim':float(ssim(a,b,data_range=span,win_size=window)) if span>0 else None,
       'deep_gradient_fidelity':float(1-np.linalg.norm(grad_pred-grad_true)/norm) if norm>0 else None,
       'deep_gradient_rmse_mps_per_m':float(np.sqrt(np.mean((grad_pred-grad_true)**2))),
       'depth_profile':np.mean(np.abs(error)/v_true,axis=1).tolist()}
    for label,region in [('bottom_left',(slice(-cz,None),slice(0,cx))),
                         ('bottom_right',(slice(-cz,None),slice(-cx,None)))]:
        truth=v_true[region]; diff=error[region]
        m[label+'_error']=float(np.mean(np.abs(diff)/truth))
        m[label+'_rmse']=float(np.sqrt(np.mean(diff**2)))
    # No composite "quality" rank: individual metrics have different meanings.
    return m
