"""Layerwise (network-layer, not geological-depth) learning rates."""
import math
import warnings
import torch

class CosineAnnealingWithWarmup:
    def __init__(self,optimizer,warmup_epochs=100,max_epochs=5000,eta_min=1e-6,last_epoch=-1):
        if not 0<=warmup_epochs<max_epochs or max_epochs<2:
            raise ValueError('Require 0 <= warmup_epochs < max_epochs and max_epochs >= 2')
        self.optimizer=optimizer
        self.warmup_epochs=warmup_epochs; self.max_epochs=max_epochs
        self.eta_min=float(eta_min); self.last_epoch=last_epoch
        self.base_lrs=[g['lr'] for g in optimizer.param_groups]
        if not math.isfinite(self.eta_min) or not 0<=self.eta_min<=min(self.base_lrs):
            raise ValueError('Invalid absolute eta_min')
    def step(self,epoch=None):
        epoch=self.last_epoch+1 if epoch is None else epoch
        if not 0<=epoch<self.max_epochs: raise ValueError('Step outside scheduler horizon')
        self.last_epoch=epoch
        for group,base in zip(self.optimizer.param_groups,self.base_lrs):
            if epoch<self.warmup_epochs:
                lr=base*(epoch+1)/self.warmup_epochs
            else:
                start=max(self.warmup_epochs-1,0)
                t=(epoch-start)/(self.max_epochs-1-start)
                lr=self.eta_min+(base-self.eta_min)*(1+math.cos(math.pi*t))/2
            group['lr']=lr
    def get_lr(self): return [g['lr'] for g in self.optimizer.param_groups]
    def state_dict(self):
        return {k:getattr(self,k) for k in ('warmup_epochs','max_epochs','eta_min','last_epoch','base_lrs')}
    def load_state_dict(self,state):
        for k in ('warmup_epochs','max_epochs','eta_min','base_lrs'):
            if state[k]!=getattr(self,k): raise ValueError('Scheduler mismatch: '+k)
        self.last_epoch=state['last_epoch']
        if self.last_epoch>=0: self.step(self.last_epoch)


def create_improved_optimizer(params,config):
    kind=config.get('optimizer_type','adam').lower()
    if kind=='depth_adaptive':
        warnings.warn('depth_adaptive means network layerwise_lr, not spatial depth',FutureWarning)
        kind='layerwise_lr'
    if config.get('use_grad_modifier'):
        raise ValueError('Use gradient_preconditioner on velocity; parameter-gradient modification is invalid')
    lr=float(config.get('learning_rate',1e-4))
    if not math.isfinite(lr) or lr<=0: raise ValueError('Invalid learning rate')
    if kind=='layerwise_lr':
        if not hasattr(params,'named_parameters') or not hasattr(params,'linear'):
            raise ValueError('layerwise_lr needs the velocity network with linear backbone')
        threshold=float(config.get('deep_threshold',.5))
        late_lr=float(config.get('lr_deep',5*lr))
        if not 0<threshold<1 or not math.isfinite(late_lr) or late_lr<=0:
            raise ValueError('Invalid layerwise configuration')
        split=max(1,min(len(params.linear)-1,int(len(params.linear)*threshold)))
        early_ids={id(p) for layer in list(params.linear)[:split] for p in layer.parameters()}
        early=[]; late=[]
        for p in params.parameters():
            if p.requires_grad: (early if id(p) in early_ids else late).append(p)
        groups=[{'params':early,'lr':lr,'name':'early_network_layers'},
                {'params':late,'lr':late_lr,'name':'late_network_layers_and_attention'}]
        opt=torch.optim.Adam(groups)
    elif kind in ('adam','adamw'):
        ps=params.parameters() if hasattr(params,'parameters') else params
        cls=torch.optim.Adam if kind=='adam' else torch.optim.AdamW
        opt=cls(ps,lr=lr,**({'weight_decay':config.get('weight_decay',1e-4)} if kind=='adamw' else {}))
    else: raise ValueError('Unknown optimizer: '+kind)
    scheduler=CosineAnnealingWithWarmup(opt,**config.get('scheduler_params',{})) if config.get('use_scheduler') else None
    return opt,scheduler,None
