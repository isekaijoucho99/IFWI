"""Complete model/Adam/RNG snapshots with atomic, non-overwriting publication.

Each snapshot is serialized once to a unique same-directory prepared file.
os.link atomically creates the target only if absent, unlike os.replace. Only
that link publication is retried. The prepared hard link remains evidence on
success or failure; successful links share the payload storage on NTFS/POSIX.
No training, optimizer update or ledger operation is retried here.
"""
import copy
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

SCHEMA='ifwi_encoding_checkpoint_v1'
DELAYS=(.01,.02,.04,.08,.10)


class CheckpointPublishError(RuntimeError):
    def __init__(self,source,target,attempts,last_error):
        self.source=str(source);self.target=str(target);self.attempts=attempts
        self.winerror=getattr(last_error,'winerror',None)
        self.last_error=last_error
        super().__init__(f'Checkpoint publish failed after {attempts} attempts: '
                         f'{source} -> {target}; winerror={self.winerror}')


def _require(condition,message):
    if not condition:raise ValueError(message)


def _metadata_text(value):
    def basic(item):
        if type(item)is dict:
            return all(type(key)is str and basic(v)for key,v in item.items())
        if type(item)is list:return all(basic(v)for v in item)
        return (item is None or type(item)in(str,int,bool)or
                (type(item)is float and math.isfinite(item)))
    _require(type(value)is dict and basic(value),'metadata must be finite JSON-only basic dict/list values')
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _finite(value,label):
    if isinstance(value,torch.Tensor):
        _require(bool(torch.isfinite(value).all()),f'Nonfinite {label}')
    elif isinstance(value,dict):
        for key,item in value.items():_finite(item,f'{label}.{key}')
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):_finite(item,f'{label}[{index}]')
    elif isinstance(value,float):_require(math.isfinite(value),f'Nonfinite {label}')
    else:_require(value is None or type(value)in(str,int,bool),f'Unsupported checkpoint value in {label}')


def _cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):
        result=copy.copy(value)
        for key,item in value.items():result[key]=_cpu_tree(item)
        return result
    if isinstance(value,list):return [_cpu_tree(item)for item in value]
    if isinstance(value,tuple):return tuple(_cpu_tree(item)for item in value)
    return copy.deepcopy(value)


def _exact(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor)and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict)and set(a)==set(b)and all(_exact(a[key],b[key])for key in a)
    if isinstance(a,(list,tuple)):return type(a)is type(b)and len(a)==len(b)and all(_exact(x,y)for x,y in zip(a,b))
    return type(a)is type(b)and a==b


def _parameter_names(model,optimizer):
    names={id(parameter):name for name,parameter in model.named_parameters()}
    groups=[];seen=[]
    for group in optimizer.param_groups:
        group_names=[]
        for parameter in group['params']:
            _require(id(parameter)in names,'Optimizer contains a parameter outside model')
            group_names.append(names[id(parameter)]);seen.append(id(parameter))
        groups.append(group_names)
    _require(len(seen)==len(set(seen))and set(seen)==set(names),'Optimizer must cover model parameters exactly once')
    return groups


def _capture_rng():
    n=np.random.get_state()
    initialized=torch.cuda.is_initialized()
    return dict(python=random.getstate(),
        numpy=dict(bit_generator=n[0],keys=[int(v)for v in n[1]],pos=int(n[2]),
            has_gauss=int(n[3]),cached_gaussian=float(n[4])),
        torch_cpu=torch.get_rng_state().cpu().clone(),cuda_initialized=initialized,
        torch_cuda=[state.cpu().clone()for state in torch.cuda.get_rng_state_all()]if initialized else[])


def _validate_rng(rng):
    try:
        _require(set(rng)=={'python','numpy','torch_cpu','cuda_initialized','torch_cuda'},'Incomplete RNG state')
        independent=random.Random(0);independent.setstate(rng['python'])
        n=rng['numpy']
        _require(n['bit_generator']=='MT19937'and len(n['keys'])==624 and
            all(type(v)is int and 0<=v<2**32 for v in n['keys'])and
            type(n['pos'])is int and 0<=n['pos']<=624 and n['has_gauss']in(0,1),'Invalid NumPy RNG')
        independent_numpy=np.random.RandomState(0)
        independent_numpy.set_state((n['bit_generator'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
        cpu=rng['torch_cpu']
        _require(isinstance(cpu,torch.Tensor)and cpu.dtype==torch.uint8 and cpu.ndim==1,'Invalid CPU Torch RNG')
        torch.Generator(device='cpu').set_state(cpu.cpu())
        _require(type(rng['cuda_initialized'])is bool and type(rng['torch_cuda'])is list,'Invalid CUDA RNG container')
        _require(rng['cuda_initialized']==bool(rng['torch_cuda']),'CUDA initialization/state count disagree')
        for state in rng['torch_cuda']:
            _require(isinstance(state,torch.Tensor)and state.dtype==torch.uint8 and state.ndim==1 and state.numel()>0,'Invalid saved CUDA RNG')
        _finite(rng,'rng')
    except (KeyError,TypeError,RuntimeError,ValueError)as error:
        raise ValueError('Invalid complete RNG state')from error


def _validate_payload(payload):
    try:
        _require(payload.get('schema')==SCHEMA,'Wrong checkpoint schema')
        completed=payload['completed']
        _require(type(completed)is int and completed>=0,'completed must be a nonnegative integer')
        _metadata_text(payload['metadata'])
        state=payload['model'];optimizer=payload['optimizer']
        _require(isinstance(state,dict)and isinstance(optimizer,dict),'Missing model/optimizer state')
        _finite(state,'model');_finite(optimizer,'optimizer')
        groups=optimizer['param_groups'];group_names=payload['optimizer_parameter_names']
        _require(isinstance(groups,list)and isinstance(group_names,list)and len(groups)==len(group_names),
            'Optimizer groups/names differ')
        parameter_ids=[];parameter_names=[]
        for group,names in zip(groups,group_names):
            _require(isinstance(group,dict)and isinstance(group['params'],list)and isinstance(names,list)and
                len(group['params'])==len(names),'Optimizer group parameter/name count differs')
            _require(all(type(key)is int for key in group['params'])and
                all(type(name)is str and name in state and isinstance(state[name],torch.Tensor)for name in names),
                'Invalid optimizer parameter IDs/names')
            parameter_ids.extend(group['params']);parameter_names.extend(names)
        _require(len(parameter_ids)==len(set(parameter_ids)),'Duplicate optimizer parameter IDs')
        _require(len(parameter_names)==len(set(parameter_names)),'Duplicate optimizer parameter names')
        moments=optimizer['state']
        _require((completed==0 and not moments)or set(moments)==set(parameter_ids),'Incomplete Adam state')
        for key,name in zip(parameter_ids,parameter_names):
            if key not in moments:continue
            item=moments[key];parameter=state[name]
            step=item['step']
            if isinstance(step,torch.Tensor):
                _require(step.numel()==1,'Invalid Adam step tensor');step=step.item()
            _require(not isinstance(step,bool)and float(step)==completed,'Adam step differs from completed')
            _require('exp_avg'in item and 'exp_avg_sq'in item,'Missing Adam moments')
            for moment_name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if moment_name not in item:continue
                moment=item[moment_name]
                _require(isinstance(moment,torch.Tensor)and moment.shape==parameter.shape and
                    moment.dtype==parameter.dtype,'Adam moment shape/dtype differs: '+name+'.'+moment_name)
        for name,value in state.items():
            if name=='completed_updates'or name.endswith('.completed_updates'):
                _require(isinstance(value,torch.Tensor)and value.numel()==1 and
                    value.dtype==torch.int64 and int(value.item())==completed,'Progressive buffer differs from completed')
        _validate_rng(payload['rng'])
    except (KeyError,TypeError,RuntimeError,ValueError)as error:
        raise ValueError('Invalid complete encoding checkpoint: '+str(error))from error


def _publish_once_prepared(source,target):
    for index in range(6):
        try:
            os.link(source,target)
            return
        except OSError as error:
            if getattr(error,'winerror',None)not in(5,32,33):raise
            if index==5:raise CheckpointPublishError(source,target,6,error)from error
            time.sleep(DELAYS[index])


def save_checkpoint(model,opt,path,completed,metadata) -> Path:
    """Save complete state without consuming RNG or overwriting any path.

    Metadata must be JSON-only. CUDA states are included only when CUDA has
    actually been initialized; this function never initializes CUDA for a CPU
    run. The prepared hard link is deliberately retained as publication evidence.
    """
    path=Path(path)
    if path.exists()or path.is_symlink():raise FileExistsError(path)
    _require(isinstance(opt,torch.optim.Adam),'This checkpoint contract requires Adam')
    _metadata_text(metadata)
    payload=dict(schema=SCHEMA,completed=completed,metadata=copy.deepcopy(metadata),
        model=_cpu_tree(model.state_dict()),optimizer=_cpu_tree(opt.state_dict()),
        optimizer_parameter_names=_parameter_names(model,opt),rng=_capture_rng())
    _validate_payload(payload)
    source=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.prepared')
    # Parent creation is the caller's responsibility; no unexpected directory tree.
    with source.open('xb')as stream:
        torch.save(payload,stream)
        stream.flush();os.fsync(stream.fileno())
    _publish_once_prepared(source,path)
    return path


def load_checkpoint(model,opt,path,metadata) -> int:
    """Strictly validate, restore same-model Adam and all RNG, return completed.

    Metadata, step, finite state and model compatibility are checked before any
    mutation. CUDA initialization/device count must match the saved run. CPU
    loading for inspection should use torch.load(weights_only=True) directly.
    """
    expected_metadata=_metadata_text(metadata)
    _require(isinstance(opt,torch.optim.Adam),'This checkpoint contract requires Adam')
    payload=torch.load(Path(path),map_location='cpu',weights_only=True)
    _validate_payload(payload)
    _require(_metadata_text(payload['metadata'])==expected_metadata,'Checkpoint metadata differs')
    current=model.state_dict();saved=payload['model']
    _require(set(current)==set(saved),'Model state keys differ')
    for name,value in current.items():
        other=saved[name]
        _require(isinstance(value,torch.Tensor)and isinstance(other,torch.Tensor)and
            value.shape==other.shape and value.dtype==other.dtype,'Model state shape/dtype differs: '+name)
    _require(_parameter_names(model,opt)==payload.get('optimizer_parameter_names'),'Optimizer parameter order/names differ')
    _require(len(opt.param_groups)==len(payload['optimizer']['param_groups']),'Optimizer group count differs')
    _require(all(len(group['params'])==len(saved_group['params'])for group,saved_group in
        zip(opt.param_groups,payload['optimizer']['param_groups'])),'Optimizer parameter group size differs')
    rng=payload['rng']
    _require(torch.cuda.is_initialized()==rng['cuda_initialized'],'CUDA initialization differs from saved run')
    if rng['cuda_initialized']:
        _require(torch.cuda.device_count()==len(rng['torch_cuda']),'CUDA RNG device count differs')
    model.load_state_dict(saved,strict=True)
    opt.load_state_dict(payload['optimizer'])
    _require(_exact(model.state_dict(),saved)and _exact(opt.state_dict(),payload['optimizer']),'Loaded model/Adam differs')
    random.setstate(rng['python'])
    n=rng['numpy'];np.random.set_state((n['bit_generator'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(rng['torch_cpu'].cpu())
    if rng['cuda_initialized']:torch.cuda.set_rng_state_all([state.cpu()for state in rng['torch_cuda']])
    _require(_exact(_capture_rng(),rng),'Restored RNG differs')
    return payload['completed']
