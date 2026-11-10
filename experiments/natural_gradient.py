"""IFWI 自然梯度核心：归一化模型空间 L2 度量 + 阻尼矩阵无关 CG。

不是完整物理 Gauss–Newton；不包含 A.T A。CG 内部不调用物理求解器。
不承诺加速。默认 sum 像素度量；改成 mean 时必须同步重新定义阻尼尺度。
"""
import math
import torch
from torch.func import functional_call, jvp, vjp


def finite(x, name):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f'Non-finite {name}')


def dot64(x, y):
    # CG 标量用 float64 累积，防止小 float32 梯度平方下溢。
    return torch.dot(x.double(), y.double())


def norm64(x):
    return torch.linalg.vector_norm(x.double())


def positive(x, name):
    if not math.isfinite(float(x)) or float(x) <= 0:
        raise ValueError(f'{name} must be finite and positive')


def cg(matvec, rhs, max_iter=10, rtol=1e-3):
    """解 A p = rhs，A 对称正定；零初值，截断后返回残差供审计。"""
    if not isinstance(max_iter,int) or max_iter < 1:
        raise ValueError('max_iter must be a positive integer')
    positive(rtol,'rtol'); finite(rhs,'rhs')
    if rhs.dtype not in (torch.float32, torch.float64):
        raise ValueError('CG supports only float32/float64')
    x=torch.zeros_like(rhs)
    r=rhs.clone(); direction=r.clone(); rr=dot64(r,r)
    initial=float(rr.sqrt())
    if initial == 0:
        return x,dict(iterations=0,relative_residual=0.,converged=True)
    for k in range(max_iter):
        Ad=matvec(direction); finite(Ad,'CG product')
        curvature=dot64(direction,Ad)
        if not torch.isfinite(curvature) or curvature <= 0:
            raise FloatingPointError('CG encountered nonpositive curvature')
        alpha=rr/curvature
        x=x+alpha*direction
        r=r-alpha*Ad
        rr_new=dot64(r,r); finite(rr_new,'CG residual')
        if float(rr_new.sqrt()) <= rtol*initial:
            rr=rr_new
            break
        direction=r+(rr_new/rr)*direction
        rr=rr_new
    # 真残差再核对一次，只增加网络乘积，不增加 PDE。
    residual=rhs-matvec(x); finite(x,'CG solution')
    finite(residual,'true CG residual')
    relative=float(norm64(residual))/initial
    return x,dict(iterations=k+1,relative_residual=relative,converged=relative<=rtol)


def linear_system(q_fn, theta, g_q, damping):
    """冻结当前 theta。rhs = J^T g_q；matvec(p) = J^T J p + lambda p。"""
    positive(damping,'damping'); finite(theta,'theta'); finite(g_q,'g_q')
    _,pullback=vjp(q_fn,theta)
    rhs=pullback(g_q.detach())[0].detach()
    def matvec(p):
        # JVP: 参数方向 p 在速度图像中会造成怎样的变化？
        _,Jp=jvp(q_fn,(theta,),(p,))
        # VJP: 把速度变化映射回参数空间。无需显式构建 J。
        return (pullback(Jp)[0]+damping*p).detach()
    finite(rhs,'parameter rhs')
    return rhs,matvec


def functional_coordinates(model, *, offset=0., scale=1000.):
    """返回固定当前模型的扁平参数和 q(theta)=(velocity-offset)/scale。

    快照 buffers，保持训练网络不变；只更新 requires_grad 参数。
    要求确定性、无可变状态的前向，不支持 dropout/BatchNorm 等。
    """
    positive(scale,'scale')
    for layer in model.modules():
        if isinstance(layer,(torch.nn.modules.batchnorm._BatchNorm,
                             torch.nn.modules.dropout._DropoutNd)):
            raise ValueError('Only deterministic stateless velocity networks are supported')
    all_params=dict(model.named_parameters())
    names=[n for n,p in all_params.items() if p.requires_grad]
    if not names: raise ValueError('No trainable parameters')
    tensors=[all_params[n] for n in names]
    if tensors[0].dtype not in (torch.float32,torch.float64):
        raise ValueError('Parameters must use float32/float64')
    if any(p.dtype!=tensors[0].dtype or p.device!=tensors[0].device for p in tensors):
        raise ValueError('Trainable parameters must share dtype/device')
    theta=torch.cat([p.detach().reshape(-1) for p in tensors]).clone()
    frozen={n:p.detach().clone() for n,p in all_params.items() if n not in names}
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    base=torch.as_tensor(offset,dtype=theta.dtype,device=theta.device).detach().clone()
    finite(base,'offset')
    shapes=[p.shape for p in tensors]; sizes=[p.numel() for p in tensors]
    def q_fn(flat):
        replacements=dict(frozen)
        replacements.update({n:t.reshape(shape) for n,t,shape in zip(names,flat.split(sizes),shapes)})
        velocity=functional_call(model,(replacements,buffers),(),strict=True)
        return (velocity-base)/scale
    return theta,q_fn


def step(model, loss_batches, *, scale, damping, step_size, offset=0.,
         cg_max_iter=10, cg_rtol=1e-3, validate_candidate=None, observer=None):
    """一次显式自然梯度更新。loss_batches 每项是原批次标量损失 callable。

    调用者负责保留原有全炮归一化、频带和边界。这里不自动平均批次。
    前向/loss/validator 必须无副作用；回调不可修改模型或保存外部可变状态。
    失败不改变参数；不会调用 optimizer.step，也不会推进 Adam 状态。
    候选只做网络/CFL等检查，不做损失线搜索，故不保证有限步真实损失下降。
    """
    positive(step_size,'step_size')
    positive(damping,'damping')
    theta,q_fn=functional_coordinates(model,offset=offset,scale=scale)
    q0=q_fn(theta).detach()
    velocity=(torch.as_tensor(offset,device=q0.device,dtype=q0.dtype)+scale*q0).detach().requires_grad_(True)
    finite(velocity,'velocity')
    gm=torch.zeros_like(velocity); loss_value=0.; batches=0
    for loss_fn in loss_batches:
        # 只有这个循环接触正演；每个批次使用同一速度状态。
        loss=loss_fn(velocity)
        if loss.ndim!=0: raise ValueError('Each loss must be scalar')
        finite(loss,'physical loss')
        if observer is not None: observer('backward_attempted', batches)
        gm+=torch.autograd.grad(loss,velocity)[0].detach()
        if observer is not None: observer('backward_completed', batches)
        loss_value+=float(loss.detach()); batches+=1
        del loss
    if not batches: raise ValueError('No loss batches')
    finite(gm,'physical gradient')
    rhs,A=linear_system(q_fn,theta,scale*gm,damping) # 链式法则 g_q = s*g_m
    def counted_A(p):
        if observer is not None: observer('network_matvec', None)
        return A(p)
    p,info=cg(counted_A,rhs,max_iter=cg_max_iter,rtol=cg_rtol)
    dot=float(dot64(rhs,p))
    if not math.isfinite(dot) or (bool(torch.any(rhs != 0)) and dot<=0):
        raise FloatingPointError('Non-descent direction')
    candidate=theta-step_size*p
    finite(candidate,'candidate parameters')
    q1=q_fn(candidate).detach(); finite(q1,'candidate velocity coordinates')
    m1=torch.as_tensor(offset,device=q1.device,dtype=q1.dtype)+scale*q1
    finite(m1,'candidate physical velocity')
    if validate_candidate is not None: validate_candidate(m1)
    _,Jdelta=jvp(q_fn,(theta,),(-step_size*p,))
    finite(Jdelta,'linearized model change')
    denom=max(float(norm64(q1-q0)),torch.finfo(q1.dtype).eps)
    lin_error=float(norm64(q1-q0-Jdelta.detach()))/denom
    if not math.isfinite(lin_error): raise FloatingPointError('Non-finite linearization diagnostic')
    # 到此才提交更新：前面任何检查失败，原模型不变。
    trainable=[p for p in model.parameters() if p.requires_grad]
    sizes=[p.numel() for p in trainable]
    try:
        with torch.no_grad():
            for parameter,chunk in zip(trainable,candidate.split(sizes)):
                parameter.copy_(chunk.reshape_as(parameter))
    except BaseException:
        # 可捕获异常时恢复；进程强杀/断电仍必须依靠外部检查点。
        with torch.no_grad():
            for parameter,chunk in zip(trainable,theta.split(sizes)):
                parameter.copy_(chunk.reshape_as(parameter))
        raise
    for parameter in model.parameters(): parameter.grad=None
    return dict(loss=loss_value,physical_batches=batches,direction_dot_rhs=dot,
                linearization_relative_error=lin_error,**info)


class ResidualSirenView(torch.nn.Module):
    """现有 VelocityParameterization 的可函数化视图；共享参数，不复制训练。

    作者 IRN.forward 中 coords.requires_grad_ 与 torch.func 不兼容。
    此视图复现其前向数值，只省去坐标梯度开关（本更新只需要参数梯度）。
    不改原始作者文件；第一版只支持固定 omega、无 dropout 的普通 SIREN。
    """
    def __init__(self, model):
        super().__init__()
        if getattr(model,'low_rank',None) is not None or model.net.d_flag or getattr(model,'omega_layers',[]):
            raise ValueError('Low-rank/trainable-omega/dropout variants not yet qualified')
        self.model=model
    def forward(self):
        m=self.model; net=m.net
        feature=net.activation(net.omega_0*net.linear[0](m.coords))
        for i,layer in enumerate(net.linear[1:]):
            feature=layer(feature)
            if not net.outermost_linear or i+2<len(net.linear):
                feature=net.activation(net.omega_0*feature)
        raw=feature.squeeze(-1)
        if m.parameterization=='residual':
            return m.fixed_init_mps+raw*(1000*m.std_kmps)
        return (raw*m.std_kmps+m.mean_kmps)*1000


def train_update_ng(model, problem, *, simulate, objective, **ng_options):
    """显式适配原 train_update 的批次；不改变或替换默认 Adam 入口。

    simulate 使用仓库原 _simulate；objective(pred,obs,total_shots)须绑定
    当前训练的全部频带/增益/归一化。抽炮请在调用前按原逻辑执行。
    完整配置/保存/恢复由 natural_gradient_experiment 管理。
    """
    total=problem['observed'].shape[1]
    def batch_loss(item):
        def evaluate(velocity):
            predicted=simulate(item['solver'],velocity,problem['wavelet'])
            observed=problem['observed'][:,item['first']:item['last']]
            return objective(predicted,observed,total)
        return evaluate
    return step(model,[batch_loss(item) for item in problem['propagators']],**ng_options)
