import copy

import torch

from experiments.gated_siren import GatedSirenVelocity
from ifwi_research.masked_adam import MaskedAdam


def test_capacity_physical_initialization_and_hard_schedule():
    initial = torch.full((4,5),2400.)
    m = GatedSirenVelocity(initial,15.)
    assert sum(p.numel() for p in m.parameters()) == 50049
    assert torch.equal(m(),initial[None])
    assert torch.equal(m.first_omega,torch.linspace(4,30,128))
    for k in [0,1,2,3,99,199,200,201]:
        m.set_progress(k)
        expected = 32 + 96 * min(k,200) // 200
        assert m.active_channels == expected
        assert set(m.gate.tolist()) <= {0.,1.}
        assert m.gate.sum() == expected
        masks = m.optimizer_masks()
        first, second = m.net.linear[:2]
        assert torch.equal(masks[first.weight][:,0],m.gate.bool())
        assert torch.equal(masks[first.bias],m.gate.bool())
        assert torch.equal(masks[second.weight][0],m.gate.bool())
        assert masks[second.bias].all()


def test_gate_applies_to_first_sine_and_restore_rng_is_exact():
    torch.manual_seed(21)
    m = GatedSirenVelocity(torch.full((3,4),2300.),15.)
    with torch.no_grad():
        m.net.linear[-1].weight.fill_(.01)
    x = torch.sin(m.net.linear[0](m.coords)*m.first_omega)*m.gate
    for layer in m.net.linear[1:-1]:
        x = torch.sin(30*layer(x))
    expected = m.fixed_init_mps + 1000*m.net.linear[-1](x).squeeze(-1)
    assert torch.equal(m(),expected)
    opt = MaskedAdam(m.parameters(),lr=.0001,weight_decay=.2)
    def update(model,optimizer,k):
        model.set_progress(k); optimizer.zero_grad()
        ((model()/1000-torch.rand_like(model()))**2).mean().backward()
        optimizer.step(masks=model.optimizer_masks())
    update(m,opt,200)
    before = {p:p.detach().clone() for p in m.parameters()}
    states = {p:copy.deepcopy(opt.state[p]) for p in m.parameters()}
    update(m,opt,0)
    for p,mask in m.optimizer_masks().items():
        assert torch.equal(p[~mask],before[p][~mask])
        for key in states[p]:
            assert torch.equal(opt.state[p][key][~mask],states[p][key][~mask])
    saved_model = copy.deepcopy(m.state_dict()); saved_opt = copy.deepcopy(opt.state_dict())
    rng = torch.get_rng_state()
    update(m,opt,3)
    r = GatedSirenVelocity(torch.full((3,4),2300.),15.)
    o = MaskedAdam(r.parameters()); r.load_state_dict(saved_model); o.load_state_dict(saved_opt)
    torch.set_rng_state(rng); update(r,o,3)
    for p,q in zip(m.parameters(),r.parameters()):
        assert torch.equal(p,q)
        for key in opt.state[p]:
            assert torch.equal(opt.state[p][key],o.state[q][key])
