"""Frozen-family actors and deterministic design utilities; no simulator at import."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EDGES = ((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))
ROOT = Path(__file__).resolve().parent


def seed_for(split: str, *identity: object) -> int:
    if split not in {'qa','train','selection','heldout','causal','bootstrap'}:
        raise ValueError('Only explicitly development-only split namespaces are allowed')
    raw = json.dumps(['PD-B1S-D-MW-v1', split, *identity], separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big')


def masks() -> list[str]:
    return [''.join(map(str,b)) for b in itertools.product((0,1),repeat=6) if sum(b)<=3]


def matrix(mask: str) -> np.ndarray:
    if len(mask)!=6 or set(mask)-{'0','1'}:
        raise ValueError('A support is exactly six binary digits in EDGES order')
    s=np.zeros((3,3),dtype=np.float32)
    for digit,(i,j) in zip(mask,EDGES):
        s[i,j]=int(digit)
    return s


def degree_class(mask: str) -> list[str]:
    s=matrix(mask)
    return [b for b in masks() if b!=mask and np.array_equal(matrix(b).sum(0),s.sum(0)) and np.array_equal(matrix(b).sum(1),s.sum(1))]


def log_squashed_gaussian(pre: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Joint 12-D density; stable exact tanh log-Jacobian, not action clipping."""
    dist=torch.distributions.Normal(mu,log_std.exp())
    log_jac=2*(math.log(2)-pre-F.softplus(-2*pre))
    return (dist.log_prob(pre)-log_jac).sum(-1)


class RoutingActor(nn.Module):
    def __init__(self, mask: str):
        super().__init__()
        self.E=nn.Linear(96,16); self.A=nn.Linear(32,8)
        self.B=nn.Linear(24,16); self.D=nn.Linear(16,4)
        self.log_std=nn.Parameter(torch.full((4,),-0.5))
        self.register_buffer('support',torch.from_numpy(matrix(mask)))
        self.mask=mask
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.E,self.A,self.B,self.D):
            nn.init.orthogonal_(layer.weight,math.sqrt(2)); nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.D.weight,0.01)

    def forward(self,z,lesion: tuple[int,...]=(),trace: bool=False):
        if z.ndim!=2 or z.shape[1]!=93:
            raise ValueError('Raw shared observation must be [batch,93]')
        onehot=torch.eye(3,device=z.device,dtype=z.dtype).expand(z.shape[0],-1,-1)
        h=torch.tanh(self.E(torch.cat([z[:,None,:].expand(-1,3,-1),onehot],-1)))
        # All six messages are always computed, independent of support and lesion.
        msgs=torch.stack([torch.tanh(self.A(torch.cat((h[:,i],h[:,j]),-1))) for i,j in EDGES],1)
        gate=z.new_tensor([self.support[i,j].item() for i,j in EDGES])
        lam=torch.ones_like(gate)
        if len(set(lesion))!=len(lesion) or any(e not in range(6) for e in lesion):
            raise ValueError('Invalid edge lesion')
        for e in lesion:
            lam[e]=0
        used=msgs*(gate*lam)[None,:,None]
        # Fixed denominator 2. Never renormalize remaining routes after a lesion.
        r=torch.stack([used[:,[e for e,(_,j) in enumerate(EDGES) if j==node]].sum(1)/2 for node in range(3)],1)
        g=torch.tanh(self.B(torch.cat((h,r),-1)))
        mu=self.D(g).flatten(1)
        ls=self.log_std.repeat(3).expand_as(mu)
        if trace:
            return mu,ls,{'h':h,'m':msgs,'used':used,'r':r,'lambda':lam,'support':self.support.clone(),'mu':mu,'action':mu.tanh()}
        return mu,ls

    def reference_forward(self,z,lesion=()):
        """Separately expressed receiver loop for algebraic correspondence testing."""
        hs=[]
        for i in range(3):
            unit=z.new_zeros((len(z),3));unit[:,i]=1
            hs.append(torch.tanh(F.linear(torch.cat((z,unit),-1),self.E.weight,self.E.bias)))
        mus=[]
        for j in range(3):
            r=z.new_zeros((len(z),8))
            for e,(i,jj) in enumerate(EDGES):
                if jj==j:
                    msg=torch.tanh(F.linear(torch.cat((hs[i],hs[j]),-1),self.A.weight,self.A.bias))
                    r=r+msg*self.support[i,j]*(0.0 if e in lesion else 1.0)/2
            g=torch.tanh(F.linear(torch.cat((hs[j],r),-1),self.B.weight,self.B.bias))
            mus.append(F.linear(g,self.D.weight,self.D.bias))
        return torch.cat(mus,-1)


class GenericActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(93,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,12))
        self.log_std=nn.Parameter(torch.full((12,),-0.5))
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,math.sqrt(2));nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight,0.01)

    def forward(self,z,lesion=(),trace=False):
        if lesion: raise ValueError('Non-graph controller has no edge lesion')
        mu=self.net(z)
        return mu,self.log_std.expand_as(mu)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(93,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,math.sqrt(2));nn.init.zeros_(layer.bias)
    def forward(self,z):return self.net(z).squeeze(-1)


def model_digest(model: nn.Module) -> str:
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n'
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(text);tmp.replace(path)


def core_tests() -> dict:
    torch.set_num_threads(1);torch.manual_seed(seed_for('qa','core'))
    assert len(masks())==42 and masks()[0]=='000000'
    assert len({tuple(matrix(x).flatten()) for x in masks()})==42
    z=torch.randn(8,93)
    counts=0;maxerr=0.0
    for mask in masks()+['111111']:
        a=RoutingActor(mask)
        mu,_,tr=a(z,trace=True)
        ref=a.reference_forward(z)
        err=float((mu-ref).abs().max().detach()); maxerr=max(maxerr,err)
        assert torch.allclose(mu,ref,rtol=0,atol=2e-7)
        for e,(i,j) in enumerate(EDGES):
            other,_,ot=a(z,(e,),True)
            assert torch.equal(tr['m'],ot['m']) and torch.equal(tr['h'],ot['h'])
            for node in range(3):
                if node!=j:assert torch.equal(tr['r'][:,node],ot['r'][:,node])
            expected=tr['r'][:,j]-tr['m'][:,e]*a.support[i,j]/2
            assert torch.allclose(expected,ot['r'][:,j],rtol=0,atol=2e-7)
            if mask[e]=='0':assert torch.equal(mu,other)
            counts+=1
        a.zero_grad();mu.square().sum().backward()
        grad=float(a.A.weight.grad.abs().sum())
        assert (grad==0) == (mask=='000000')
    pre=torch.randn(16,12,dtype=torch.float64);mu=torch.randn_like(pre);ls=torch.zeros_like(pre)
    td=torch.distributions.TransformedDistribution(torch.distributions.Normal(mu,ls.exp()),[torch.distributions.TanhTransform(cache_size=1)])
    assert torch.allclose(log_squashed_gaussian(pre,mu,ls),td.log_prob(pre.tanh()).sum(-1),atol=1e-9,rtol=0)
    # Mathematical identity only; NOT a native 16-channel catalogue map.
    pair=torch.rand(64,2,dtype=torch.float64);d=pair[:,0]-pair[:,1];q=pair.sum(1)
    back=torch.stack(((q+d)/2,(q-d)/2),1)
    assert torch.allclose(pair,back,atol=2e-16,rtol=0)
    seeds=[seed_for(s,i) for s in ('qa','train','selection','heldout','causal') for i in range(256)]
    assert len(set(seeds))==len(seeds)
    try:seed_for('final',0)
    except ValueError:pass
    else:raise AssertionError('Final namespace was accepted')
    return {'status':'PASS','route_cases':counts,'mask_count':42,'max_algebraic_error':maxerr,'density_jacobian':'PASS','gradient_paths':'PASS','mathematical_coordinate_identity':'PASS; not a catalogue realization','B1E_executed':False}

if __name__=='__main__':print(json.dumps(core_tests(),indent=2))
