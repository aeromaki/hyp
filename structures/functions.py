import torch

EPS = 1e-06

def pseudo_polar(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    return (x / (r := x.norm(dim=-1, keepdim=True) + EPS), r)

def polar_hyperboloid(
    d: torch.Tensor,
    r: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    return (r.sinh() * d, r.cosh())

def hyperboloid(
        x: torch.Tensor,
        from_klein: bool = False
    ) -> (torch.Tensor, torch.Tensor):
    if from_klein:
        return rev_klein_hyp(x)
    else:
        d, r = pseudo_polar(x)
        return polar_hyperboloid(d, r)

def hyper_klein(
    v: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    return v / (n + EPS)

def klein(x: torch.Tensor) -> torch.Tensor:
    v, n = hyperboloid(x)
    return hyper_klein(v, n)

def lorentz_factor(
    v: torch.Tensor,
) -> torch.Tensor:
    return 1 / torch.sqrt(1 - v.norm(dim=-1, keepdim=True).pow(2) + EPS)

def rev_klein_hyp(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    lorentz = lorentz_factor(x)
    return (lorentz * x, lorentz)

def rev_klein(x: torch.Tensor) -> torch.Tensor:
    d, r = rev_klein_hyp(x)
    pr = r.clamp(min=1+EPS).arccosh()

    rsinh = r.sinh()
    pd = d / (rsinh.sign() * rsinh.abs().clamp(min=EPS))

    return pr * pd

def minkowski(
    q: (torch.Tensor, torch.Tensor),
    k: (torch.Tensor, torch.Tensor)
) -> torch.Tensor:
    bi = q[0] @ k[0].transpose(-2, -1) - q[1] @ k[1].transpose(-2, -1)
    return bi

def dist_h(
    q: (torch.Tensor, torch.Tensor),
    k: (torch.Tensor, torch.Tensor),
) -> torch.Tensor:
    d = -minkowski(q, k)
    d = d.clamp(min=1+EPS)
    return torch.arccosh(d)

def einstein_midpoint(
    alpha: torch.Tensor,
    v_k: torch.Tensor,
) -> torch.Tensor:
    gamma = lorentz_factor(v_k)
    w = alpha * gamma.transpose(-2, -1)
    w_n = w / (w.sum(dim=-1, keepdim=True) + EPS)
    return (w_n.unsqueeze(-1) * v_k.unsqueeze(-3)).sum(dim=-2)