import torch
import math

def linear_beta_schedule(T, b1=1e-4, b2=0.02):
    return torch.linspace(b1, b2, T)

## TO_DO - fix it 
def cosine_beta_schedule(T, s=0.008):
    x = torch.linspace(0, T, T+1)
    alphas_cum = torch.cos(((x / T) + s) / (1 + s) * math.pi/2)**2
    alphas_cum /= alphas_cum[0]
    betas = 1 - alphas_cum[1:] / alphas_cum[:-1]
    return torch.clip(betas, 1e-5, 0.999)