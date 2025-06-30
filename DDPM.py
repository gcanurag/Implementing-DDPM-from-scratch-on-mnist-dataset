import torch
import torch.nn as nn
import torch.nn.functional as F

"""
forward process: q(x_t | x_0) = sqrt(ā_t)*x_0 + sqrt(1 - ā_t)*noise
"""

class Denoising_Diffusion_Probabilistic_Model(nn.Module):
    def __init__(self, model, betas):
        super().__init__()

        self.unet=model # our backbone model is unet
        self.device=next(model.parameters()).device

        # Noise schedule: beta_t 
        self.betas=betas.to(self.device) # shape [T] T=no of timesteps
        self.alphas=1-self.betas # α_t = 1 - β_t
        self.alpha_cumulative_prd=torch.cumprod(self.alphas, dim=0) # making cum prod across rows \bar{α}_t = product of α_1 to α_t

        # Precomputing square roots of alpha_cumulative_prd for fast access in future
        self.sqrt_alpha_cumulative_prd=torch.sqrt(self.alpha_cumulative_prd) #  √\bar{α}_t
        self.sqrt_one_minus_alpha_cumulative_prod= torch.sqrt(1-self.alpha_cumulative_prd) # √(1 - \bar{α}_t)


    # define forward diffusion process
    def forward_diffusion(self, x0, t):
        """
        t is expected to be a 1D tensor (a vector) of shape [batch_size]
        """
        noise=torch.randn_like(x0) # this generates a gaussian noise with same shape as x0 which can be added to x0

        sqrt_alpha_cumulative_prd=self.alpha_cumulative_prd[t][:, None, None, None] # shape [B,1,1,1],  
        sqrt_one_minus_alpha_cumulative_prod=self.sqrt_one_minus_alpha_cumulative_prod[t][:, None, None, None]

        x_t=sqrt_alpha_cumulative_prd*x0+sqrt_one_minus_alpha_cumulative_prod*noise #  sqrt(ā_t)*x_0 + sqrt(1 - ā_t)*noise

        return x_t, noise
    
    def loss_function(self, x0, t):
        """
        t is vector of shape [batchsize]
        """
        x_t, noise= self.forward_diffusion(x0, t) # create x_t and target noise
        predicted_noise=self.unet(x_t, t)
        return F.mse_loss(predicted_noise, noise)
    

    def sample(self, shape, ema=None):
        # start with pure noise
        x=torch.randn(shape, device=self.device)  # shape = shape of generated images you want to sample, typically shape = (batch_size, channels, height, width)

        if ema:
            ema.apply_to(self.unet)

        # total number of timeseteps =T
        T=len(self.betas)

        for t in reversed(range(T)): # reverse diffusion from T-1 to 0
            t_batch=torch.full((shape[0],), t, dtype=torch.long, device=self.device)
            """
            shape = (4, 3, 64, 64)  # Want to generate 4 images
            t = 1000 - 1 = 999      # Current timestep in the reverse loop
            t_batch = torch.full((4,), 999, dtype=torch.long, device='cuda')
            t_batch = tensor([999, 999, 999, 999], device='cuda:0')  # shape [4]

            This means 
            “We are denoising all 4 images at timestep 999.”
            """

            predicted_noise=self.unet(x, t_batch)

            beta_t=self.betas[t]
            alpha_t=self.alphas[t]
            alpha_bar_t=self.alpha_cumulative_prd[t]
            
            # For all but the last step, add some noise again, this is written as z in paper in sampling algorithm
            if t>0:
                noise=torch.randn_like(x)
            else:
                noise=0
            
            # Compute q(x_{t-1} | x_t, x_0)
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            x = coef1 * (x - coef2 * predicted_noise) + torch.sqrt(beta_t) * noise # this is DDPM reverse equation in paper they have σ^2_t=β_t

        return x


