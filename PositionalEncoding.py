import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model  # total positional encoding/embedding dimension is even- half ones for sine and hald ones for cosine

    def forward(self, timeteps):
        """
        In this context, batch size = number of timesteps provided at once.


        timesteps is tensor of shape [batchsize] like [0,1,2,...,T-1]
        Each element of timesteps is equivalent to pos in the original
        formula of "Attention is ALl You Need"
        Eg timesteps = torch.tensor([10, 50, 100])
        pos1=10, pos2= 50, pos3=100
        """

        device = timeteps.device
        halfed_d_model = (self.d_model // 2)  # half ones for sine and hald ones for cosine , halfed_d_model is denoted as i in original paper,

        # Computing the denominator part 10000^(2i/d_model)
        i = torch.arange(halfed_d_model, dtype=torch.float32, device=device) #  for d_model =6 i will be 0, 1, 2 only
        denominator_term = 10000 ** (2 * i / self.d_model)  # denomiator_term is like tensor([  1.0000,  21.5443, 464.1590])--- size: [halfed_d_model]

        pos = timeteps[:, None].float()
        """
        if timesteps = [1, 2, 3]  then it is converted into like 
        tensor([[1.],
                [2.],
                [3.]])   ----- size [batchsize,1]
        """

        # Now compute angles by pos/denominator_term, it is 2D matrix of [batchsize, 1] / [hal_d_model, 1] =[ batchsize, half_d_model]
        angles=pos/denominator_term

        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # needs to make changes here make even positions for sine and odd positions for cosine
        return encoding #shape[batch_size, d_model]
