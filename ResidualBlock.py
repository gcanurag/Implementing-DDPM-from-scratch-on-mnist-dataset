import torch
import torch.nn as nn

""""
# Parameters
time_emb_dim = 8
out_channels = 4
batch_size = 1

Step 1 - Sinusoidal Time Embedding:
tensor([[-9.5892e-01,  2.3000e-01,  1.0772e-02,  5.0000e-04,  2.8366e-01,
          9.7319e-01,  9.9994e-01,  1.0000e+00]])   (Given the time_emb_dimension or positional encoding dimesntion is 8)

Step 2 - Projected Time Embedding (via Linear):
tensor([[ 0.3463, -0.3599, -0.0818, -0.3526]], grad_fn=<AddmmBackward0>)

Step 3 - Time Embedding After Reshape for CNN Addition:
tensor([[[[ 0.3463]],

         [[-0.3599]],

         [[-0.0818]],

         [[-0.3526]]]], grad_fn=<UnsqueezeBackward0>)

Original Feature Map: size (B, in_ch, H, W)
tensor([[[[ 0.8330,  0.3738],
          [ 1.2985, -0.2803]],

         [[-0.3292, -0.2963],
          [-0.6828, -0.1154]],

         [[ 0.7259,  0.4507],
          [-0.1744, -1.3573]],

         [[ 0.8100, -1.6058],
          [ 0.7939, -0.3663]]]])

when original fearure map is taken through block1 it will be of shape (B, out_ch, H, W)

Output Feature Map After Adding Time Embedding:
tensor([[[[ 1.1793,  0.7201],
          [ 1.6448,  0.0660]],

         [[-0.6891, -0.6562],
          [-1.0427, -0.4753]],

         [[ 0.6441,  0.3689],
          [-0.2562, -1.4391]],

         [[ 0.4574, -1.9584],
          [ 0.4412, -0.7190]]]], grad_fn=<AddBackward0>)
"""


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):  # time_emb_dim vaneko positional encoding vector ko dimension ho
        super().__init__()

        self.time_embedding_projection_mlp = nn.Linear(in_features=time_emb_dim, out_features=out_ch)   # takes an input of size time_emb_dim and then converts to size of out_ch

        # First Block
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
        )

        # padding=1 is to  Preserve spatial size, note that padding=1 is not as padding=same in keras.
        # in keras if we make padding=same then keras tries to maintain size of feature map as same as input image by adjusting
        # the value of padding itself. But making size of feature map as equal to image will always not be possible like in case
        # of image of 256x256 kernel size 4x4 stride =1

        # Second Block
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),  # padding=1 is to  Preserve spatial size
        )

        self.ResCon = (nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)if in_ch != out_ch else nn.Identity())

    def forward(self, x, time_embedding_vector):
        """
        x: feature map tensor is of shape (batchsize, in_ch, H, W)
        time_embedding_vector: time embedding tensor of shape (batchsize, time_emb_dim) or [batchsize, d_model] as in PositionalEncoding.py
        """

        h = self.block1(x)  # it outputs tensor of shape: (B, out_ch, H, W)

        # Add projected time embedding , time embedding vaneko positional encoding ho
        projected_time_emb = self.time_embedding_projection_mlp(time_embedding_vector)  # outputs tensor of shape # shape: (B, out_ch)

        h = h + projected_time_emb[:, :, None, None]  # shape: (B, out_ch, H, W)

        h = self.block2(h) # shape: (B, out_ch, H, W)

        # Add the (possibly transformed) residual connection
        return h + self.ResCon(x)   # shape: (B, out_ch, H, W), we have preserved spatial dimensions unlike as shown in popular Unet diagram where it shows to decrease H, W within Residual Block
