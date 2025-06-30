import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from ResidualBlock import ResidualBlock
from AttentionBlock import AttentionBlock
from Up_Sample_Down_Sample import UpSample, DownSample


class UNet(nn.Module):
    def __init__(self,image_shape, in_ch=3,base_ch=128,mults=(1, 2, 2, 4),time_emb_dim=256,attn_res=(16, 8)):
        super().__init__()
        self.time_embedding_block = nn.Sequential(
            PositionalEncoding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),  ## need to understand why has it done so ==> TO_DO
        )

        # Channel setup
        channels = [base_ch * m for m in mults]  # chnnels=[128,256,256,512]

        # Initial input convolution 3channels ==> 128 channels
        self.input_conv = nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=3, padding=1)

        # DownPath (Encoder path)
        """
        Pattern here is : increase (or maintain) channels via Residual Blocks (optionally Attention Block) ==> then spatially downsample ==> repeat
        """
        self.down_blocks, self.downs = nn.ModuleList(), nn.ModuleList()
        res = image_shape
        c = base_ch
        for ch in channels:
            self.down_blocks.append(ResidualBlock(c, ch, time_emb_dim))
            res //= 2
            if res in attn_res:
                self.down_blocks.append(AttentionBlock(ch))
            self.downs.append(DownSample(ch))
            c = ch

        # Middle Blocks
        self.mid1 = ResidualBlock(c, c, time_emb_dim)  # here c has become c=512
        self.mid_attn = AttentionBlock(c)
        self.mid2 = ResidualBlock(c, c, time_emb_dim)

        # UpSample Path(Decoder path)
        self.up_blocks, self.ups = nn.ModuleList(), nn.ModuleList()
        for ch in reversed(channels):  # ch=[512,256,256,128]
            self.ups.append(UpSample(c))
            self.up_blocks.append(ResidualBlock(c + ch, ch, time_emb_dim))
            res *= 2
            if res in attn_res:
                self.up_blocks.append(AttentionBlock(ch))
            c = ch

        # Final ouput convolution block
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, c),
            nn.SiLU(),
            nn.Conv2d(in_channels=c, out_channels=in_ch, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        """
        x: inpupt image
        t is timestep a scaler of [batchsize] or [batchsize,1]
        """
        time_embedding_vector = self.time_embedding_block(t)
        h = self.input_conv(x)

        # Encoder
        skips = []  # to store residual blokcs form where we take skip connections

        for i, block in enumerate(self.down_blocks):
            h = (block(h, time_embedding_vector)if isinstance(block, ResidualBlock)else block(h))
            """
            if block is instance of Residual Block
                from here we are actually calling the forward of ResidualBlock class by passing 
                input image which is h here and time_embedding_vecotr

            else 
                from here we are calling AttentionBlock and passing just input image
            """

            if isinstance(block, ResidualBlock):
                skips.append(h)  # we are storing the current image or current feature map for skip connections

            if i < len(self.downs):
                h = self.downs[i](h)

        # Mid Path (BottleNeck)
        h = self.mid1(h, time_embedding_vector)
        # print("h shape after mid 1", h.shape)
        h = self.mid_attn(h)
        # print("h shape after mid attn", h.shape)
        h = self.mid2(h, time_embedding_vector)
        # print("h shape after mid 2", h.shape)

        # Upsampling path (Decoder path)

        for i, block in enumerate(self.up_blocks):
            if i < len(self.ups):
                h = self.ups[i](h)

            if isinstance(block, ResidualBlock):
                skip = skips.pop()
                # print(f" h.shape = {h.shape}, skip.shape = {skip.shape}")
                
                h = torch.cat([h, skip], dim=1)
                """"
                the tensor h and skip will be like [B,C,H,W] dim=1 means along C ie channel. 
                So it means concatenating along dimensionof channel hence 
                the final size will be [B, c_h+c_skip, H, E]  ==> not necessary c_h=c_skip
                """
                h = block(h, time_embedding_vector)  # Calling Residual Block

            else:
                h = block(h)  # Calling Attention Block
        return self.output_conv(h)
