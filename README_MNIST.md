# DDPM - Denoising Diffusion Probabilistic Model (MNIST)

This project implements a full DDPM from scratch using PyTorch, closely following the original DDPM paper. The model has been trained on the **MNIST** handwritten digit dataset.

## Features

- ✅ Full U-Net architecture with residual blocks and optional self-attention
- ✅ Sinusoidal positional encoding
- ✅ Linear & cosine noise schedulers
- ✅ EMA (Exponential Moving Average) for stable training
- ✅ Sampling using DDPM reverse process
- ✅ Checkpointing every 10 epochs
- ✅ Sample image generation every 10 epochs

## Dataset

- **MNIST**: 28×28 grayscale handwritten digits
- Input size: **1 × 28 × 28**
- Normalized to `[-1, 1]`

## Architecture

- U-Net with:
  - `base_ch = 64`
  - `mults = (1, 2)` (2 levels of downsampling)
  - Optional attention at 7×7 resolution (can be disabled for MNIST)
- Time embedding: sinusoidal + MLP

## Training

```bash
python train.py
```

- Uses Adam optimizer (lr = 2e-4)
- Trains for 200 epochs by default
- Saves:
  - UNet EMA weights: `checkpoints/unet_ema_epoch_X.pt`
  - EMA state dict: `checkpoints/ema_shadow_epoch_X.safetensors`
  - Full DDPM model: `checkpoints/ddpm_epoch_X.safetensors`
  - Samples: `samples/sample_epoch_X.png`

## Sample Output

<p align="center">
  <img src="samples/sample_epoch_200.png" width="300"/>
</p>

## How to Load a Saved Model

```python
from safetensors.torch import load_file

state_dict = load_file("checkpoints/ddpm_epoch_200.safetensors")
model.load_state_dict(state_dict)
```

## References

- [DDPM Paper (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [OpenAI's Guided Diffusion repo](https://github.com/openai/guided-diffusion)
- [Lucidrains' `denoising-diffusion-pytorch`](https://github.com/lucidrains/denoising-diffusion-pytorch)

---

Developed with ❤️ for academic and learning purposes.
