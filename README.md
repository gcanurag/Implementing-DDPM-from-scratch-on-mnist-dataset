
# DDPM - Denoising Diffusion Probabilistic Model (MNIST)

This project implements a full DDPM from scratch using PyTorch, closely following the original DDPM paper. The model has been trained on the **MNIST** dataset.

## Features

- ✅ Full U-Net architecture with residual blocks and self-attention
- ✅ Sinusoidal positional encoding
- ✅ Linear & cosine noise schedulers
- ✅ EMA (Exponential Moving Average) for stability
- ✅ Sampling using DDPM reverse process
- ✅ Model and sample image saving every 10 epochs
- ✅ Full checkpointing in `.pt` and `.safetensors` formats

## Dataset

- **mnist**: 
- Input size: **1 × 28 × 28**
- Normalized to `[-1, 1]`

## Architecture

- U-Net with:
  - `base_ch = 64`
  - `mults = (1, 2)`
  - Self-attention at resolutions: 7x7
- Time embedding: sinusoidal + MLP

## Training

```bash
python train.py
```

- Uses Adam optimizer (lr = 2e-4)
- Trains for 50 epochs by default
- Saves: ddpm_mnist_i_epochs.pt


## Sample Output

<p align="center">
  <img src="result/after 40 epochs_1.png" width="500"/>
</p>

<p align="center">
  <img src="result/after 40 epochs_2.png" width="500"/>
</p>
## How to Load a Saved Model

```python
from safetensors.torch import load_file

state_dict = load_file("checkpoints/ddpm_epoch_200.safetensors")
model.load_state_dict(state_dict)
```

## References

- [DDPM Paper (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)


---

Developed with ❤️ for academic purposes.  
Author: Anurag G.C.
