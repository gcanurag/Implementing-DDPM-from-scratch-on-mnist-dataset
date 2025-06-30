import torch
from Unet import UNet
from DDPM import Denoising_Diffusion_Probabilistic_Model
from BetaScheduler import linear_beta_schedule

# ---- Match the training configuration exactly ----
T = 1000
betas = linear_beta_schedule(T)

model = UNet(
    in_ch=1,
    base_ch=64,
    mults=(1, 2),
    time_emb_dim=128,
    attn_res=(7,),
    image_shape=28
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("ddpm_mnist_50_epochs.pt"))
model.to(device)
model.eval()

# ---- Wrap with DDPM ----
ddpm = Denoising_Diffusion_Probabilistic_Model(model, betas)
ddpm.to(device)

# ---- Sample ----
with torch.no_grad():
    samples = ddpm.sample((1, 1, 28, 28))  # Generate 16 MNIST-like images

# ---- Visualize if needed ----
import matplotlib.pyplot as plt

samples = (samples.clamp(-1, 1) + 1) / 2  # Convert from [-1, 1] to [0, 1]
# grid = samples.view(1, 1, 28, 28).permute(0, 2, 1, 3).reshape(4*28, 4*28)

# plt.imshow(grid.cpu().numpy(), cmap="gray")
# plt.title("Generated Samples")
# plt.axis("off")
# plt.show()
image = samples.view(28, 28)

plt.imshow(image.cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title('Generated Sample')
plt.show()