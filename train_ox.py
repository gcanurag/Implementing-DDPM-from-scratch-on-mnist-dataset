import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from EMA import EMA
from Unet import UNet
from DDPM import Denoising_Diffusion_Probabilistic_Model
from BetaScheduler import linear_beta_schedule, cosine_beta_schedule
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from pathlib import Path
import torchvision.utils as vutils



# ----------- Configuration -----------
epochs = 50
batch_size = 128
img_size = 28
val_split = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# saving directory confg
save_dir = Path("./checkpoints_pet/")
save_dir.mkdir(parents=True, exist_ok=True)
sample_dir = save_dir / "samples"
sample_dir.mkdir(exist_ok=True)

# ----------- Dataset -----------

transform = Compose([
    Resize(img_size),
    CenterCrop(img_size),
    ToTensor(),
    Normalize([0.5]*3, [0.5]*3)  # [-1, 1] normalization for RGB
])

dataset = datasets.OxfordIIITPet(root="./data", split='test', download=False, transform=transform) # make it download=True when downloading dataset for first time
val_len = int(len(dataset) * val_split)
train_len = len(dataset) - val_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# Get one batch of images and labels from the train_loader
images, labels = next(iter(train_loader))
image_shape=images.shape[1:] # shape= (3, 28, 28) we are extracting the last one



# ----------- UNet / DDPM Setup -----------
T = 1000
betas = linear_beta_schedule(T)  # explore about cosine beta schedule also
model = UNet(
    image_shape=image_shape[-1],
    in_ch=3,
    base_ch=128,               # we need to reduce base channels for MNIST
    mults=(1, 2),             # only two levels of downsample
    time_emb_dim=128,        # smaller time embedding is fine
    attn_res=(7,)            # add attention at 7×7
).to(device)

ddpm = Denoising_Diffusion_Probabilistic_Model(model, betas).to(device)
optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
ema = EMA(model)

# ----------- Training Loop -----------
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            t = torch.randint(0, T, (imgs.size(0),), device=device).long() # Randomly sample timestep t
            loss = ddpm.loss_function(imgs, t) # Compute diffusion loss
            total_loss += loss.item() * imgs.size(0) # Return average loss over the dataset
    return total_loss / len(loader.dataset)


# def save_sample_images(epoch, ema_model, image_shape):
#     ema_model.apply_to(model)
#     model.eval()
#     with torch.no_grad():
#         samples = ddpm.sample((16, *image_shape), ema=ema_model)  # 16 images
#         samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
#         grid = vutils.make_grid(samples, nrow=4)
#         vutils.save_image(grid, sample_dir / f"epoch_{epoch:03d}.png")


def save_sample_images(epoch, ema_model, image_shape):
    backup = model.state_dict()  # save current model state
    ema_model.apply_to(model)    # replace with EMA weights temporarily
    model.eval()

    with torch.no_grad():
        samples = ddpm.sample((16, *image_shape), ema=ema_model)
        samples = (samples + 1) / 2
        grid = vutils.make_grid(samples, nrow=4)
        vutils.save_image(grid, sample_dir / f"epoch_{epoch:03d}.png")

    model.load_state_dict(backup)  # restore original training weights



def train():
    for epoch in range(1, epochs + 1):
        ddpm.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for imgs, _ in pbar:
            imgs = imgs.to(device)
            t = torch.randint(0, T, (imgs.size(0),), device=device).long()
            loss = ddpm.loss_function(imgs, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            total_train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(train_loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = evaluate(ddpm, val_loader)

        print(f"\nEpoch {epoch} Completed:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if epoch % 10 == 0:
            print(f"Saving checkpoint at epoch {epoch}...")

            # Save model components
            torch.save({
                "unet_state_dict": model.state_dict(),
                "ddpm_state_dict": ddpm.state_dict(),
                "ema_shadow": ema.shadow
            }, save_dir / f"ddpm_checkpoint_epoch_{epoch:03d}.safetensors")

            # Save generated samples
            save_sample_images(epoch, ema, image_shape)

train()
