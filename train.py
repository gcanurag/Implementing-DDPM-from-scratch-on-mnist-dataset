import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from EMA import EMA
from Unet import UNet
from DDPM import Denoising_Diffusion_Probabilistic_Model
from BetaScheduler import linear_beta_schedule, cosine_beta_schedule



# ----------- Configuration -----------
epochs = 50
batch_size = 128
img_size = 28
save_path = "./ddpm_mnist.pt"
val_split = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Dataset -----------
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),              # [0, 1]
    transforms.Lambda(lambda x: x * 2 - 1)  # => [-1, 1]
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
val_len = int(len(dataset) * val_split)
train_len = len(dataset) - val_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# Get one batch of images and labels from the train_loader
images, labels = next(iter(train_loader))
image_shape=images.shape[2]


# ----------- UNet / DDPM Setup -----------
T = 1000
betas = linear_beta_schedule(T)  # explore about cosine beta schedule also
model = UNet(
    in_ch=1,
    base_ch=64,               # we need to reduce base channels for MNIST
    mults=(1, 2),             # only two levels of downsample
    time_emb_dim=128,        # smaller time embedding is fine
    attn_res=(7,),            # add attention at 7×7
    image_shape=image_shape
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

def train():
    """
    pbar wraps train_loader, which is an iterable (the batches of your training data).
    Now, when we write:
    for imgs, _ in pbar: ==>It is looping over batches just like usual, but tqdm shows a real-time progress bar in the console.
    """
    for epoch in range(1, epochs + 1):
        ddpm.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            t = torch.randint(0, T, (imgs.size(0),), device=device).long() # Randomly sample timestep t
            loss = ddpm.loss_function(imgs, t) # Compute diffusion loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            total_train_loss += loss.item() * imgs.size(0)# Return average loss over the dataset
            pbar.set_postfix(train_loss=loss.item()) # Adds extra info (like current training loss) to the progress bar display.

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = evaluate(ddpm, val_loader)

        print(f"\nEpoch {epoch} Completed:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if epoch % 10 == 0:
            print(f"Saving model at epoch {epoch}...")
            torch.save(model.state_dict(), save_path)

train()
