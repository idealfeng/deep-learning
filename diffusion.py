import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- 超参数 -------------------
batch_size = 128
lr = 1e-4
epochs = 100
img_size = 28
timesteps = 1000
base_ch = 64
save_dir = "ddpm_results"
os.makedirs(save_dir, exist_ok=True)

# ------------------- 数据集 -------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # [-1, 1]
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------- cosine beta schedule -------------------
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, 0.0001, 0.999), dtype=torch.float32)

beta = cosine_beta_schedule(timesteps).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# ------------------- 时间嵌入 -------------------
def timestep_embedding(timesteps, dim=128):
    half = dim // 2
    emb = np.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

# ------------------- 残差块（带时间注入） -------------------
class ResidualBlockWithTime(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.relu = nn.ReLU()
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.relu(h))
        res = x if self.res_conv is None else self.res_conv(x)
        return self.relu(h + res)

# ------------------- UNet -------------------
class UNet(nn.Module):
    def __init__(self, time_dim=128, base_ch=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        self.down1 = ResidualBlockWithTime(1, base_ch, time_dim)
        self.down2 = ResidualBlockWithTime(base_ch, base_ch*2, time_dim)
        self.down3 = ResidualBlockWithTime(base_ch*2, base_ch*4, time_dim)

        self.up1 = ResidualBlockWithTime(base_ch*4, base_ch*2, time_dim)
        self.up2 = ResidualBlockWithTime(base_ch*4, base_ch, time_dim)  # skip
        self.up3 = ResidualBlockWithTime(base_ch*2, base_ch, time_dim)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        t_emb = timestep_embedding(t, 128)
        t_emb = self.time_mlp(t_emb)

        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)

        u1 = self.up1(self.upsample(d3), t_emb)
        u2 = self.up2(torch.cat([u1, d2], dim=1), t_emb)
        u3 = self.up3(torch.cat([self.upsample(u2), d1], dim=1), t_emb)
        return self.out_conv(u3)

# ------------------- 前向扩散 -------------------
def q_sample(x0, t):
    noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise, noise

# ------------------- 采样（DDPM） -------------------
def p_sample(model, x, t):
    t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
    pred_noise = model(x, t_batch)
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    if t > 0:
        noise = torch.randn_like(x)
    else:
        noise = torch.zeros_like(x)
    x_prev = (1 / torch.sqrt(alpha_t)) * (
        x - (beta[t] / torch.sqrt(1 - alpha_bar_t)) * pred_noise
    ) + torch.sqrt(beta[t]) * noise
    return x_prev

def sample(model, n=64):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n, 1, img_size, img_size, device=device)
        for t in reversed(range(timesteps)):
            x = p_sample(model, x, t)
        x = (x + 1) / 2
        grid = make_grid(x, nrow=8)
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis("off")
        plt.show()

# ------------------- 训练 -------------------
model = UNet(base_ch=base_ch).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
mse = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)
        x_t, noise = q_sample(imgs, t)

        pred_noise = model(x_t, t)
        loss = mse(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    # 每10个epoch保存一次生成图片
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            x = torch.randn(64, 1, img_size, img_size, device=device)
            for t_ in reversed(range(timesteps)):
                x = p_sample(model, x, t_)
            x = (x + 1) / 2  # [-1,1] -> [0,1]

            grid = make_grid(x, nrow=8)
            plt.figure(figsize=(6,6))
            plt.imshow(grid.cpu().permute(1,2,0))
            plt.axis("off")
            plt.savefig(f"{save_dir}/sample_epoch_{epoch+1}.png")
            plt.close()
