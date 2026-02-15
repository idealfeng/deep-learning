import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid   # ✅ 加这一行
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# =============== 超参数 ===============
batch_size = 128
lr = 0.0002
epochs = 100
z_dim = 100  # 噪声维度
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============== 数据集 MNIST ===============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =============== 定义生成器 ===============
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# =============== 定义判别器 ===============
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(-1, 28 * 28))

# =============== 初始化模型 ===============
# 初始化
# 创建模型
G = Generator().to(device)
D = Discriminator().to(device)

# 初始化权重
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

G.apply(weights_init)
D.apply(weights_init)

optimizer_G = optim.Adam(G.parameters(), lr=0.0003, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# 训练循环
for epoch in range(epochs):
    for real, _ in dataloader:
        real = real.to(device)
        batch_size = real.size(0)

        real_label = torch.full((batch_size, 1), 0.9, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # === 1. 训练判别器 ===
        z = torch.randn(batch_size, z_dim, device=device)
        fake = G(z)
        loss_D_real = criterion(D(real), real_label)
        loss_D_fake = criterion(D(fake.detach()), fake_label)
        loss_D = (loss_D_real + loss_D_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # === 2. 训练生成器（多次） ===
        for _ in range(2):
            z = torch.randn(batch_size, z_dim, device=device)
            fake = G(z)
            loss_G = criterion(D(fake), real_label)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {loss_D:.4f} Loss_G: {loss_G:.4f}")

    # 可视化
    from torchvision.utils import make_grid

    # 可视化
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            test_z = torch.randn(64, z_dim, device=device)
            fake = G(test_z).cpu() * 0.5 + 0.5  # [-1,1] -> [0,1]
            grid = make_grid(fake, nrow=8)  # 拼成网格
            plt.imshow(grid.permute(1, 2, 0))  # 调整维度 HWC
            plt.title(f'Epoch {epoch + 1}')
            plt.axis('off')

            # 保存图像
            plt.savefig(f'generated_epoch_{epoch + 1}.png')  # 文件名可自定义
            plt.show()
            plt.close()  # 释放内存，避免窗口堆积

