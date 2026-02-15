import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ==================== 1. DQN 网络 ====================
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.fc(x)


# ==================== 2. 经验回放 ====================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ==================== 3. 训练配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

env = gym.make('MountainCar-v0')

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayBuffer(capacity=10000)

# 超参数
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
gamma = 0.99
batch_size = 64
target_update = 10

# ==================== 4. 记录训练数据 ====================
episode_rewards = []
episode_steps = []
success_episodes = []
epsilon_history = []

# ==================== 5. 训练过程 ====================
print("=" * 50)
print("开始 DQN 训练...")
print("=" * 50)

num_episodes = 500
success_count = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(200):
        # Epsilon-greedy 策略
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax().item()

        next_state, reward, done, truncated, _ = env.step(action)

        # 轻微的奖励塑造
        position, velocity = next_state
        shaped_reward = reward + abs(velocity) * 5

        memory.push(state, action, shaped_reward, next_state, done)
        state = next_state
        total_reward += reward

        # 训练网络
        if len(memory) >= batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # Q-learning 更新
            current_q = policy_net(states).gather(1, actions.unsqueeze(1))
            next_q = target_net(next_states).max(1)[0].detach()
            target_q = rewards + gamma * next_q * (1 - dones)

            loss = nn.MSELoss()(current_q.squeeze(), target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            success_count += 1
            success_episodes.append(episode)
            break

    # 记录数据
    episode_rewards.append(total_reward)
    episode_steps.append(step + 1)
    epsilon_history.append(epsilon)

    # 衰减 epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # 更新目标网络
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 打印进度
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        avg_steps = np.mean(episode_steps[-50:])
        recent_success = sum(1 for e in success_episodes if e >= episode - 49)
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"成功率: {recent_success}/50 | "
              f"平均奖励: {avg_reward:.1f} | "
              f"平均步数: {avg_steps:.1f} | "
              f"ε: {epsilon:.3f}")

print("\n" + "=" * 50)
print(f"训练完成！总成功次数: {success_count}/{num_episodes}")
print("=" * 50)

# ==================== 6. 绘制训练曲线 ====================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('DQN 训练过程可视化', fontsize=16, fontweight='bold')

# 子图1: 每轮奖励
axes[0, 0].plot(episode_rewards, alpha=0.6, label='每轮奖励')
axes[0, 0].plot(np.convolve(episode_rewards, np.ones(50) / 50, mode='valid'),
                'r-', linewidth=2, label='50轮均值')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].set_title('奖励曲线')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 每轮步数
axes[0, 1].plot(episode_steps, alpha=0.6, label='每轮步数')
axes[0, 1].plot(np.convolve(episode_steps, np.ones(50) / 50, mode='valid'),
                'g-', linewidth=2, label='50轮均值')
axes[0, 1].axhline(y=110, color='r', linestyle='--', label='人类水平')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Steps')
axes[0, 1].set_title('步数曲线（越低越好）')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 子图3: 成功率统计
window = 50
success_rate = []
for i in range(len(success_episodes)):
    if i < window:
        rate = len([e for e in success_episodes if e <= i]) / (i + 1) * 100
    else:
        rate = len([e for e in success_episodes if e > i - window and e <= i]) / window * 100
    success_rate.append(rate)

if success_rate:
    axes[1, 0].plot(success_rate, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title(f'成功率（滑动窗口={window}）')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 105])

# 子图4: Epsilon 衰减
axes[1, 1].plot(epsilon_history, 'purple', linewidth=2)
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Epsilon')
axes[1, 1].set_title('探索率衰减')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dqn_training_results.png', dpi=300, bbox_inches='tight')
print("\n📊 训练曲线已保存为 'dqn_training_results.png'")
plt.show()

# ==================== 7. 保存模型 ====================
torch.save({
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
}, 'mountaincar_dqn.pth')
print("💾 模型已保存为 'mountaincar_dqn.pth'")

# ==================== 8. 测试阶段 ====================
print("\n" + "=" * 50)
print("开始测试（带可视化）...")
print("=" * 50)

env.close()
env = gym.make('MountainCar-v0', render_mode='human')

for test_num in range(3):
    print(f"\n🎮 测试 {test_num + 1}/3:")
    state, _ = env.reset()
    total_reward = 0

    for step in range(300):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).argmax().item()

        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        time.sleep(0.02)  # 放慢速度便于观察

        if done:
            print(f"✅ 成功冲顶！用时 {step + 1} 步，总奖励: {total_reward:.0f}")
            time.sleep(1)
            break
    else:
        print(f"❌ 未成功，用时 {step + 1} 步，总奖励: {total_reward:.0f}")

env.close()
print("\n" + "=" * 50)
print("所有测试完成！")
print("=" * 50)