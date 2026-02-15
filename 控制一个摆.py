import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# DDPG 超参数
actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.99
tau = 0.005
memory_size = 100000
batch_size = 128
episodes = 500  # DDPG 收敛更快


# Ornstein-Uhlenbeck 噪声（用于探索）
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# Actor 网络（策略网络）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 输出范围 [-1, 1]
        return x * self.max_action  # 缩放到 [-max_action, max_action]


# Critic 网络（Q 网络）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # Actor 网络
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic 网络
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 经验回放
        self.memory = deque(maxlen=memory_size)

        # 探索噪声
        self.noise = OUNoise(action_dim)
        self.max_action = max_action

    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()

        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < batch_size:
            return 0, 0

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in batch])).unsqueeze(1).to(self.device)

        # 更新 Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + gamma * target_q * (1 - dones)

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()


def train_ddpg():
    print("=" * 60)
    print("开始训练倒立摆智能体 (DDPG)")
    print("=" * 60)

    env = gym.make("Pendulum-v1", render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)

    reward_history = []
    critic_loss_history = []
    actor_loss_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        agent.noise.reset()

        total_reward = 0
        total_critic_loss = 0
        total_actor_loss = 0
        step_count = 0

        for step in range(200):
            action = agent.act(state, add_noise=True)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store_experience(state, action, reward, next_state, done or truncated)
            critic_loss, actor_loss = agent.learn()

            total_reward += reward
            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            step_count += 1

            state = next_state

            if done or truncated:
                break

        reward_history.append(total_reward)
        critic_loss_history.append(total_critic_loss / step_count)
        actor_loss_history.append(total_actor_loss / step_count)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(reward_history[-20:])
            print(f"Episode {episode + 1:3d}/{episodes} | "
                  f"奖励: {total_reward:7.2f} | "
                  f"平均: {avg_reward:7.2f} | "
                  f"Critic Loss: {total_critic_loss / step_count:.4f}")

    env.close()
    print("\n✅ 训练完成！")
    return agent, reward_history


def test_ddpg(agent):
    print("\n测试 DDPG 智能体...")
    env = gym.make("Pendulum-v1", render_mode="human")

    for ep in range(5):
        state, _ = env.reset()
        total_reward = 0

        for step in range(200):
            action = agent.act(state, add_noise=False)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(0.02)

            if done or truncated:
                break

        print(f"测试 {ep + 1}: 奖励 = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    agent, reward_history = train_ddpg()

    # 简单绘图
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.6)
    plt.plot(np.convolve(reward_history, np.ones(20) / 20, mode='valid'), 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DDPG 训练曲线')
    plt.grid(True)
    plt.savefig('ddpg_results.png')
    plt.show()

    test_ddpg(agent)