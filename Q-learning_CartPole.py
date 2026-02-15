import numpy as np
import gymnasium as gym
import random

# 创建CartPole环境，指定渲染模式为 "human"
env = gym.make("CartPole-v1", render_mode="human")

# 初始化Q表
state_space_size = [20, 20]  # 状态空间的离散化维度
action_space_size = env.action_space.n  # 动作空间的大小
q_table = np.zeros(state_space_size + [action_space_size])

# 离散化状态空间
def discretize_state(state):
    # 创建离散化的bins
    state_bins = [np.linspace(-x, x, num=n) for x, n in
                  zip([env.observation_space.high[0], 1.0, 0.5, 1.0], state_space_size)]

    # 单独对每个状态元素进行digitize
    state_discretized = [int(np.digitize(x, bins)) for x, bins in zip(state, state_bins)]

    return tuple(state_discretized)

# 超参数
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1  # 探索的概率
episodes = 1000
max_steps = 200

# Q-learning 算法
for episode in range(episodes):
    state, _ = env.reset()  # 重置环境
    state = discretize_state(state)  # 离散化状态
    done = False

    for step in range(max_steps):
        # 选择动作
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 随机选择动作（探索）
        else:
            action = np.argmax(q_table[state])  # 选择当前最优动作（利用）

        # 执行动作
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)

        # 更新Q值
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += learning_rate * (
                    reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action])

        state = next_state

        if done:
            break

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1} completed")

# 测试模型
state, _ = env.reset()
state = discretize_state(state)
done = False
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _, _ = env.step(action)
    state = discretize_state(state)
    env.render()  # 显示环境
env.close()
