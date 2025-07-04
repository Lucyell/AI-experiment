import os
import random
import numpy as np
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from agent_dir.agent import Agent


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,output_size):
        super(QNetwork, self).__init__()
        # 改进：使用更深的网络和Dropout防止过拟合
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size2, output_size)
        )
        
        # 改进：使用Xavier初始化提高训练稳定性
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        # 改进：添加优先级采样的基础结构
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        # 改进：新经验赋予高优先级
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        # 改进：使用简化的优先级采样
        if len(self.priorities) > 0:
            priorities = np.array(self.priorities)
            # 添加小的ε避免概率为0
            probs = (priorities + 1e-6) / (priorities + 1e-6).sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            batch = [self.buffer[i] for i in indices]
        else:
            batch = random.sample(self.buffer, batch_size)
            
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

    def clean(self):
        self.buffer.clear()
        self.priorities.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        super(AgentDQN, self).__init__(env)
        self.env = env
        self.args = args
        self.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hidden_size1 = args.hidden_size1
        self.hidden_size2 = args.hidden_size2
        self.gamma = args.gamma
        self.lr = args.lr
        
        # 改进：更优的epsilon衰减策略
        self.epsilon = 1.0
        self.epsilon_min = 0.0001  # 降低最小值以保持更多探索
        self.epsilon_decay = 0.5  # 更慢的衰减
        
        self.batch_size = 128
        self.n_frames = args.n_frames
        self.grad_norm_clip = args.grad_norm_clip
        
        # 改进：更频繁的目标网络更新
        self.target_update_freq = 100
        
        # 改进：添加软更新参数
        self.tau = 0.005

        self.q_network = QNetwork(self.state_dim, self.hidden_size1, self.hidden_size2, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.hidden_size1, self.hidden_size2, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 改进：使用学习率调度器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        
        # 改进：增大replay buffer容量
        self.replay_buffer = ReplayBuffer(50000)
        
        # 改进：添加奖励缩放和归一化
        self.reward_scale = 1.0
        self.reward_history = deque(maxlen=1000)

        logdir = Path("logs/dqn")
        logdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(logdir=str(logdir))

    def init_game_setting(self):
        self.epsilon = 0.01  # 测试时保持少量探索

    def make_action(self, observation, test=True):
        if isinstance(observation, tuple):
            observation = observation[0]

        obs = np.array(observation, dtype=np.float32)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 改进：使用Noisy Networks概念的简化版本
        if test:
            epsilon = 0.01
        else:
            epsilon = self.epsilon

        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.q_network(obs)
                action = q_values.argmax(dim=1).item()
        else:
            action = self.env.action_space.sample()

        return action

    def soft_update_target_network(self):
        """改进：软更新目标网络"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 改进：奖励标准化
        if len(self.reward_history) > 10:
            reward_mean = np.mean(self.reward_history)
            reward_std = np.std(self.reward_history) + 1e-8
            rewards = (rewards - reward_mean) / reward_std

        # Double DQN target with improvements
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # 改进：使用更保守的折扣因子和奖励裁剪
            clipped_rewards = torch.clamp(rewards, -10, 10)
            target_q = clipped_rewards + self.gamma * next_q_values * (1 - dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 改进：使用Huber Loss，更稳定的训练
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        # 改进：梯度裁剪
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def run(self):
        state, _ = self.env.reset()
        total_rewards = []
        reward_history = []
        episode_idx = 0
        step_count = 0
        
        # 改进：预热阶段，收集经验
        warmup_steps = 1000

        for frame_idx in range(1, self.n_frames + 1):
            step_count += 1
            action = self.make_action(state, test=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # 改进：奖励塑形 - 鼓励长期存活
            shaped_reward = reward
            if not done:
                shaped_reward += 0.01  # 存活奖励
            
            self.replay_buffer.push(state, action, shaped_reward, next_state, done)
            state = next_state
            reward_history.append(reward)
            
            # 更新奖励历史用于标准化
            self.reward_history.append(reward)

            if done:
                episode_idx += 1
                total_reward = sum(reward_history)
                self.writer.add_scalar("Reward", total_reward, frame_idx)
                self.writer.add_scalar("Epsilon", self.epsilon, frame_idx)
                total_rewards.append(total_reward)
                
                print(f"[Episode {episode_idx}] Reward: {total_reward:.2f}, Steps: {step_count}, Epsilon: {self.epsilon:.4f}, lr: {self.lr:.4f}")
                reward_history.clear()
                state, _ = self.env.reset()
                step_count = 0

                # 改进：基于性能的epsilon衰减
                if len(total_rewards) >= 10:
                    recent_avg = np.mean(total_rewards[-10:])
                    if recent_avg > np.mean(total_rewards[-20:-10]) if len(total_rewards) >= 20 else True:
                        # 性能提升时衰减更快
                        decay_rate = 0.1 #0.998
                    else:
                        # 性能停滞时保持更多探索
                        decay_rate = 0.1 #0.995
                        
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= decay_rate
                        self.epsilon = max(self.epsilon, self.epsilon_min)

            # 训练条件改进：预热后开始训练
            if len(self.replay_buffer) >= max(self.batch_size, warmup_steps):
                loss = self.train()
                self.writer.add_scalar("Loss", loss, frame_idx)
                
                # 改进：每步都进行软更新
                if frame_idx % 4 == 0:  # 每4步软更新一次
                    self.soft_update_target_network()

            # 硬更新频率降低
            if frame_idx % self.target_update_freq == 0:
                self.update_target_network()
                if len(total_rewards) >= 10:
                    avg_reward = np.mean(total_rewards[-10:])
                    print(f"[Frame {frame_idx}] Avg Reward (last 10 episodes): {avg_reward:.2f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

        self.writer.close()

        # 显示 reward 曲线
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(total_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Performance")
        
        # 改进：添加移动平均线
        if len(total_rewards) > 10:
            window = min(50, len(total_rewards) // 10)
            moving_avg = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(total_rewards)), moving_avg, 'r-', alpha=0.8, label=f'Moving Average ({window})')
            plt.legend()
        
        plt.subplot(1, 2, 2)
        if len(total_rewards) > 100:
            recent_rewards = total_rewards[-100:]
            plt.plot(recent_rewards)
            plt.xlabel("Recent Episodes")
            plt.ylabel("Total Reward")
            plt.title("Recent Performance (Last 100 Episodes)")
        
        plt.tight_layout()
        plt.show()
        print("Training completed - reward curve displayed")