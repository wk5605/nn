import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import cv2
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.grasp import GraspRobot
from modules.ddpg import ReplayBuffer, Transition

IMAGE_SIZE = 32
ACTION_SIZE = IMAGE_SIZE * IMAGE_SIZE
BATCH_SIZE = 64
MEM_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 800
GAMMA = 0.99
LR = 0.001
TARGET_UPDATE = 100

class VisualFeatureEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128 * IMAGE_SIZE * IMAGE_SIZE, ACTION_SIZE)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQN_Trainer:
    def __init__(self, render_mode="human"):
        self.env = GraspRobot(render_mode=render_mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(MEM_SIZE, simple=False)
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.steps_done = 0
        self.save_dir = os.path.join("grasprl", "dataset", "grasp_samples")
        os.makedirs(self.save_dir, exist_ok=True)
        self.enhancer = VisualFeatureEnhancer().to(self.device).eval()

    def transform_state(self, obs):
        depth = obs["depth"].astype(np.float32)
        depth = depth.max() - depth
        if depth.max() > 0:
            depth = depth / depth.max()
        rgb = obs["rgb"].astype(np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(self.device)
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device)
        state = torch.cat([rgb_t, depth_t], dim=1).float()
        with torch.no_grad():
            state = self.enhancer(state)
        return state

    def transform_action(self, action_idx, depth_img):
        idx = action_idx.item()
        px = idx % IMAGE_SIZE
        py = idx // IMAGE_SIZE
        px = max(0, min(px, IMAGE_SIZE-1))
        py = max(0, min(py, IMAGE_SIZE-1))
        depth_val = depth_img[py][px] if depth_img[py][px] > 0 else np.mean(depth_img)
        action = self.env.pixel2world(1, px, py, depth_val)
        action = np.clip(action, [-0.25, -0.25, 1.05], [0.25, 0.25, 1.3])
        return action.tolist()

    def select_action(self, state):
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                q_vals = self.policy_net(state)
                action = q_vals.argmax().item()
                return torch.tensor([[action]], device=self.device)
        else:
            return torch.tensor([[random.randint(0, ACTION_SIZE-1)]], device=self.device)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        s = torch.cat(batch.state).to(self.device)
        a = torch.cat(batch.action).to(self.device)
        r = torch.cat(batch.reward).to(self.device)
        ns = torch.cat(batch.next_state).to(self.device)
        q_values = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1, keepdim=True)[0]
            target = r + GAMMA * next_q
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_sample(self, action, reward, info, i):
        rgb = self.env.observation["rgb"]
        cv2.imwrite(f"{self.save_dir}/rgb_{i}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        label = {"grasp_success": 1 if info["grasp"]=="Success" else 0, "reward": reward}
        np.save(f"{self.save_dir}/label_{i}.npy", label)

def main():
    max_episodes = 200
    trainer = DQN_Trainer(render_mode="human")
    success_count = 0
    loop = tqdm(range(1, max_episodes+1))
    for ep in loop:
        obs = trainer.env.reset_without_random()
        state = trainer.transform_state(obs)
        done = False
        step = 0
        ep_reward = 0
        while not done and step < 10:
            action_idx = trainer.select_action(state)
            action = trainer.transform_action(action_idx, obs["depth"])
            next_obs, reward, done, info = trainer.env.step(action)
            next_state = trainer.transform_state(next_obs)
            trainer.memory.push(state, action_idx, torch.tensor([[reward]], device=trainer.device), next_state)
            state = next_state
            obs = next_obs
            ep_reward += reward
            step += 1
            trainer.learn()
        if info.get("grasp") == "Success":
            success_count += 1
        loop.set_postfix(success_rate=f"{success_count/ep:.2f}", ep_reward=f"{ep_reward:.2f}")
        trainer.save_sample(action, reward, info, ep)
    os.makedirs("grasprl/trained", exist_ok=True)
    torch.save(trainer.policy_net.state_dict(), "grasprl/trained/dqn_final.pth")
    trainer.env.close()

if __name__ == "__main__":
    main()