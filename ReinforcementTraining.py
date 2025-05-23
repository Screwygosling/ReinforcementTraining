import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple, deque
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class Track:
    def __init__(self, Track_size=100, Track_width=10):
        self.Track_size = Track_size
        self.Track_width = Track_width
        self.position = 0
        self.lane_position = Track_width // 2
        self.speed = 0
        self.max_speed = 5
        self.obstacles = self._generate_obstacles()
        self.done = False
        self.steps = 0
        self.max_steps = 500

    def _generate_obstacles(self):
        obstacles = []
        num_obstacles = self.Track_size // 10
        for _ in range(num_obstacles):
            pos = random.randint(0, self.Track_size - 1)
            lane = random.randint(0, self.Track_width - 1)
            obstacles.append((pos, lane))
        return obstacles

    def reset(self):
        self.position = 0
        self.lane_position = self.Track_width // 2
        self.speed = 0
        self.steps = 0
        self.done = False
        self.obstacles = self._generate_obstacles()
        return self._get_state()

    def _get_state(self):
        nearby_obstacles = []
        lookAhead = 20

        for i in range(lookAhead):
            pos = (self.position + i) % self.Track_size
            obstacle_present = [0] * self.Track_width

            for obs_pos, obs_lane in self.obstacles:
                if obs_pos == pos:
                    obstacle_present[obs_lane] = 1
            nearby_obstacles.extend(obstacle_present)

        norm_position = self.position / self.Track_size
        norm_lane_position = self.lane_position / (self.Track_width - 1)
        norm_speed = self.speed / self.max_speed

        return np.array([norm_position, norm_lane_position, norm_speed] + nearby_obstacles)

    def step(self, action):
        self.steps += 1

        if action == 0:
            self.speed = min(self.speed + 1, self.max_speed)
        elif action == 1:
            self.speed = max(self.speed - 1, 0)
        elif action == 2:
            self.lane_position = max(self.lane_position - 1, 0)
        elif action == 3:
            self.lane_position = min(self.lane_position + 1, self.Track_width - 1)

        self.position = (self.position + self.speed) % self.Track_size  # makes the car loop

        collision = False
        for obs_pos, obs_lane in self.obstacles:
            if obs_pos == self.position and obs_lane == self.lane_position:
                collision = True
                break

        reward = 0
        if collision:
            reward = -10
            self.done = True  # Fixed typo: self.Done -> self.done
        elif self.speed == 0:
            reward = -1
        else:
            reward = self.speed
            if self.position < self.speed and self.speed > 0:
                reward += 20

        if self.steps >= self.max_steps:
            self.done = True
        return self._get_state(), reward, self.done, {}

    def render(self):
        track = [['.' for _ in range(self.Track_width)] for _ in range(10)]

        for i in range(10):
            pos = (self.position - 5 + i) % self.Track_size
            for obs_pos, obs_lane in self.obstacles:
                if obs_pos == pos:
                    track[i][obs_lane] = 'X'

        car_row = 5
        track[car_row][self.lane_position] = 'C'

        print(f"Position: {self.position}, Lane: {self.lane_position}, Speed: {self.speed}")
        for row in track:
            print(''.join(row))
        print("-" * 30)

class DQN(nn.Module):  # Corrected indentation
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128) # change 128 to 512
        self.fc2 = nn.Linear(128, 128) # change 128 to 512
        self.fc3 = nn.Linear(128, action_size) # change 128 to 512

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:  # Corrected indentation
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def train_car_agent():  # Corrected indentation
    env = Track(Track_size=200, Track_width=5)
    state_size = len(env.reset())
    action_size = 4  # Corrected to 4 actions (0-3)

    learning_rate = 0.01
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    buffer_size = 10000
    batch_size = 64 # change 64 to 128
    target_update = 10

    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    replay_buffer = ReplayBuffer(buffer_size)

    num_episodes = 1000
    max_steps = 500
    epsilon = epsilon_start

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            replay_buffer.push(state, action, next_state, reward, done)

            state = next_state

            if len(replay_buffer) > batch_size:
                transitions = replay_buffer.sample(batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)),
                                            device=device, dtype=torch.bool)
                state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
                reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)

                q_values = policy_net(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(batch_size, 1, device=device)
                if any(non_final_mask):
                    non_final_next_states = torch.FloatTensor(np.array([s for s, d in zip(batch.next_state, batch.done) if not d])).to(device)
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].unsqueeze(1).detach()

                expected_q_values = reward_batch + (gamma * next_state_values)
                loss = criterion(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
                optimizer.step()



            if done:
                break
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Epsilon: {epsilon:.2f}")
                
        if device == "cuda":
            print(f"GPU Memory Allocated: {cuda.memory_allocated(device)/1e6:.2f} MB")
            print(f"GPU Memory Cached: {cuda.memory_reserved(device)/1e6:.2f} MB")
            print(f"GPU Utilization: {cuda.utilization(device)}%")


    torch.save(policy_net.state_dict(), "car_agent.pth")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1,2,2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.savefig('Learning_curve.png')
    plt.show()

    return policy_net, env

def test_agent(model, env, num_episodes=5, render=True):  # Corrected indentation
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < env.max_steps:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            step += 1

            if render:
                env.render()
                time.sleep(0.1)

        print(f"Test Episode {episode+1}, Total Reward: {total_reward}, Steps: {step}")
        total_rewards.append(total_reward)

    print(f"Average Test Reward: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    trained_model, env = train_car_agent()
    test_agent(trained_model, env)