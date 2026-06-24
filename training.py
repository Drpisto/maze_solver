import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from training_mazes import mazes
from game import MazeGame
from model import MazeTransformer


def get_current_maze(game, original_maze):
    m = original_maze.copy()
    m[m == 3] = 0
    m[game.agent_pos] = 3
    return m


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    def __init__(self, agent, optimizer, mazes, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, device="cpu", batch_size=64, target_update_freq=100,
                 buffer_capacity=10000):
        self.agent = agent
        self.target_agent = type(agent)(num_cell_types=5, embed_dim=32, num_heads=4).to(device)
        self.target_agent.load_state_dict(agent.state_dict())
        self.target_agent.eval()
        self.optimizer = optimizer
        self.mazes = mazes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.buffer = ReplayBuffer(buffer_capacity)
        self.steps_done = 0

    def choose_action(self, maze):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        with torch.no_grad():
            maze_tensor = torch.tensor(maze, dtype=torch.long).to(self.device)
            q_values = self.agent(maze_tensor)
            return torch.argmax(q_values).item()

    def shaped_reward(self, reward, prev_pos, game):
        prev_dist = abs(prev_pos[0] - game.goal_pos[0]) + abs(prev_pos[1] - game.goal_pos[1])
        curr_dist = abs(game.agent_pos[0] - game.goal_pos[0]) + abs(game.agent_pos[1] - game.goal_pos[1])
        shaping = prev_dist - curr_dist
        return reward + 0.5 * shaping

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.long).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.long).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.agent(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_agent(next_states)
            next_state_values = next_q_values.max(dim=1)[0]
            target_values = rewards + self.gamma * next_state_values * (1 - dones)

        loss = nn.MSELoss()(state_action_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())

    def train(self, num_episodes=100, maze_type="random", max_steps=100):
        print("=" * 60)
        print(f"Training on {len(self.mazes)} mazes | Type: {maze_type}")
        print("=" * 60)

        best_reward = float("-inf")

        for episode in range(num_episodes):
            original_maze = self._select_maze(maze_type)
            game = MazeGame(original_maze)
            game.reset()
            episode_reward = 0
            steps = 0

            state = get_current_maze(game, original_maze)

            while steps < max_steps:
                action = self.choose_action(state)
                prev_pos = list(game.agent_pos)
                _, reward, done = game.step(action)
                reward = self.shaped_reward(reward, prev_pos, game)
                next_state = get_current_maze(game, original_maze)
                episode_reward += reward
                steps += 1

                self.buffer.push(state, action, reward, next_state, done)
                self.train_step()

                state = next_state

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode_reward > best_reward:
                best_reward = episode_reward

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | Steps: {steps} | Points: {game.points}"
                      f" | Eps: {self.epsilon:.3f} | Buffer: {len(self.buffer)}")

        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)

    def _select_maze(self, maze_type):
        if maze_type == "random":
            return random.choice(self.mazes)
        elif maze_type == "5x5":
            return random.choice(self.mazes[0])
        elif maze_type == "7x7":
            return random.choice(self.mazes[1])
        elif maze_type == "9x9":
            return random.choice(self.mazes[2])
        return random.choice(self.mazes)

    def save_model(self, path="maze_model.pth"):
        torch.save(self.agent.state_dict(), path)
        print(f"Model saved to {path}")

    def test(self, maze, max_steps=100):
        game = MazeGame(maze)
        game.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\nTesting {maze.shape[0]}x{maze.shape[1]} maze...")
        print(f"Start: {game.agent_pos} -> Goal: {game.goal_pos}")

        action_names = ["Up", "Down", "Left", "Right"]

        while not done and steps < max_steps:
            current_maze = get_current_maze(game, maze)
            action = self.choose_action(current_maze)
            _, reward, done = game.step(action)
            total_reward += reward
            steps += 1

            if steps % 10 == 0 or done:
                print(f"  Step {steps}: {action_names[action]} -> {game.agent_pos} (reward: {reward})")

        if done:
            print(f"  SOLVED in {steps} steps | Points: {game.points}")
        else:
            print(f"  Not solved | Steps: {steps}, Reward: {total_reward}")

        return steps, total_reward


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = MazeTransformer(num_cell_types=5, embed_dim=32, num_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = DQNTrainer(model, optimizer, mazes, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                         epsilon_decay=0.995, device=device, batch_size=64,
                         target_update_freq=100, buffer_capacity=10000)

    trainer.train(num_episodes=200, maze_type="5x5", max_steps=50)
    trainer.save_model("maze_model.pth")

    print("\n" + "=" * 60)
    print("Testing Results")
    print("=" * 60)

    print("Testing on 5x5 mazes:")
    trainer.test(random.choice(mazes[0]), max_steps=50)
    print("\nTesting on 7x7 mazes:")
    trainer.test(random.choice(mazes[1]), max_steps=50)
    print("\nTesting on 9x9 mazes:")
    trainer.test(random.choice(mazes[2]), max_steps=75)
