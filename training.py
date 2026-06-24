import random
import torch
import torch.nn as nn
import torch.optim as optim
from training_mazes import mazes
from game import MazeGame
from model import DQNAgent
import numpy as np

agent = DQNAgent() 

optimizer = optim.Adam(agent.parameters(), lr=0.001)


# =====================
# Hyperparameters
# =====================
gamma = 0.9
epsilon = 0.2

class DQNTrainer:
    def __init__(self, agent, optimizer, mazes, gamma=0.9, epsilon=0.2):
        self.agent = agent
        self.optimizer = optimizer
        self.mazes = mazes
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])  # Random action
        else:
            with torch.no_grad():
                q_values = self.agent(state)
                return torch.argmax(q_values).item()  # Best action
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            maze = random.choice(self.mazes)
            game = MazeGame(maze)
            state = game.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = game.step(action)

                # Convert states to tensors
                state_tensor = torch.tensor(state.flatten(), dtype=torch.long).unsqueeze(0)
                next_state_tensor = torch.tensor(next_state.flatten(), dtype=torch.long).unsqueeze(0)

                # Compute target Q-value
                with torch.no_grad():
                    target_q_value = reward + (self.gamma * torch.max(self.agent(next_state_tensor)) * (1 - int(done)))

                # Compute current Q-value
                current_q_value = self.agent(state_tensor)[0, action]

                # Compute loss
                loss = nn.MSELoss()(current_q_value, target_q_value)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state

            if episode % 100 == 0:
                print(f"Episode {episode}, Loss: {loss.item()}")