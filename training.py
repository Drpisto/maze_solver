import random
import torch
import torch.nn as nn
import torch.optim as optim
from training_mazes import mazes
from game import MazeGame
from model import MazeTransformer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = MazeTransformer(num_cell_types=5, embed_dim=32, num_heads=4).to(device)
optimizer = optim.Adam(agent.parameters(), lr=0.001)

# =====================
# Hyperparameters
# =====================
gamma = 0.9
epsilon = 0.2

class DQNTrainer:
    def __init__(self, agent, optimizer, mazes, gamma=0.9, epsilon=0.2, device="cpu"):
        self.agent = agent
        self.optimizer = optimizer
        self.mazes = mazes
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

    def choose_action(self, maze):
        """Choose action using epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])  # Random action
        else:
            with torch.no_grad():
                maze_tensor = torch.tensor(maze, dtype=torch.long).to(self.device)
                action_probs = self.agent(maze_tensor)
                return torch.argmax(action_probs).item()  # Best action
    
    def train(self, num_episodes=100):
        print("=" * 60)
        print(f"Starting Training on {len(self.mazes)} mazes")
        print("=" * 60)
        
        for episode in range(num_episodes):
            maze = random.choice(self.mazes)
            game = MazeGame(maze)
            state = game.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < 100:
                action = self.choose_action(maze)
                next_state, reward, done = game.step(action)
                episode_reward += reward
                steps += 1

                # Convert maze to tensor
                maze_tensor = torch.tensor(maze, dtype=torch.long).to(self.device)
                
                # Get policy probabilities
                action_probs = self.agent(maze_tensor)
                
                # Policy gradient loss
                loss = -torch.log(action_probs[action] + 1e-8) * reward

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | Steps: {steps} | Points: {game.points}")
        
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
    
    def save_model(self, path="maze_model.pth"):
        """Save trained model"""
        torch.save(self.agent.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def test(self, maze, max_steps=100):
        """Test the model on a maze"""
        game = MazeGame(maze)
        state = game.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\nTesting on {maze.shape[0]}x{maze.shape[1]} maze...")
        print(f"Start: {game.agent_pos}, Goal: {game.goal_pos}")

        action_names = ["Up", "Down", "Left", "Right"]
        
        while not done and steps < max_steps:
            action = self.choose_action(maze)
            state, reward, done = game.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 10 == 0 or done:
                print(f"Step {steps}: {action_names[action]} → {game.agent_pos}, Reward: {reward}")

        if done:
            print(f"✓ SOLVED in {steps} steps! Total Points: {game.points}")
        else:
            print(f"✗ Not solved (steps: {steps}, reward: {total_reward})")
        
        return steps, total_reward


if __name__ == "__main__":
    print(f"Device: {device}\n")
    
    # Initialize trainer
    trainer = DQNTrainer(agent, optimizer, mazes, gamma=0.9, epsilon=0.2, device=device)
    
    # Train
    trainer.train(num_episodes=100)
    
    # Save model
    trainer.save_model("maze_model.pth")
    
    # Test on different mazes
    print("\n" + "=" * 60)
    print("Testing Results")
    print("=" * 60)
    
    # Test on easy (5x5)
    trainer.test(mazes[0], max_steps=50)
    
    # Test on medium (7x7)
    trainer.test(mazes[3], max_steps=100)
    
    # Test on hard (9x9)
    trainer.test(mazes[6], max_steps=150)
