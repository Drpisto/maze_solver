import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from training_mazes import mazes
from game import MazeGame
from model import MazeTransformer


def get_current_maze(game, original_maze):
    """Returns maze with player at current position (not start)."""
    m = original_maze.copy()
    m[m == 3] = 0               
    m[game.agent_pos] = 3       
    return m


class DQNTrainer:
    def __init__(self, agent, optimizer, mazes, gamma=0.9, epsilon=0.2, device="cpu", epsilon_start=0.9, epsilon_end=0.05):
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon       = self.epsilon_start
        self.agent = agent
        self.optimizer = optimizer
        self.mazes = mazes
        self.gamma = gamma
        
        self.device = device
        
        

    def choose_action(self, maze,episode,num_episodes ):


        progress = episode / num_episodes
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress)
        """Choose action using epsilon-greedy policy"""
        
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])

        with torch.no_grad():
            maze_tensor = torch.tensor(maze, dtype=torch.long).to(self.device)
            log_probs, _ = self.agent(maze_tensor)
            return torch.argmax(log_probs).item()

    

    def shaped_reward(self, reward, prev_pos, game):
        """Apply reward shaping based on Manhattan distance to goal"""
        
        prev_dist = abs(prev_pos[0] - game.goal_pos[0]) + abs(prev_pos[1] - game.goal_pos[1])
        curr_dist = abs(game.agent_pos[0] - game.goal_pos[0]) + abs(game.agent_pos[1] - game.goal_pos[1])
        shaping = prev_dist - curr_dist  
        return reward + 0.5 * shaping

    def monte_carlo_returns(self, rewards):
        """Calculate Monte Carlo returns from episode rewards using discount factor gamma"""
        returns = []
        cumulative_return = 0
        for reward in reversed(rewards):
            cumulative_return = reward + self.gamma * cumulative_return
            returns.insert(0, cumulative_return)
        return returns


    def train(self, num_episodes=100, maze_type="random", max_steps=100):
        """Train the agent on mazes using Monte Carlo returns"""
        print("=" * 60)
        print(f"Training on {len(self.mazes)} mazes | Type: {maze_type}")
        print("Monte Carlo Returns enabled")
        print("=" * 60)

        for episode in range(num_episodes):
            original_maze = self._select_maze(maze_type)
            game = MazeGame(original_maze)
            game.reset()
            
            episode_transitions = []
            episode_rewards = []
            steps = 0

            while steps < max_steps:
                current_maze = get_current_maze(game, original_maze)
                action = self.choose_action(current_maze, episode, num_episodes)
                prev_pos = list(game.agent_pos)
                state, reward, done = game.step(action)
                
                reward = self.shaped_reward(reward, prev_pos, game)
                episode_rewards.append(reward)
                
                maze_tensor = torch.tensor(current_maze, dtype=torch.long).to(self.device)
                
                episode_transitions.append({
                    'maze_tensor': maze_tensor,
                    'action': action
                })
                
                steps += 1
                if done:
                    break

            returns = self.monte_carlo_returns(episode_rewards)
            
            for i, (transition, ret) in enumerate(zip(episode_transitions, returns)):
                log_probs, value = self.agent(transition['maze_tensor'])
                
                advantage = ret - value.item()
                policy_loss = -log_probs[transition['action']] * advantage
                value_loss = (value - ret) ** 2
                
                loss = policy_loss + 0.5 * value_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            episode_reward = sum(episode_rewards)
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | Steps: {steps} | Points: {game.points}")

        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)

    def _select_maze(self, maze_type):
        """Select maze based on type"""
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
        """Save trained model"""
        torch.save(self.agent.state_dict(), path)
        print(f"✓ Model saved to {path}")

    def test(self, maze, max_steps=100):
        """Test the model on a maze"""
        game = MazeGame(maze)
        game.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\nTesting {maze.shape[0]}x{maze.shape[1]} maze...")
        print(f"Start: {game.agent_pos} → Goal: {game.goal_pos}")

        action_names = ["Up", "Down", "Left", "Right"]

        while not done and steps < max_steps:
            current_maze = get_current_maze(game, maze)
            action = self.choose_action(current_maze, 1, 1)  # Epsilon is irrelevant during testing
            prev_pos = list(game.agent_pos)
            state, reward, done = game.step(action)
            reward = self.shaped_reward(reward, prev_pos, game)
            total_reward += reward
            steps += 1

            if steps % 10 == 0 or done:
                print(f"  Step {steps}: {action_names[action]} → {game.agent_pos} (reward: {reward})")

        if done:
            print(f" SOLVED in {steps} steps | Points: {game.points}")
        else:
            print(f" Not solved | Steps: {steps}, Reward: {total_reward}")

        return steps, total_reward


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = MazeTransformer(num_cell_types=5, embed_dim=32, num_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = DQNTrainer(model, optimizer, mazes, gamma=0.9, epsilon=0.2, device=device)

    trainer.train(num_episodes=100, maze_type="5x5", max_steps=50)
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
    