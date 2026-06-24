import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from training_mazes import mazes
from game import MazeGame
from model import MazeTransformer


# FIX A: دالة مساعدة تبني maze يعكس موضع اللاعب الحالي
def get_current_maze(game, original_maze):
    """Returns maze with player at current position (not start)."""
    m = original_maze.copy()
    m[m == 3] = 0               # امسح موضع اللاعب الأصلي
    m[game.agent_pos] = 3       # ضعه في موضعه الحالي
    return m


class DQNTrainer:
    def __init__(self, agent, optimizer, mazes, gamma=0.9, epsilon=0.2, device="cpu"):
        self.agent = agent
        self.optimizer = optimizer
        self.mazes = mazes
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

    def choose_action(self, maze):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])

        with torch.no_grad():
            maze_tensor = torch.tensor(maze, dtype=torch.long).to(self.device)
            log_probs, _ = self.agent(maze_tensor)          # FIX B: unpack (log_probs, value)
            return torch.argmax(log_probs).item()

    def train(self, num_episodes=100, maze_type="random"):
        """Train the agent on mazes"""
        print("=" * 60)
        print(f"Training on {len(self.mazes)} mazes | Type: {maze_type}")
        print("=" * 60)

        for episode in range(num_episodes):
            original_maze = self._select_maze(maze_type)
            game = MazeGame(original_maze)
            game.reset()
            episode_reward = 0
            steps = 0

            while steps < 100:
                # FIX A: استخدم maze محدّث يعكس موضع اللاعب الحالي
                current_maze = get_current_maze(game, original_maze)

                action = self.choose_action(current_maze)
                state, reward, done = game.step(action)
                episode_reward += reward
                steps += 1

                # FIX A + B: احسب loss على الحالة الحالية
                maze_tensor = torch.tensor(current_maze, dtype=torch.long).to(self.device)
                log_probs, value = self.agent(maze_tensor)

                # FIX C: REINFORCE مع baseline
                # advantage = r - b(s) يقلل تباين الـ gradient
                advantage = reward - value.item()
                policy_loss = -log_probs[action] * advantage

                # FIX C: value loss - يعلّم الـ baseline يتوقع الـ reward
                value_loss = (value - reward) ** 2

                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if done:
                    break

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
            return self.mazes[0]
        elif maze_type == "7x7":
            return self.mazes[3]
        elif maze_type == "9x9":
            return self.mazes[6]
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
            # FIX A: هنا أيضاً نستخدم الحالة الحالية
            current_maze = get_current_maze(game, maze)
            action = self.choose_action(current_maze)
            state, reward, done = game.step(action)
            total_reward += reward
            steps += 1

            if steps % 10 == 0 or done:
                print(f"  Step {steps}: {action_names[action]} → {game.agent_pos} (reward: {reward})")

        if done:
            print(f"✓ SOLVED in {steps} steps | Points: {game.points}")
        else:
            print(f"✗ Not solved | Steps: {steps}, Reward: {total_reward}")

        return steps, total_reward


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = MazeTransformer(num_cell_types=5, embed_dim=32, num_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = DQNTrainer(model, optimizer, mazes, gamma=0.9, epsilon=0.2, device=device)

    trainer.train(num_episodes=100, maze_type="random")
    trainer.save_model("maze_model.pth")

    print("\n" + "=" * 60)
    print("Testing Results")
    print("=" * 60)
    trainer.test(mazes[0], max_steps=50)
    trainer.test(mazes[3], max_steps=100)
    trainer.test(mazes[6], max_steps=150)