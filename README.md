# Maze Solver with DQN

A Deep Q Network agent that learns to navigate mazes using a transformer-based architecture.

## Files

- **`model.py`**  `MazeTransformer`: embeds maze cells, applies self-attention, outputs Q-values for 4 actions (Up/Down/Left/Right)
- **`game.py`**  `MazeGame`: maze environment with step(), rewards, coins, and win condition
- **`training.py`**  `DQNTrainer`: experience replay, target network, reward shaping, BFS expert seeding
- **`training_mazes.py`** 10 pre built mazes (5×5, 7×7, 9×9) with walls, coins, and goals

## How to Run

```bash
pip install -r requirements.txt
python training.py
```

## Training Features

- Epsilon greedy exploration
- Replay buffer (capacity 10000)
- Target network (updated every 500 steps)
- Reward shaping (Manhattan distance to goal)
- BFS expert trajectory seeding
- Configurable warm-up phase
