import random
import numpy as np


maze = np.array([
    [1, 2, 0, 1, 3],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 4],
])

num_cell_types = {
    0: "empty",  
    1: "wall",  
    2: "coin",  
    3: "player",    
    4: "end"    
}

class MazeGame: 
    # Editable reward configuration
    REWARDS = {
        "step": -1,           # Cost per step
        "wall_hit": -20,      # Hit wall penalty
        "out_of_bounds": -20, # Out of bounds penalty
        "win": 10,            # Win reward
        "coin": 5,            # Coin collection reward
        "win_points": 10      # Bonus points for winning
    }
    
    def __init__(self, maze ):
        self.points = 0
        self.maze = maze
        self.agent_pos = self.find_position("player")
        self.goal_pos = self.find_position("end")

    def find_position(self, cell_type):
        # Create reverse mapping of cell types
        cell_value_map = {v: k for k, v in num_cell_types.items()}
        target_value = cell_value_map.get(cell_type)
        
        if target_value is None:
            return None
            
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] == target_value:
                    return (i, j)
        return None

    def reset(self):
        self.points = 0
        self.agent_pos = self.find_position("player")
        return self.get_state()

    def get_state(self):
        state = np.copy(self.maze).astype(object)
        state[self.agent_pos] = "a"
        return state

    def reward_movement(self):
        """Reward for a normal step"""
        return self.REWARDS["step"]
    
    def reward_coin(self):
        """Reward and points for collecting a coin"""
        self.points += self.REWARDS["coin"]
        return 0  # No additional reward, just points
    
    def reward_wall_hit(self):
        """Penalty for hitting a wall"""
        return self.REWARDS["wall_hit"]
    
    def reward_out_of_bounds(self):
        """Penalty for moving out of bounds"""
        return self.REWARDS["out_of_bounds"]
    
    def reward_win(self):
        """Reward for winning and bonus points"""
        self.points += self.REWARDS["win_points"]
        return self.REWARDS["win"]

    def step(self, action):
        move_map = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        move = move_map[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        reward = self.reward_movement()
        done = False
        
        if (0 <= new_pos[0] < self.maze.shape[0] and 
            0 <= new_pos[1] < self.maze.shape[1]):
            if self.maze[new_pos] != 1:  # 1 is wall
                # Check for coin collection
                if self.maze[new_pos] == 2:  # 2 is coin
                    reward += self.reward_coin()
                self.agent_pos = new_pos
            else:
                # Hit a wall - penalty and no movement
                reward = self.reward_wall_hit()
        else:
            # Out of bounds
            reward = self.reward_out_of_bounds()
            done = True  # End the game if out of bounds
        

        if self.check_win():
            reward = self.reward_win()
            done = True

        return self.get_state(), reward, done
    
    def check_win(self):
        
        return self.agent_pos == self.goal_pos
    
    
    


