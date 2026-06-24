import torch
import torch.nn as nn
import torch.nn.functional as F


class MazeTransformer(nn.Module):
    def __init__(self,
                 num_cell_types=5,
                 embed_dim=32,
                 num_heads=4,
                 max_cells=81):      
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_cell_types,
            embedding_dim=embed_dim
        )

        
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_cells,
            embedding_dim=embed_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.policy_head = nn.Linear(embed_dim, 4)

        
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, maze):
        if not isinstance(maze, torch.Tensor):
            maze = torch.tensor(maze)

        maze = maze.long()

        x = maze.flatten()                      # [N]

        x = self.embedding(x)                   # [N, D]

        
        positions = torch.arange(x.shape[0], device=maze.device)
        x = x + self.pos_embedding(positions)   # [N, D]

        x = x.unsqueeze(0)                      # [1, N, D]

        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        maze_repr = x.mean(dim=1)               # [1, D]

        
        log_probs = F.log_softmax(self.policy_head(maze_repr), dim=-1).squeeze(0)  # [4]

        
        value = self.value_head(maze_repr).squeeze()                               # scalar

        return log_probs, value