import torch
import torch.nn as nn



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

        self.q_head = nn.Linear(embed_dim, 4)

    def forward(self, maze):
        if not isinstance(maze, torch.Tensor):
            maze = torch.tensor(maze, dtype=torch.long)

        maze = maze.long()

        if maze.dim() == 2:
            maze = maze.unsqueeze(0)

        B, H, W = maze.shape
        N = H * W
        x = maze.view(B, N)
        x = self.embedding(x)

        positions = torch.arange(N, device=maze.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)

        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        maze_repr = x.mean(dim=1)

        q_values = self.q_head(maze_repr)

        if B == 1:
            q_values = q_values.squeeze(0)

        return q_values