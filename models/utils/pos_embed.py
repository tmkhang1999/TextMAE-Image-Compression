"""2D sin-cos position embeddings, from facebookresearch/mae (Meta Platforms, CC BY-NC 4.0)."""
import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Position embedding (grid_size**2 [+ 1 for the cls token], embed_dim)."""
    assert embed_dim % 2 == 0
    grid = np.stack(np.meshgrid(np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32)))
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = np.concatenate([get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]),
                                get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])], axis=1)
    return np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0) if cls_token else pos_embed
