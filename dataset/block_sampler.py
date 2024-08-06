import torch
from torch.utils.data import Sampler

class BlockSampler(Sampler):
    def __init__(self, data_source, block_size=2):
        self.data_source = data_source
        self.block_size = block_size

    def __iter__(self):
        indices = list(range(0, len(self.data_source), self.block_size))
        shuffled_blocks = torch.randperm(len(indices))

        for block_index in shuffled_blocks:
            block_start = indices[block_index]
            block_indices = list(range(block_start, block_start + self.block_size))
            yield from block_indices

    def __len__(self):
        return len(self.data_source)