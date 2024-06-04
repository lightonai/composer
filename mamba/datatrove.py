import numpy as np
import torch
from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from typing import NamedTuple
from functools import partial

from torch.utils.data import Dataset, DataLoader
    

class MambaBatch(NamedTuple):
    # the input_ids has torch.Tensor type 
    input_ids: torch.Tensor  # shape: seq_len batch_size
    target_ids: torch.Tensor  # shape: seq_len batch_size
    loss_factors: torch.Tensor | torch.FloatTensor  # shape: seq_len batch_size
    loss_norm: torch.FloatTensor  


def collate_mamba_batch(
    data: list
) -> MambaBatch:
    # data = torch.stack(list(data))
    # tokens =  data['input_ids'] 
    # input_ids = tokens[:, :]
    
    # Extract input_ids from each sample in the list and stack them into a single tensor

    tokens = torch.stack([sample['input_ids'] for sample in data])
    input_ids = tokens[:, :-1]
    target_ids = tokens[:, 1:]
    loss_factors = None

    return MambaBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_factors=loss_factors,
    )


class DatatroveFileDataset(Dataset):
        """Dataset for a single .ds file created by datatrove
        We loop on the dataset if asking for an index larger than the dataset size

        Args:
            file_path (str): path to file on s3, locally, or some other fsspec supported path
            seq_len (int): sequence length
            token_size (int): size of a single token, in bytes. Usually 2 for vocab sizes < 65k and 4 for larger
            max_tokens (int): only read at most this number of tokens
        """

        def __init__(
            self,
            file_path: str,
            seq_len: int,
            token_size: int = 2,
            max_tokens: int | None = None,
        ):
            self.file_path: str = file_path
            self.seq_len = seq_len
            self.token_size = token_size

            self.fs: AbstractFileSystem
            self.fs, self.file_path = url_to_fs(file_path)
            fsize = self.fs.size(self.file_path)
            # total number of full contexts in this file
            num_tokens = fsize // self.token_size
            self._len = (min(max_tokens, num_tokens) if max_tokens else num_tokens) // (seq_len + 1)
            self._f = None

        def __getitem__(self, item):
            # We loop on the dataset if asking for an index larger than the dataset size
            epoch_item = item % len(self)
            if not self._f:
                self._f = self.fs.open(self.file_path, "rb")
            chunk_size = self.token_size * (self.seq_len + 1)
            self._f.seek(epoch_item * chunk_size)
            return {
                "input_ids": torch.as_tensor(
                    np.frombuffer(self._f.read(chunk_size), np.uint16 if self.token_size == 2 else np.uint32).astype(
                        np.int64
                    ),
                    dtype=torch.long,
                )
            }

        def __len__(self):
            return self._len
        
def get_mamba_dataloader(
    path: str,
    batch_size: int,
    seq_len: int,
    max_tokens: int,
    token_size: int = 2,
    # num_workers: int = 0,
    # prefetch_factor: int = 2,
):

    collate_fct = partial(collate_mamba_batch)
    dataset = DatatroveFileDataset(path, seq_len, token_size, max_tokens)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fct,
        batch_size=batch_size,
        # num_workers=num_workers,
        # prefetch_factor=prefetch_factor,
        shuffle=False,
    )
    return {
        "dataloader": dataloader,
    }

