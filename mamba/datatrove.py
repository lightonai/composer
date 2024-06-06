import numpy as np
import torch
from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from typing import NamedTuple
from itertools import count
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import List, Dict 


class MambaBatch(NamedTuple):
    # the input_ids has torch.Tensor type 
    input_ids: torch.Tensor  # shape: seq_len batch_size
    target_ids: torch.Tensor  # shape: seq_len batch_size
    loss_factors: torch.Tensor | torch.FloatTensor  | None  # shape: seq_len batch_size
    loss_norm: torch.Tensor | None  


def collate_mamba_batch(
    data: List[Dict[str, torch.Tensor]]
) -> MambaBatch:

    # Extract input_ids from each sample in the list and stack them into a single tensor
    tokens = torch.stack([sample['input_ids'] for sample in data])
    input_ids = tokens[:, :-1]
    target_ids = tokens[:, 1:]
    loss_factors = torch.ones_like(input_ids)
    loss_norm=loss_factors.sum()

    return MambaBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_factors=loss_factors,
        loss_norm=loss_norm,
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
    n_data_parallel: int,
    rank: int,
    n_samples_to_skip: int = 0,
    num_workers: int = 0,
    prefetch_factor: int | None = 2,
    max_tokens: int = 100000,
    token_size: int = 2,
):

    collate_fct = collate_mamba_batch
    dataset = DatatroveFileDataset(path, seq_len, token_size, max_tokens)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fct,
        batch_size=batch_size,
        sampler=DistributedLinearSampler(
            num_replicas=n_data_parallel,
            rank=rank,
            dataset=dataset,
            restart_sample_idx=n_samples_to_skip,
        ),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
    )
    return {
        "dataloader": dataloader,
        "get_num_samples_in_batch": get_num_samples_in_batch,
        "get_num_tokens_in_batch": get_num_tokens_in_batch,
        "split_batch": split_batch,
    }


class DistributedLinearSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = None,
        rank: int = None,
        restart_sample_idx: int = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.global_offset = 0

        self.restart_sample_idx = restart_sample_idx

    def __iter__(self):
        return iter(
            count(
                self.rank + self.restart_sample_idx + self.global_offset,
                self.num_replicas,
            )
        )

    def __len__(self):
        return float("inf")

    def set_global_offset(self, global_offset: int):
        self.global_offset = global_offset
        
        
def get_num_samples_in_batch(batch: MambaBatch):
    return batch.input_ids.shape[0]


def get_num_tokens_in_batch(batch: MambaBatch):
    return batch.input_ids.shape[0] * (batch.input_ids.shape[1] + 1)


def split_batch(batch: MambaBatch, microbatch_size):

    per_device_batch_size = get_num_samples_in_batch(batch)
    assert (
        per_device_batch_size % microbatch_size == 0
    ), f"per_device_batch_size {per_device_batch_size} must be divisble by microbatch size {microbatch_size}"
    num_chunks = per_device_batch_size // microbatch_size
    if num_chunks <= 1:
        return [batch]
    chunks = []
    for chunk_id in range(num_chunks):
        chunks.append(
            MambaBatch(
                input_ids=batch.input_ids[
                    chunk_id * microbatch_size : (chunk_id + 1) * microbatch_size, :
                ],
                target_ids=batch.target_ids[
                    chunk_id * microbatch_size : (chunk_id + 1) * microbatch_size, :
                ],
                loss_factors=batch.loss_factors[
                    chunk_id * microbatch_size : (chunk_id + 1) * microbatch_size, :
                ],
                loss_norm=batch.loss_factors[
                    chunk_id * microbatch_size : (chunk_id + 1) * microbatch_size, :
                ].sum(),
            )
        )
    return chunks