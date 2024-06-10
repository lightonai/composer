from bisect import bisect

import numpy as np
import torch
from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

from itertools import count
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import List, Dict, NamedTuple


class MambaBatch(NamedTuple):
    # the input_ids has torch.Tensor type
    input_ids: torch.Tensor  # shape: seq_len batch_size
    target_ids: torch.Tensor  # shape: seq_len batch_size


def collate_mamba_batch(data: List[Dict[str, torch.Tensor]]) -> MambaBatch:
    # Extract input_ids from each sample in the list and stack them into a single tensor
    tokens = torch.stack([sample["input_ids"] for sample in data])
    input_ids = tokens[:, :-1]
    target_ids = tokens[:, 1:]

    return MambaBatch(
        input_ids=input_ids,
        target_ids=target_ids,
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
        self._len = (min(max_tokens, num_tokens) if max_tokens else num_tokens) // (
            seq_len + 1
        )
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
                np.frombuffer(
                    self._f.read(chunk_size),
                    np.uint16 if self.token_size == 2 else np.uint32,
                ).astype(np.int64),
                dtype=torch.long,
            )
        }

    def __len__(self):
        return self._len

class DatatroveFolderDataset(Dataset):
    """
    Dataset for a folder of .ds files
    We loop on the dataset if asking for an index larger than the dataset size

    Args:
        folder_path (str): path to folder on S3, locally, or some other fsspec supported path
        seq_len (int): sequence length
        filename_pattern (Union[Pattern, str], optional): filename pattern. Defaults to None.
        recursive (bool, optional): search recursively. Defaults to True.
        token_size (int): size of a single token, in bytes. Usually 2 for vocab sizes < 65k and 4 for larger
        max_tokens (int): only read at most this number of tokens
        shuffle (bool, optional): shuffle the files in the folder. Defaults to False.
        seed (int, optional): seed for shuffling. Defaults to 42.
    """

    def __init__(
        self,
        folder_path: str,
        seq_len: int,
        filename_pattern: str = None,
        recursive: bool = True,
        token_size: int = 2,
        max_tokens: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.folder_path = folder_path
        self.filename_pattern = filename_pattern
        fs, folder_path = url_to_fs(folder_path)
        matched_files = (
            fs.find(folder_path, detail=False, maxdepth=1 if not recursive else None)
            if not filename_pattern
            else fs.glob(filename_pattern, maxdepth=1 if not recursive else None)
        )
        if not matched_files:
            raise FileNotFoundError(f'No files matching "{filename_pattern}" found in {folder_path}')

        self.files = []
        remaining_tokens = max_tokens
        for path in matched_files:
            file_data = DatatroveFileDataset(
                fs.unstrip_protocol(path),
                seq_len,
                token_size=token_size,
                max_tokens=remaining_tokens,
            )
            self.files.append(file_data)
            if remaining_tokens is not None:
                remaining_tokens -= len(file_data) * (seq_len + 1)
                if remaining_tokens <= 0:
                    break

        if shuffle:
            rand = np.random.default_rng(seed)
            ordering = rand.permutation(range(len(self.files)))
            self.files = [self.files[i] for i in ordering]

        self.lens = np.cumsum([0] + [len(f) for f in self.files]).tolist()

        self.current_file = 0

    def __getitem__(self, item):
        # check if we are in the same file as before
        if not (self.lens[self.current_file] <= item < self.lens[self.current_file + 1]):
            # figure out current file
            self.current_file = bisect(self.lens, item) - 1
        # subtract file starting offset
        return self.files[self.current_file][item - self.lens[self.current_file]]

    def __len__(self):
        return self.lens[-1] if self.lens else 0

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
    
    if path.endswith('.ds'):
        dataset = DatatroveFileDataset(path, seq_len, token_size, max_tokens)
    else:
        dataset = DatatroveFolderDataset(path, seq_len, recursive=False, token_size=token_size, max_tokens=max_tokens)
        
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
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(
                    rank, num_replicas - 1
                )
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
            )
        )
    return chunks
