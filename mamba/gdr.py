import glob
import warnings
from enum import Enum
from functools import partial
from itertools import count
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class MambaBatch(NamedTuple):
    input_ids: torch.LongTensor  # shape: seq_len batch_size
    target_ids: torch.LongTensor  # shape: seq_len batch_size
    loss_factors: torch.BoolTensor | torch.FloatTensor  # shape: seq_len batch_size
    loss_norm: torch.FloatTensor


class DataLayout(Enum):
    token_bits = (0, 16)  # max number of tokens: 64k
    position_bits = (16, 30)  # max context length of 16k
    loss_target = (30, 31)  # just one bit.


def get_bits(data: torch.Tensor, layout: DataLayout):
    assert data.dtype == torch.long
    low, high = layout.value
    high_mask = (1 << high) - 1
    masked = torch.bitwise_and(data, high_mask)
    shifted = torch.bitwise_right_shift(masked, low)
    return shifted


def collate_mamba_batch(
    data: list[torch.LongTensor], position_weighting: bool
) -> MambaBatch:
    data = torch.stack(list(data))
    tokens = get_bits(data, DataLayout.token_bits)
    input_ids = tokens[:, :-1]
    target_ids = tokens[:, 1:]

    positions = get_bits(data, DataLayout.position_bits)

    # make each subsequence start with position id 0
    # non-zero starts can occur with tree structures and for the first subsequence of a sample that has been cut off
    for sample_id in range(positions.shape[0]):
        end_ids = (
            1 + torch.where(positions[sample_id, 1:] < positions[sample_id, :-1])[0]
        ).tolist() + [positions.shape[1]]
        for i, end_id in enumerate(end_ids):
            if i == 0:
                start_id = 0
            else:
                start_id = end_ids[i - 1]
            positions[sample_id, start_id:end_id] = (
                positions[sample_id, start_id:end_id] - positions[sample_id, start_id]
            )

    assert positions.min() >= 0, "Can't have negative positions"

    if position_weighting:
        loss_factors = torch.tanh(positions[:, 1:] / 10)
    else:
        loss_factors = get_bits(data, DataLayout.loss_target).to(torch.bool)[:, 1:]
        # never have loss on first token
        loss_factors[positions[:, 1:] == 0] = 0

    return MambaBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_factors=loss_factors,
        loss_norm=loss_factors.sum(),
    )


class MemmappedDataset(Dataset):
    r"""
    Dataset that takes a memory mapped numpy array from disk and returns chunks of length context_size of it.
    """

    data: np.ndarray

    def __init__(self, path: str, n_tokens: int = 1024) -> None:
        super().__init__()
        self.path = path
        self.max_seq_len = (
            n_tokens  # naming needed for timestamp to work properly for tokens
        )
        # Note:
        # Torch will try to pickle this object, when workers > 0. (to send it to the worker threads)
        # If we memmap the entire dataset it will happily try to pickle it and dump the dataset to disk..
        # In order to work around this we set the data to zero.
        self.data = None

    def __getitem__(self, index: int):
        if self.data is None:
            # safe to store it now, we're on the worker thread.
            self.data = np.memmap(self.path, dtype=np.int32, mode="readonly")

        start = index * (self.max_seq_len + 1)
        end = (index + 1) * (self.max_seq_len + 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = torch.as_tensor(self.data[start:end], dtype=torch.long)
        return data

    def __len__(self):
        if self.data is None:
            # See above note, before we call the first __getitem__ we cannot store the mmap file.
            data = np.memmap(self.path, dtype=np.int32, mode="readonly")
            return data.shape[0] // (self.max_seq_len + 1)

        return self.data.shape[0] // (self.max_seq_len + 1)


def get_mamba_dataloader(
    path: str | list[tuple[str, float, int]],
    batch_size: int,
    seq_len: int,
    n_data_parallel: int,
    rank: int,
    position_weighting: bool = False,
    n_samples_to_skip: int = 0,
    num_workers: int = 1,
    prefetch_factor: int = 2,
):
    collate_fct = partial(collate_mamba_batch, position_weighting=position_weighting)
    dataset = get_dataset(path, seq_len)
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


class GroupedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cum_dataset_lengths = np.cumsum([len(d) for d in self.datasets])

    def __getitem__(self, index: int):
        current_dataset_idx = int(np.where(index < self.cum_dataset_lengths)[0][0])
        offset = (
            0
            if current_dataset_idx == 0
            else int(self.cum_dataset_lengths[current_dataset_idx - 1])
        )
        return self.datasets[current_dataset_idx][index - offset]

    def __len__(self) -> int:
        return int(self.cum_dataset_lengths[-1])


class MergedDataset(Dataset):
    r"""
    Takes a list containing tuples of datasets and their sampling weight and returns a combined dataset.
    For simplicity we do not respect the sampling probabilities exactly. But it will be approximately correct.
    Longer sampling pattern will result in less error against specified sampling weight globally but more error locally
    This is the standard low-discrepancy trade-off
    """

    def __init__(
        self,
        datasets_weights_offsets: list[tuple[Dataset, float, int]],
        sampling_pattern_length=1_000,
        seed=42,
    ) -> None:
        super().__init__()
        # note sampling pattern length is approximate it may be slightly longer or shorter.
        datasets, weights, offsets = zip(*datasets_weights_offsets)
        weights = np.array(weights)

        weights *= sampling_pattern_length / weights.sum()
        weights = weights.round().astype(int)

        sampling_pattern: list[float] = []
        for i, w in enumerate(weights):
            sampling_pattern += [i] * int(w)

        np.random.seed(seed)
        self.sampling_pattern = np.random.permutation(sampling_pattern)
        self.datasets = datasets
        self.num_samples_per_pattern = weights
        self.sampling_probabilities = weights / len(self.sampling_pattern)

        self.in_pattern_counter = cumcount(self.sampling_pattern)
        self.offsets = offsets

    def __getitem__(self, index: int):
        in_index = index % len(self.sampling_pattern)
        num_completed_patterns = index // len(self.sampling_pattern)
        ds_index = self.sampling_pattern[in_index]
        ds = self.datasets[ds_index]
        completed_offset = (
            self.num_samples_per_pattern[ds_index] * num_completed_patterns
        )
        inside_offset = self.in_pattern_counter[in_index]

        length = len(ds)
        return ds[(completed_offset + inside_offset + self.offsets[ds_index]) % length]

    def __len__(self):
        return sum([len(d) for d in self.datasets])  # sketchy


def get_dataset(
    data_path: str | list[tuple[str, float, int]],
    seq_len: int = 2048,
    dataset_cls: Dataset = MemmappedDataset,
) -> MergedDataset | None:
    def check_extension(data_path):
        assert Path(data_path).suffix == ".npy"

    def get_grouped_dataset(f: str, seq_len: int) -> GroupedDataset:
        if f.endswith("npy"):
            sorted_matching_files = [f]
        else:
            matching_files = glob.glob(f)
            if len(matching_files) == 0:
                assert (
                    len(matching_files) > 0
                ), f"Unable to find {f} in {matching_files}"

            sorted_matching_files = sorted(matching_files)

        grouped_memmap_datasets = [
            dataset_cls(matching_file, n_tokens=seq_len)
            for matching_file in sorted_matching_files
        ]
        return GroupedDataset(grouped_memmap_datasets)

    datasets: list[tuple[GroupedDataset, float, int]] = []

    data_paths: list[tuple[str, float, int]] = []
    if type(data_path) is not list:
        data_paths.append((data_path, 1.0, 0))
    else:
        data_paths = data_path

    for f, w, o in data_paths:
        check_extension(f)
        grouped_dataset = get_grouped_dataset(
            f,
            seq_len,
        )
        datasets.append((grouped_dataset, w, o))

    assert len(datasets) > 0, f"found no .npy files in {data_path=}"
    return MergedDataset(datasets)


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


# https://stackoverflow.com/questions/40602269/how-to-use-numpy-to-get-the-cumulative-count-by-unique-values-in-linear-time
def dfill(a):
    n = a.size
    b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [n]])
    return np.arange(n)[b[:-1]].repeat(np.diff(b))


def argunsort(s):
    n = s.size
    u = np.empty(n, dtype=np.int64)
    u[s] = np.arange(n)
    return u


def cumcount(a):
    n = a.size
    s = a.argsort(kind="mergesort")
    i = argunsort(s)
    b = a[s]
    return (np.arange(n) - dfill(b))[i]


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
