# How to use?

This repository contains a demo of how to run the training code for the Mamba model. Here, you will find instructions on how to install the required packages, run the training script, and data format.

 ## Requirements

To set up the environment and install the necessary packages, follow these steps:

1. Create a virtual environment with Python 3.10 or higher.
2. Install Composer in editable mode along with its dependencies:
   ```bash
   pip install -e .[all]
   ```
3. Install the `mamba-ssm` and `causal-conv1d` packages for efficient CUDA kernels:
   ```bash
   pip install causal-conv1d>=1.2.0
   pip install mamba-ssm>=1.2.0
   ```

 ## Run

To run the Mamba training script, follow these instructions for single-node and multi-node setups.

### Single Node (8 GPUs Example)

You can run the training on a single node using one of the following methods:

1. Using `composer`:
   ```bash
   composer -n 8 mamba/train.py
   ```
2. If you prefer to use SLURM, modify `slurm.sh` according to your setup and submit your job with:
   ```bash
   sbatch slurm.sh
   ```

 ### Multi Node (2 Nodes, 16 GPUs Example)

To run the training on multiple nodes, ensure you have a suitable distributed cluster environment and follow these steps:

1. Modify `slurm.sh` according to your setup, specifying the appropriate resources for 2 nodes and 16 GPUs.
2. Submit your job with:
   ```bash
   sbatch slurm.sh
   ```

 ## Options

The training process can be configured using the `config.yaml` file, which includes several items:

### model
- `vocab_size`: The size of the vocabulary used in the model. In this example, it is set to 65024.
- `d_model`: The embedding size for each token in the input sequence. Here, it is set to 2688.
- `n_layer`: The number of layers (Mamba blocks) in the model. This example uses 28 layers.
- `fsdp_layer_wrap`: A boolean indicating whether to use Fully Sharded Data Parallelism (FSDP) layer wrapping or not. It is set to true in this example.
- `activation_checkpointing`: A boolean indicating whether to use activation checkpointing during training or not; it is enabled here with a value of true. 
### data
- `global_batch_size`: The total number of samples processed in parallel across all GPUs during training. In this example, it is set to 896.
- `micro_batch_size`: The number of samples processed simultaneously on each GPU within a single iteration (gradient accumulation step). It is set to 7 in this case.
- `eval_batch_size`: The batch size used during evaluation and testing phases, which can differ from the training batch sizes. Here, it has a value of 8.
- `seq_len`: The maximum sequence length supported by the model; any input sequence longer than this will be truncated or split into chunks with lengths not exceeding this limit (4096 in this example).
- `num_workers`: Number of subprocesses used for data loading; these workers fetch data from disk and feed them to the model during training or evaluation (set to 2 here).
- `prefetch_factor`: A multiplier that determines how many batches should be prefetched ahead when using multiprocessed data loading; increasing prefetch factor improves performance but consumes more memory (value is 2 here).
- `n_total_tokens`: Total number of tokens present in your dataset after tokenization.
### optimizer
* `lr` (learning rate): it is set to 0.00045.
* `beta1`: The exponential decay rate for the first moment in the AdamW optimizer's update rule. It is set to 0.9 here.
* `beta2`: The exponential decay rate for the second moment in the AdamW optimizer's update rule. It is set to 0.95 here.
* `weight_decay`: The weight decay regularization. In this case, it is set to 0.1.
### FSDP
+ `sharding_strategy`: Set to "FULL\_SHARD" to shard model parameters across all devices.
+ `process_group`: Null, default process group is used.
+ `backward_prefetch`: Set to "BACKWARD\_PRE" for backward prefetching of gradients.
+ `activation_checkpointing_reentrant`: True allows activation checkpointing reentrant.
+ `activation_checkpointing`: True enables activation checkpointing(block-wise).
+ `limit_all_gathers`: False indicates that there's no limit on all-gather operations by default.
+ `verbose`: True sets the verbosity level.
+ *Note*: If you want to generate a new config file from scratch, run `create_mamba_config.py` script.

## Notes about the data format

The tokenized dataset is saved in a specific format using the `DataLayout` class, which defines three ranges for storing different types of information, each token is represented using `int32`:

```python
class DataLayout(Enum):
    token_bits = (0, 16)      # max number of tokens: 64k
    position_bits = (16, 30)  # max context length of 16k
    loss_target = (30, 31)    # just one bit.
```

- `token_bits` (range 0 to 16): This range stores the token IDs with a maximum capacity of 64k tokens. It uses 16 bits to represent each token's index in the vocabulary.
- `position_bits` (range 16 to 30): This range stores position embeddings with a maximum context length of 16k. It utilizes 14 bits for representing positions, allowing you to capture long sequences while keeping memory usage low.
- `loss_target` (range 30 to 31): This single bit is used for loss calculation during training by specifying which positions should be considered when computing losses. In this case, it only requires one bit since there are just two possible states: either a position contributes to the loss or not.

To use the provided dataloaders, you need to tokenize your dataset and save it in this exact layout format as described above. Alternatively, if you prefer not to follow this specific layout or have unique requirements, you can create and use your own dataloaders that suit your needs best.