import yaml

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import List, Optional, Tuple


def get_n_tokens(data_path: str):
    size_path = ".".join([data_path, "size"])
    with open(size_path, "r") as f:
        content = f.read()
    n_tokens = int(content.strip())
    return n_tokens


def safe_asdict(obj):
    if not is_dataclass(obj):
        raise TypeError("safe_asdict() should be called on dataclass instances")

    result = {}
    for f in fields(obj):
        value = getattr(obj, f.name, None)
        if value is not None:
            result[f.name] = value
    return result


@dataclass
class ModelConfig:
    vocab_size: int = 64000
    d_model: int = 1024
    d_intermediate: int = 0
    n_layer: int = 48
    fsdp_layer_wrap: bool = True
    activation_checkpointing: bool = True
    ssm_cfg_layer: str = "Mamba1"


@dataclass
class PathConfig:
    path: str
    # weight: float
    # order: int


@dataclass
class DataConfig:
    global_batch_size: int = 512
    micro_batch_size: int = 4
    eval_batch_size: int = 8
    seq_len: int = 4096
    num_workers: int = 1
    prefetch_factor: int = 2
    n_total_tokens: int = 0
    n_samples_to_skip: int = 0
    max_tokens: int = 0
    token_size: int = 2
    text_paths: List[PathConfig] = field(default_factory=list)
    code_paths: List[PathConfig] = field(default_factory=list)
    code_percentage: float = field(init=False)
    text_percentage: float = field(init=False)

    # def __post_init__(self):
    #     self.code_percentage = sum(p.weight for p in self.code_paths)
    #     self.text_percentage = sum(p.weight for p in self.text_paths)


@dataclass
class OptimizerConfig:
    lr: float = 2e-3
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    t_warmup: str = "1006878720tok"
    t_max: str = "10068787200tok"
    t_decay: str = "1006878720tok"
    alpha_f: float = 0.1


@dataclass
class FSDPConfig:
    sharding_strategy: str = "FULL_SHARD"
    process_group: Optional[Tuple[str, str]] = None
    backward_prefetch: str = "BACKWARD_PRE"
    activation_checkpointing_reentrant: bool = True
    activation_checkpointing: bool = True
    limit_all_gathers: bool = False
    verbose: bool = True


@dataclass
class TrainerConfig:
    cp_interval: str = "1006878720tok"
    eval_interval: str = "1006878720tok"
    save_interval: str = "1006878720tok"
    eval_subset_num_batches: int = 16
    save_folder: str = "checkpoints"
    auto_log_hparams: bool = True
    autoresume: bool = True
    load_path: str | None = ""
    save_num_checkpoints_to_keep: int = 1


@dataclass
class GeneralConfig:
    project_name: str = "mamba-science"
    full_name: str = "mambaoutai"
    window_size: Optional[int] = 10
    clipping_threshold: float = 1.0
    position_weighting: bool = False


def instantiate_data_config(TEXT_PATHS, CODE_PATHS, N_TOTAL_TOKENS):
    # TEXT_PATHS
    text_weights = [
        (path, get_n_tokens(path) / N_TOTAL_TOKENS, 0) for path in TEXT_PATHS
    ]

    text_paths = [PathConfig(path=path) for path in text_weights]

    data_config = DataConfig(
        text_paths=text_paths,
    )

    return data_config


def write_config_to_yaml(
    model_config,
    data_config,
    optimizer_config,
    scheduler_config,
    fsdp_config,
    trainer_config,
    general_config,
    file_name="config_mambarabic.yaml",
):
    config = {
        "model": asdict(model_config),
        "data": asdict(data_config),
        "optimizer": asdict(optimizer_config),
        "scheduler": asdict(scheduler_config),
        "fsdp": asdict(fsdp_config),
        "trainer": asdict(trainer_config),
        "general": asdict(general_config),
    }
    with open(file_name, "w") as file:
        yaml.dump(config, file, sort_keys=False)


def load_config_from_yaml(file_name: str = "config.yaml"):
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)
    model_config = ModelConfig(**config["model"])

    # handle dataconfig separately
    text_paths = [PathConfig(**path) for path in config["data"]["text_paths"]]

    config["data"]["text_paths"] = text_paths

    # don't give post init kwargs to constructor
    data_config = DataConfig(
        **{
            k: v
            for k, v in config["data"].items()
            if k not in ["code_percentage", "text_percentage"]
        }
    )

    optimizer_config = OptimizerConfig(**config["optimizer"])
    scheduler_config = SchedulerConfig(**config["scheduler"])
    fsdp_config = FSDPConfig(**config["fsdp"])
    trainer_config = TrainerConfig(**config["trainer"])
    general_config = GeneralConfig(**config["general"])
    return (
        model_config,
        data_config,
        optimizer_config,
        scheduler_config,
        fsdp_config,
        trainer_config,
        general_config,
    )


def main():
    ### TEXT DATA ###

    OTHER_PATHS = [
        "/home/shared/data/mamba-v1/redpajama-v1__arxiv.train.gig.npy",
        "/home/shared/data/mamba-v1/redpajama-v1__book.train.gig.npy",
        "/home/shared/data/mamba-v1/redpajama-v1__wikipedia__en.train.gig.npy",
        "/home/shared/data/mamba-v1/redpajama-v1__wikipedia__fr.train.gig.npy",
    ]
    C4_PATHS = [
        "/home/shared/data/mamba-v1/redpajama-v2-sample-100B-30-shards__en__dedup-gopher-c4.train.gig.npy",
        "/home/shared/data/mamba-v1/redpajama-v2-sample-100B-300-shards__fr__dedup-gopher-c4.train.gig.npy",
    ]

    TEXT_PATHS = C4_PATHS + OTHER_PATHS

    ### CODE DATA ###

    CODE_PATHS = [
        "/home/shared/data/mamba-v1/the-stack-dedup__c.train.gig.npy",
        "/home/shared/data/mamba-v1/the-stack-dedup__cpp.train.gig.npy",
        "/home/shared/data/mamba-v1/the-stack-dedup__go.train.gig.npy",
        "/home/shared/data/mamba-v1/the-stack-dedup__java.train.gig.npy",
        "/home/shared/data/mamba-v1/the-stack-dedup__javascript.train.gig.npy",
        "/home/shared/data/mamba-v1/the-stack-dedup__php.train.gig.npy",
        "/home/shared/data/mamba-v1/the-stack-dedup__python.train.gig.npy",
        "/home/shared/data/mamba-v1/the-stack-dedup__rust.train.gig.npy",
    ]

    N_TOTAL_TOKENS = sum([get_n_tokens(path) for path in TEXT_PATHS + CODE_PATHS])
    print(f"{N_TOTAL_TOKENS=}")

    # default configurations
    model_config = ModelConfig()
    data_config = instantiate_data_config(TEXT_PATHS, CODE_PATHS, N_TOTAL_TOKENS)
    data_config.n_total_tokens = N_TOTAL_TOKENS
    optimizer_config = OptimizerConfig()
    scheduler_config = SchedulerConfig()
    fsdp_config = FSDPConfig()
    trainer_config = TrainerConfig()
    general_config = GeneralConfig()

    # save default configurations
    write_config_to_yaml(
        model_config,
        data_config,
        optimizer_config,
        scheduler_config,
        fsdp_config,
        trainer_config,
        general_config,
        file_name="mamba/config.yaml",
    )

    # init configs
    (
        model_config,
        data_config,
        optimizer_config,
        scheduler_config,
        fsdp_config,
        trainer_config,
        general_config,
    ) = load_config_from_yaml("mamba/config.yaml")

    # verify loaded configurations
    print(
        model_config,
        data_config,
        optimizer_config,
        scheduler_config,
        fsdp_config,
        trainer_config,
        general_config,
        sep="\n\n",
    )

    print(data_config.text_paths)
    print(data_config.code_paths)


if __name__ == "__main__":
    main()
