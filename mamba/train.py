import torch
from torch import optim

from composer import Trainer
from composer.algorithms import GradientClipping
from composer.callbacks import SpeedMonitor
from composer.core import Evaluator, Precision
from composer.loggers import WandBLogger
from composer.optim import CosineAnnealingWithWarmupScheduler
from composer.utils import dist

from create_mamba_config import load_config_from_yaml, safe_asdict
from mamba import Mamba2Model
from mamba_ssm.models.config_mamba import MambaConfig

# from datatrove import get_mamba_dataloader
from gdr import get_mamba_dataloader
from scheduler import WarmupStableDecayScheduler
from dataclasses import asdict
import argparse
import gc
import os


def main():
    parser = argparse.ArgumentParser(description="Read config from YAML file.")
    parser.add_argument(
        "--config",
        type=str,
        default="mamba/configs/config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    # init configs
    (
        model_config,
        data_config,
        optimizer_config,
        scheduler_config,
        fsdp_config,
        trainer_config,
        general_config,
    ) = load_config_from_yaml(args.config)

    # model config
    cfg = MambaConfig(
        d_model=model_config.d_model,
        d_intermediate=model_config.d_intermediate,
        n_layer=model_config.n_layer,
        vocab_size=model_config.vocab_size,
        ssm_cfg={
            "layer": model_config.ssm_cfg_layer,
            "headdim": model_config.headdim
            },
        tie_embeddings=model_config.tie_embeddings,
    )

    # build model
    model = Mamba2Model(cfg)

    print(model)
    print(sum([p.numel() for p in model.parameters()]))
    n_params_str = (
        "{:.2f}".format(sum([p.numel() for p in model.parameters()]) / 1e9) + "B"
    )
    print(f"The model has : {n_params_str} parameters.")

    train_paths = [(pc.path) for pc in data_config.text_paths]

    train_path = train_paths[0]

    print("train_path", train_path)
    print("Number of GPU:s", dist.get_world_size())
    print("{:.1f}".format(data_config.n_total_tokens / 1e9) + "B tokens")

    # create train dataloader
    train_dataloader = get_mamba_dataloader(
        path=train_path,
        batch_size=int(data_config.global_batch_size / dist.get_world_size()),
        seq_len=data_config.seq_len,
        n_data_parallel=dist.get_world_size(),
        rank=dist.get_global_rank(),
        position_weighting=False,
        n_samples_to_skip=data_config.n_samples_to_skip,
        num_workers=data_config.num_workers,
        prefetch_factor=data_config.prefetch_factor,
        # max_tokens=data_config.n_total_tokens,
        # token_size=data_config.token_size,
    )

    # create val dataloader
    eval_dataloader = get_mamba_dataloader(
        path=train_path,
        batch_size=data_config.eval_batch_size,
        seq_len=data_config.seq_len,
        n_data_parallel=dist.get_world_size(),
        rank=dist.get_global_rank(),
        n_samples_to_skip=0,
        num_workers=data_config.num_workers,
        prefetch_factor=None,
        # max_tokens=61571512,
        # token_size=data_config.token_size,
    )
    evaluator = [
        Evaluator(label="english", dataloader=eval_dataloader),
    ]

    # create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_config.lr,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
    )

    # create LR scheduler
    if scheduler_config.name == "cosine":
        lr_scheduler = CosineAnnealingWithWarmupScheduler(
            t_warmup=scheduler_config.t_warmup,
            t_max=scheduler_config.t_max,
            alpha_f=scheduler_config.alpha_f,
        )
    elif scheduler_config.name == "wsd":
        lr_scheduler = WarmupStableDecayScheduler(
            t_warmup=scheduler_config.t_warmup,
            t_max=scheduler_config.t_max,
            t_decay=scheduler_config.t_decay,
            alpha_f=scheduler_config.alpha_f,
        )
    else:
        raise ValueError(
            f"scheduler: {scheduler_config.name} is not supported. Please use one of [cosine, wsd]"
        )

    # algorithms
    gradient_clipper = GradientClipping(
        clipping_type="norm",
        clipping_threshold=general_config.clipping_threshold,
    )

    # monitoring
    run_name = f"{n_params_str}_{general_config.full_name}-{model_config.n_layer}-{model_config.d_model}"
    wandb_logger = WandBLogger(
        project=general_config.project_name,
        name=run_name,
    )
    speed_monitor = SpeedMonitor(window_size=general_config.window_size)


    print(f"{trainer_config.load_path=}")
    print(f"{fsdp_config.sharding_strategy=}")
    print(f"{fsdp_config=}")
    
    # trainer
    trainer = Trainer(
        run_name=general_config.full_name,
        autoresume=trainer_config.autoresume,
        load_path=trainer_config.load_path,
        model=model,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=[gradient_clipper],
        train_dataloader=train_dataloader,
        max_duration=scheduler_config.t_max,
        loggers=[wandb_logger],
        callbacks=[speed_monitor],
        fsdp_config=asdict(fsdp_config) if fsdp_config.sharding_strategy is not None else None,
        precision=Precision.AMP_BF16,
        device_train_microbatch_size=data_config.micro_batch_size,
        eval_dataloader=evaluator,
        eval_interval=trainer_config.eval_interval,
        eval_subset_num_batches=trainer_config.eval_subset_num_batches,
        save_folder=f"{trainer_config.save_folder}/{run_name}",
        save_interval=trainer_config.save_interval,
        auto_log_hparams=trainer_config.auto_log_hparams,
        save_num_checkpoints_to_keep=trainer_config.save_num_checkpoints_to_keep,
    )

    # log the config to wandb
    if os.getenv("WANDB_API_KEY"):
        import wandb

        if wandb.run:
            wandb.config.update(safe_asdict(model_config), allow_val_change=True)
            wandb.config.update(safe_asdict(data_config), allow_val_change=True)
            wandb.config.update(safe_asdict(optimizer_config), allow_val_change=True)
            wandb.config.update(safe_asdict(scheduler_config), allow_val_change=True)
            wandb.config.update(safe_asdict(fsdp_config), allow_val_change=True)
            wandb.config.update(safe_asdict(trainer_config), allow_val_change=True)
            wandb.config.update(safe_asdict(general_config), allow_val_change=True)

    # clean memory
    torch.cuda.empty_cache()
    gc.collect()

    trainer.fit()


if __name__ == "__main__":
    main()
