from torch import optim

from composer import Trainer
from composer.algorithms import GradientClipping
from composer.callbacks import SpeedMonitor
from composer.core import Precision #Evaluator, 
from composer.loggers import WandBLogger
from composer.optim import CosineAnnealingWithWarmupScheduler
from composer.utils import dist

from create_mamba_config import load_config_from_yaml
from mamba import MambaModel
from datatrove import get_mamba_dataloader
from scheduler import WarmupStableDecayScheduler

from dataclasses import asdict


def main():
    # init configs
    (
        model_config,
        data_config,
        optimizer_config,
        scheduler_config,
        fsdp_config,
        trainer_config,
        general_config,
    ) = load_config_from_yaml("mamba/config_mambarabic.yaml")

    # build model
    model = MambaModel(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_layer=model_config.n_layer,
        fsdp_layer_wrap=model_config.fsdp_layer_wrap,
        activation_checkpointing=model_config.activation_checkpointing,
    )
    n_params_str = (
        "{:.1f}".format(sum([p.numel() for p in model.parameters()]) / 1e9) + "B"
    )
    print(f"The model has : {n_params_str} parameters.")

    # train_paths = [
    #     (pc.path, pc.weight, pc.order)
    #     for pc in data_config.code_paths + data_config.text_paths
    # ]

    train_paths = "s3://mambadata/arabic/tokenization-v1/merged-document/000_Arabic-data-merged.ds"
    print(train_paths)
    # create train dataloader
    train_dataloader = get_mamba_dataloader(
        path=train_paths,
        n_data_parallel=dist.get_world_size(),
        rank=dist.get_global_rank(),
        seq_len=data_config.seq_len,
        batch_size=int(data_config.global_batch_size / dist.get_world_size()),
        position_weighting=general_config.position_weighting,
        n_samples_to_skip=data_config.n_samples_to_skip,
        num_workers=data_config.num_workers,
        prefetch_factor=data_config.prefetch_factor,
    )

    # # create val dataloader
    # french_eval_dataloader = get_mamba_dataloader(
    #     path="/home/shared/data/mamba-v1/redpajama-v2-sample-100B-300-shards__fr__dedup-gopher-c4.validation.gig.npy",
    #     n_data_parallel=dist.get_world_size(),
    #     rank=dist.get_global_rank(),
    #     seq_len=data_config.seq_len,
    #     batch_size=data_config.eval_batch_size,
    # )
    # english_eval_dataloader = get_mamba_dataloader(
    #     path="/home/shared/data/mamba-v1/redpajama-v2-sample-100B-30-shards__en__dedup-gopher-c4.validation.gig.npy",
    #     n_data_parallel=dist.get_world_size(),
    #     rank=dist.get_global_rank(),
    #     seq_len=data_config.seq_len,
    #     batch_size=data_config.eval_batch_size,
    # )
    # evaluators = [
    #     Evaluator(label="french", dataloader=french_eval_dataloader),
    #     Evaluator(label="english", dataloader=english_eval_dataloader),
    # ]

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
    wandb_logger = WandBLogger(
        project=general_config.project_name,
        name=f"{n_params_str}_{general_config.full_name}",
    )
    speed_monitor = SpeedMonitor(window_size=general_config.window_size)

    # trainer
    trainer = Trainer(
        run_name=general_config.full_name,
        autoresume=trainer_config.autoresume,
        model=model,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=[gradient_clipper],
        train_dataloader=train_dataloader,
        max_duration=scheduler_config.t_max,
        loggers=[wandb_logger],
        callbacks=[speed_monitor],
        fsdp_config=asdict(fsdp_config),
        precision=Precision.AMP_BF16,
        device_train_microbatch_size=data_config.micro_batch_size,
#        eval_dataloader=evaluators,
        eval_interval=trainer_config.eval_interval,
        eval_subset_num_batches=trainer_config.eval_subset_num_batches,
        save_folder=trainer_config.save_folder,
        save_interval=trainer_config.save_interval,
        auto_log_hparams=trainer_config.auto_log_hparams,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
