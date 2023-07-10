import time

import hydra
import pytorch_lightning as pl
import wandb
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from datamodules.babelnet_dm import BabelNetDataModule
from models.model import BabelNetTransformer


@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    loggers = [init_wandb(cfg)] if cfg.logger.name == "wandb" else []
    loggers.append(CSVLogger("csv_logs"))

    babelnet_dm = BabelNetDataModule(
        model_name=cfg.model.encoder_name,
        alpha=cfg.dataset.alpha,
        batch_size=cfg.dataset.batch_size,
        train_path=to_absolute_path(cfg.dataset.dir),
        val_bli_file=cfg.val.bli_file,
        sel_langs=cfg.dataset.sel_langs,
        input_type=cfg.model.input_type,
        val_train_overlap=cfg.dataset.val_train_overlap,
        vocab_dir=cfg.val.vocab_dir,
    )

    babelnet_model = BabelNetTransformer(
        model_name=cfg.model.encoder_name,
        training_type=cfg.model.training_type,
        learning_rate=cfg.model.learning_rate,
        loss=cfg.model.loss,
        similarity=cfg.model.similarity,
        word_repr_type=cfg.model.word_repr_type,
        reduction_factor=cfg.model.adapter.reduction_factor,
        max_epochs=cfg.trainer.max_epochs,
        warmup_peak=cfg.model.warmup_peak,
        temperature=cfg.model.temperature,
        checkpoint_file=cfg.model.checkpoint_file,
        layerwise_averaging=cfg.model.layerwise_averaging,
        weight_decay=cfg.model.weight_decay,
    )

    gpu_id = [cfg.trainer.gpus]
    print(f"Using GPU {gpu_id[0]} as requested.")
    callbacks = []
    if babelnet_model.warmup_peak != -1:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        print(f"Logging learning rate.")

    trainer = pl.Trainer(
        gpus=gpu_id,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        fast_dev_run=cfg.trainer.fast_dev_run,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        val_check_interval=cfg.trainer.val_check_interval,
        deterministic=cfg.trainer.deterministic,
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=False,
        overfit_batches=cfg.trainer.overfit_batches,
        limit_train_batches=cfg.trainer.limit_train_batches,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
    )

    if cfg.val_only:
        print("--- Validation only ---")
    trainer.validate(babelnet_model, datamodule=babelnet_dm)
    if not cfg.val_only:
        if cfg.trainer.ckpt_flag:
            print("Checkpoint saving activated.")
            mode = "max"
            metric = "val/mrr_bli_tr_avg_diff"
            checkpoint_callback = ModelCheckpoint(
                monitor=metric,
                mode=mode,
                dirpath=cfg.checkpoint_path,
                filename=f"epoch_{{epoch}}-step_{{step}}-mrr_avg_diff"
                + f"_{{{metric}:.2f}}",
                auto_insert_metric_name=False,
            )
            trainer.callbacks.append(checkpoint_callback)
        if cfg.trainer.patience != -1:
            print(
                f"Using early stopping with patience {cfg.trainer.patience} and min improvement of {cfg.trainer.min_delta}"
            )
            metric = "val/mrr_bli_tr_avg_diff"
            trainer.callbacks.append(
                EarlyStopping(
                    monitor=metric,
                    min_delta=cfg.trainer.min_delta,
                    patience=cfg.trainer.patience,
                    verbose=True,
                    mode=mode,
                )
            )
            trainer.callbacks.append(LogEarlyStopping())

        trainer.fit(babelnet_model, datamodule=babelnet_dm)
    if cfg.trainer.ckpt_flag:
        print(f" +++ Best checkpoint path {checkpoint_callback.best_model_path}")
        if cfg.logger.name == "wandb":
            wandb.config.update(
                {"best_model_path": checkpoint_callback.best_model_path}
            )
    if cfg.logger.name == "wandb":
        wandb.finish()
        time.sleep(180)
    print("***** Training Finished *****")


def init_wandb(cfg):
    yaml_dict = yaml.load(OmegaConf.to_yaml(cfg), Loader=yaml.FullLoader)
    run_name = f"{cfg.model.encoder_name}_lr_{str(cfg.model.learning_rate)}_alpha_{cfg.dataset.alpha}_batch_{cfg.dataset.batch_size}"
    group = cfg.logger.group
    logger = WandbLogger(
        project=cfg.logger.project,
        log_model=cfg.logger.upload_checkpoints,
        name=run_name,
        config=yaml_dict,
        group=group,
    )
    if group is not None:
        print(f"Run belongs to group {group}")
    if cfg.logger.upload_checkpoints is True or cfg.logger.upload_checkpoints == "all":
        print("Uploading checkpoints to W&B.")
    return logger


class LogEarlyStopping(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, *args):
        early_stopping_callbacks = [
            c for c in trainer.callbacks if isinstance(c, EarlyStopping)
        ]
        if early_stopping_callbacks:
            for c in early_stopping_callbacks:
                c.monitor
                self.log(f"{c.monitor}_best", c.best_score)


if __name__ == "__main__":
    main()
