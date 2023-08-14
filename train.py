import data
import models
import config
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config.parse_encoder(parser)
    args = parser.parse_args()

    devices = torch.cuda.device_count()
    if devices > args.n_devices:
        exit(f"Too many devices avaiable than requested: {devices} > {args.n_devices}")

    datamodule = data._hic_datamodule_pl(args)

    strategy = pl.strategies.DDPStrategy(accelerator="gpu", find_unused_parameters=True)

    tb_logger = TensorBoardLogger(save_dir=args.save_path, name=f"tag{args.tag}")

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        save_top_k=10,
        mode="max",
        save_last=True,
        filename="{epoch}-{val_acc:.4f}",
    )

    plugins = []
    if args.env == "slurm":
        from pytorch_lightning.plugins.environments import SLURMEnvironment
        import signal

        plugins.append(SLURMEnvironment(requeue_signal=signal.SIGHUP))

    trainer = pl.Trainer(
        strategy=strategy,
        devices=devices,
        num_nodes=args.n_nodes,
        max_epochs=args.n_epoch,
        logger=tb_logger,
        log_every_n_steps=5,
        callbacks=[checkpoint],
        plugins=plugins,
    )

    model = models.pl_model(args)

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)
