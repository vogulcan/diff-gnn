import utils
import data
import models
import config
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

#torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config.parse_encoder(parser)
    args = parser.parse_args()

    devices = torch.cuda.device_count()
    if devices > args.n_devices:
        exit(f"Too many devices avaiable than requested: {devices} > {args.n_devices}")

    step_dict = {
        "train": args.steps_per_train,
        "val": args.steps_per_val,
        "test": args.steps_per_test,
    }

    datamodule = data._hic_datamodule_pl(args, steps_per_epoch=step_dict)

    strategy = pl.strategies.DDPStrategy(accelerator="gpu", find_unused_parameters=True)

    tb_logger = TensorBoardLogger(save_dir=args.save_path, name=f"tag{args.tag}")

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_acc", save_top_k=10, mode="max", save_last=True, filename="{epoch}-{val_acc:.4f}"
    )
    trainer = pl.Trainer(
        strategy=strategy,
        devices=devices,
        max_epochs=args.n_epoch,
        logger=tb_logger,
        log_every_n_steps=5,
        callbacks=[checkpoint],
    )

    model = models.pl_model(args)

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)
