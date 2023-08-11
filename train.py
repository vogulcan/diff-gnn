import utils
import data
import models
from config import parse_encoder

import argparse

import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()
    
    #step_dict = {'train': 5000, 'val': 1000, 'test': 5000}
    step_dict = {'train': 100, 'val': 100, 'test': 50}
    
    datamodule = data._hic_datamodule_pl(args, steps_per_epoch = step_dict)

    devices = torch.cuda.device_count()
    #strategy = pl.strategies.DDPStrategy(accelerator="gpu", find_unused_parameters=True)
    strategy = pl.strategies.DDPStrategy(accelerator="gpu")

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_acc", save_top_k=1, mode="max"
    )
    trainer = pl.Trainer(
        strategy=strategy,
        devices=devices,
        max_epochs=5,
        log_every_n_steps=5,
        callbacks=[checkpoint],
    )

    model = models.pl_model(args)

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)
