import argparse
from config import parse_predict, merge_args
from data import _cool_datamodule_pl

import torch
import pytorch_lightning as pl
from models import pl_model_predict


import pandas as pd
import numpy as np

# from pytorch_lightning.callbacks import BasePredictionWriter
# class CustomWriter(BasePredictionWriter):
#     def __init__(self, output_dir, write_interval):
#         return

#     def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
#         raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parse_predict(parser)
    predict_args = parser.parse_args()

    checkpoint = torch.load(predict_args.checkpoint_path)
    hyper_parameters = checkpoint["hyper_parameters"]
    hyper_parameters = merge_args(predict_args, checkpoint["hyper_parameters"]["args"])

    model = pl_model_predict(**hyper_parameters)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    comp_data_module = _cool_datamodule_pl(**hyper_parameters)

    strategy = pl.strategies.DDPStrategy(accelerator="gpu", find_unused_parameters=True)
    devices = torch.cuda.device_count()
    trainer = pl.Trainer(inference_mode=True, devices=devices, strategy=strategy)

    outputs = {
        f"{comp_data_module.predict_dataset.comp}": trainer.predict(
            model, datamodule=comp_data_module
        )
    }
    comp_data_module.predict_dataset.flip()
    outputs.update(
        {
            f"{comp_data_module.predict_dataset.comp}": trainer.predict(
                model, datamodule=comp_data_module
            )
        }
    )

    comps = list(outputs.keys())

    chr_names = comp_data_module.predict_dataset.chr_names
    loaders = comp_data_module.predict_dataset.loaders

    res_df = {'chr': [], 'start': [], 'end': [], comps[0]: [], comps[1]: [], f'{comps[0]}-clf': [], f'{comps[1]}-clf': []}

for res1, res2 in zip(outputs[comps[0]], outputs[comps[1]]):

    pred1, pred2 = res1['pred'], res2['pred']
    meta1, meta2 = res1['metadata'], res2['metadata']
    dummy1, dummy2 = res1['dummy'], res2['dummy']

    assert meta1[0] == meta2[0]
    assert meta1[1] == meta2[1]

    if dummy1 or dummy2:
        continue
    else:
        common_idx = [int(i) for i in meta1[2] if i in meta2[2]]
        common_1 = sum(meta1[2]==i for i in common_idx).bool()
        common_2 = sum(meta2[2]==i for i in common_idx).bool()

        pred1, pred2 = pred1[common_1], pred2[common_2]

        chr_idx, chunk_idx = meta1[0], meta1[1]
        chr_name = chr_names[chr_idx]
        coords_idx = loaders[chr_idx][chunk_idx]
        start_coord = int(chunk_idx * hyper_parameters['args'].batch_size * 10000)
        coords = np.array(coords_idx)[common_idx]
        coords *= 10000
        coords += start_coord

        res_df['chr'].extend([chr_name] * len(coords))
        res_df['start'].extend(coords.tolist())
        res_df['end'].extend((coords + 10000).tolist())
        res_df[comps[0]].extend(pred1.tolist())
        res_df[comps[1]].extend(pred2.tolist())
        res_df[f'{comps[0]}-clf'].extend(model.clf_model(pred1).argmax(1).detach().numpy().tolist())
        res_df[f'{comps[1]}-clf'].extend(model.clf_model(pred2).argmax(1).detach().numpy().tolist())

    res_df = pd.DataFrame(res_df)
    res_df.to_csv(hyper_parameters['args'].res_path, index=False, sep='\t')