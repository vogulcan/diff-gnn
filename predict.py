import argparse
from config import parse_predict, merge_args
from data import _cool_datamodule_pl

import torch
import pytorch_lightning as pl
from models import pl_model_predict


import pandas as pd
import numpy as np
import bioframe as bf

#torch.set_float32_matmul_precision('medium')

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
    resolution = hyper_parameters['args'].resolution
    batch_size = hyper_parameters['args'].batch_size

    comps = list(outputs.keys())
    res_df = {
        'chr': [], 
        'start': [], 
        'end': [], 
        comps[0]: [], 
        comps[1]: [], 
        f'{comps[0]}-clf': [], 
        f'{comps[1]}-clf': []
        }

    
    hg38_chromsizes = bf.fetch_chromsizes('hg38')
    hg38_cens = bf.fetch_centromeres('hg38')
    hg38_arms = bf.make_chromarms(hg38_chromsizes,  hg38_cens)
    hg38_arms = hg38_arms.reset_index(drop=True)

    df_list = []
    for chrname in chr_names:
        start = int(hg38_arms[hg38_arms['name'] == chrname]['start'].iloc[0])
        end = int(hg38_arms[hg38_arms['name'] == chrname]['end'].iloc[0])
        df_local = {'chr': [], 'start': [], 'end': [], 'name': []}

        start_bin = start // resolution
        end_bin = end // resolution + 1

        if start_bin != 0:
            start_bin += 1
            end_bin += 1

        df_local['start'] = np.array(range(start_bin, end_bin)) * resolution
        df_local['chr'] = [chrname.split('_')[0]] * len(df_local['start'])
        df_local['end'] = df_local['start'] + resolution
        df_local['name'] = [chrname] * len(df_local['start'])
        df_local[comps[0]] = np.nan * np.zeros(len(df_local['start']))
        df_local[comps[1]] = np.nan * np.zeros(len(df_local['start']))
        df_local[f'{comps[0]}-clf'] = np.nan * np.zeros(len(df_local['start']))
        df_local[f'{comps[1]}-clf'] = np.nan * np.zeros(len(df_local['start']))
        df_list.append(pd.DataFrame(df_local))

    df = pd.concat(df_list, ignore_index=True)

    def holder(batch_size, pred, present_both):
        holder = np.nan * np.zeros((batch_size))
        holder[present_both] = pred
        return holder.copy()    

    for i, (res1, res2) in enumerate(zip(outputs[comps[0]], outputs[comps[1]])):

        pred1, pred2 = res1['pred'], res2['pred']
        meta1, meta2 = res1['metadata'], res2['metadata']
        dummy1, dummy2 = res1['dummy'], res2['dummy']

        assert meta1[0] == meta2[0]
        assert meta1[1] == meta2[1]

        if dummy1 or dummy2:
            continue
        else:
            
            present_both = [int(i) for i in meta1[2] if i in meta2[2]]
            pred1, pred2 = pred1[sum(meta1[2]==i for i in present_both).bool()], pred2[sum(meta2[2]==i for i in present_both).bool()]
            pred1 = holder(batch_size, pred1, present_both)
            pred2 = holder(batch_size, pred2, present_both)

            chr_idx, chunk_idx = meta1[0], meta1[1]
            current_start = int(chunk_idx * batch_size)
            current_present = np.array(present_both) + current_start

            chr_name = chr_names[chr_idx]
            start_index = df[df['name'] == chr_name].index[0]
            start_index += current_start

            clf1 = model.clf_model(torch.tensor(pred1, dtype=torch.float32).reshape(-1, 1)).argmax(1).detach().numpy()
            clf2 = model.clf_model(torch.tensor(pred2, dtype=torch.float32).reshape(-1, 1)).argmax(1).detach().numpy()
            
            size = batch_size
            if chunk_idx + 1 == len(loaders[chr_idx]):
                size = df.iloc[start_index:, :].shape[0]

            df.iloc[start_index:start_index+batch_size, 4] = pred1[:size]
            df.iloc[start_index:start_index+batch_size, 5] = pred2[:size]
            df.iloc[start_index:start_index+batch_size, 6] = clf1[:size]
            df.iloc[start_index:start_index+batch_size, 7] = clf2[:size]

    df.to_csv(predict_args.results_save, sep='\t', index=False)