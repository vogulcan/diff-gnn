import argparse
import os
import sys
import tqdm
import json

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import models
import data
import utils
from testing import validation
from config import parse_encoder

from torch_geometric import seed_everything
seed_everything(0)

def build_model(args):
    model = models.Embedder(args.input_dim, args.hidden_dim, args)
    model.to(utils.get_device())
    if args.test and args.model_path:
        print(f"Loading model from {args.model_path}", flush=True)
        model.load_state_dict(
            torch.load(args.model_path, map_location=utils.get_device())
        )
    return model


def train(args, model, logger, in_queue, out_queue):
    scheduler, opt = utils.build_optimizer(args, model.parameters())
    clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)

    done = False
    local_epoch = 0

    while not done:
        local_epoch += 1

        data_source = data.DataSource(args)

        loaders = data_source.gen_data_loaders(
            args.eval_interval * args.batch_size, args.batch_size
        )

        for batch_target, _, _ in zip(*loaders):
            msg, _ = in_queue.get()
            if msg == "done":
                done = True
                break

            # train
            model.train()
            model.zero_grad()
            as_, bs_, labels = data_source.gen_batch(batch_target, train=True)

            if not as_:
                out_queue.put(("step", (-1, -1)))
                continue

            emb_as_= model.emb_model(as_.x, as_.edge_index, as_.edge_attr, as_.batch)
            emb_bs_ = model.emb_model(bs_.x, bs_.edge_index, bs_.edge_attr, bs_.batch)
            
            pred = model(emb_as_, emb_bs_)
            loss = model.criterion(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler:
                scheduler.step()

            with torch.no_grad():
                pred = model.predict(pred)
            model.clf_model.zero_grad()
            pred = model.clf_model(pred.unsqueeze(1))
            criterion = nn.NLLLoss()
            clf_loss = criterion(pred, labels)
            clf_loss.backward()
            clf_opt.step()

            pred = pred.argmax(dim=-1)
            acc = torch.mean((pred == labels).type(torch.float))
            train_loss = loss.item()
            train_acc = acc.item()

            out_queue.put(("step", (loss.item(), acc, scheduler.get_last_lr()[0])))


def train_loop(args):
    print(vars(args), flush=True)

    if args.tag != "":
        os.makedirs(args.model_save_dir, exist_ok = True) 
        args.model_path = f"{args.model_save_dir}/tag_{args.tag}.ckpt"
        
        with open(f"{args.model_path}.args.json", "w") as file:
            json.dump(vars(args), file, indent=1)
    else:
        sys.exit("\nPlease specify a tag for the model to save/load!")

    in_queue, out_queue = mp.Queue(), mp.Queue()

    logger = SummaryWriter(comment=f"-tag{args.tag}")

    model = build_model(args)
    model.share_memory()

    clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)

    data_source = data.DataSource(args, test_points=True)
    loaders = data_source.gen_data_loaders(
        args.val_size,
        args.batch_size,
    )

    test_pts = []
    for batch_target, _, _ in zip(*loaders):
        as_, bs_, labels = data_source.gen_batch(batch_target, train=False)
        if as_:
            as_ = as_.to(torch.device("cpu"))
            bs_ = bs_.to(torch.device("cpu"))
            labels = labels.to(torch.device("cpu"))
        test_pts.append((as_, bs_, labels))

    workers = []
    for i in range(args.n_workers):
        worker = mp.Process(target=train, args=(args, model, None, in_queue, out_queue))
        worker.start()
        workers.append(worker)

    if args.test:
        validation(args, model, test_pts, logger, 0, 0, verbose=True)
    else:
        batch_n = 0
        for epoch in range(args.n_batches // args.eval_interval):
            for i in range(args.eval_interval):
                
                in_queue.put(("step", None))

            for i in range(args.eval_interval):
                
                msg, params = out_queue.get()
                train_loss, train_acc, local_lr = params

                logger.add_scalar("Loss/train", train_loss, batch_n)
                logger.add_scalar("Accuracy/train", train_acc, batch_n)
                logger.add_scalar("LR/train", local_lr, batch_n)

                batch_n += 1

            validation(args, model, test_pts, logger, batch_n, epoch)
    
    print("Done training.", flush=True)

def main(test=False):
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if test:
        args.test = True
    train_loop(args)


if __name__ == "__main__":
    main()
