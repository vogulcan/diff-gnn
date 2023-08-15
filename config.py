import argparse


def parse_encoder(parser):
    enc_parser = parser.add_argument_group()

    enc_parser.add_argument("--dataset", type=str, help="Dataset")

    # NN
    enc_parser.add_argument("--conv_type", type=str, help="type of convolution")
    enc_parser.add_argument("--n_layers", type=int, help="Number of graph conv layers")
    enc_parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of graphs per batch as a batch of a single disconnected graph",
    )
    enc_parser.add_argument("--hidden_dim", type=int, help="Training hidden size")
    enc_parser.add_argument("--input_dim", type=int, help="Training hidden size")
    enc_parser.add_argument("--dropout", type=float, help="Dropout rate")
    enc_parser.add_argument(
        "--margin", type=float, help="Margin for embedding loss function/hinge loss"
    )

    # Data
    enc_parser.add_argument(
        "--edge_margin", type=float, help="Margin for base differentian of edges"
    )
    enc_parser.add_argument("--min_size", type=float, help="min_size")
    enc_parser.add_argument("--max_size_Q", type=float, help="max_size for query")
    enc_parser.add_argument("--max_size_T", type=float, help="max_size for target")

    # Optimization
    enc_parser.add_argument("--opt", dest="opt", type=str, help="Type of optimizer")
    enc_parser.add_argument(
        "--opt-scheduler",
        dest="opt_scheduler",
        type=str,
        help="Type of optimizer scheduler. By default none",
    )
    enc_parser.add_argument(
        "--opt-restart",
        dest="opt_restart",
        type=int,
        help="Number of epochs before restart (by default set to 0 which means no restart)",
    )
    enc_parser.add_argument("--lr", type=float)
    enc_parser.add_argument(
        "--opt-decay-step",
        dest="opt_decay_step",
        type=int,
        help="Number of epochs before decay",
    )
    enc_parser.add_argument(
        "--opt-decay-rate",
        dest="opt_decay_rate",
        type=float,
        help="Learning rate decay ratio",
    )
    enc_parser.add_argument(
        "--weight_decay", type=float, help="Optimizer weight decay."
    )

    # Environment
    enc_parser.add_argument("--n_workers", type=int, help="Number of workers")
    enc_parser.add_argument("--n_devices", type=int, help="Number of devices")
    enc_parser.add_argument("--n_nodes", type=int, help="Number of nodes")
    enc_parser.add_argument("--env", type=str, help="Environment")
    enc_parser.add_argument("--tag", type=str, help="Tag to label the run")
    enc_parser.add_argument(
        "--save_path", type=str, help="path to save the model/run info"
    )

    # Training
    enc_parser.add_argument("--n_epoch", type=int)
    enc_parser.add_argument("--steps_per_train", type=int)
    enc_parser.add_argument("--steps_per_val", type=int)
    enc_parser.add_argument("--steps_per_test", type=int)

    enc_parser.set_defaults(
        dataset="/home/carlos/Desktop/projects/diff-gnn/datasets/hic_4DNFIBM9QCFG_nMax51_nMin31_perc10",
        conv_type="GINE",
        n_layers=3,
        batch_size=64,
        hidden_dim=128,
        input_dim=2,
        dropout=0.0,
        margin=0.1,
        n_epoch=100,
        steps_per_train=5000,
        steps_per_val=24,
        steps_per_test=24,
        opt="adam",
        opt_scheduler="cos",
        opt_restart=100,
        weight_decay=0.0,
        lr=1e-3,
        n_workers=4,
        n_devices=1,
        n_nodes=1,
        env="local",
        save_path="./experiments",
        tag="debug",
        edge_margin=0.15,
        min_size=31,
        max_size_Q=41,
        max_size_T=51,
    )


def parse_predict(parser):
    predict_parser = parser.add_argument_group()

    predict_parser.add_argument(
        "--checkpoint_path", type=str, help="Path to checkpoint"
    )
    predict_parser.add_argument(
        "--method", type=str, help="Method to use for prediction"
    )
    predict_parser.add_argument(
        "--npz_path", type=str, help="Path to npz file with prediction data"
    )

    predict_parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of graphs per batch as a batch of a single disconnected graph",
    )

    predict_parser.add_argument("--sample1", type=str, help="Sample 1")
    predict_parser.add_argument("--sample2", type=str, help="Sample 2")
    predict_parser.add_argument("--results_save", type=str, help="Results path")
    predict_parser.add_argument("--n_workers", type=int, help="Number of workers")
    predict_parser.add_argument("--resolution", type=int, help="Resolution")

    predict_parser.set_defaults(method="raw", batch_size=512, n_workers=8, resolution=10000)


def merge_args(*args):
    # First arg has priority
    res = {}
    for arg in args:
        for k, v in vars(arg).items():
            if k in res:
                continue
            else:
                res[k] = v
    return {"args": argparse.Namespace(**res)}
