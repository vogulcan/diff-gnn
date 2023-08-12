import argparse


def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()

    enc_parser.add_argument("--dataset", type=str, help="Dataset")

    enc_parser.add_argument("--conv_type", type=str, help="type of convolution")
    enc_parser.add_argument("--n_layers", type=int, help="Number of graph conv layers")
    enc_parser.add_argument("--batch_size", type=int, help="Training batch size")
    enc_parser.add_argument("--hidden_dim", type=int, help="Training hidden size")
    enc_parser.add_argument("--input_dim", type=int, help="Training hidden size")
    enc_parser.add_argument("--dropout", type=float, help="Dropout rate")
    enc_parser.add_argument("--margin", type=float, help="margin for loss")

    enc_parser.add_argument(
        "--edge_margin", type=float, help="margin for base differentian of edges"
    )
    enc_parser.add_argument("--min_size", type=float, help="min_size")
    enc_parser.add_argument("--max_size_Q", type=float, help="max_size for query")
    enc_parser.add_argument("--max_size_T", type=float, help="max_size for target")

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

    enc_parser.add_argument("--n_workers", type=int)
    enc_parser.add_argument("--n_devices", type=int)
    enc_parser.add_argument("--tag", type=str, help="tag to identify the run")
    enc_parser.add_argument(
        "--save_path", type=str, help="path to save the model/run info"
    )

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
        save_path="./experiments",
        tag=None,
        edge_margin=0.15,
        min_size=31,
        max_size_Q=41,
        max_size_T=51,
    )
