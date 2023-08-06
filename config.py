import argparse


def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()

    enc_parser.add_argument("--conv_type", type=str, help="type of convolution")
    enc_parser.add_argument("--batch_size", type=int, help="Training batch size")
    enc_parser.add_argument("--n_layers", type=int, help="Number of graph conv layers")
    enc_parser.add_argument("--hidden_dim", type=int, help="Training hidden size")
    enc_parser.add_argument("--input_dim", type=int, help="Training hidden size")

    enc_parser.add_argument("--skip", type=str, help='"all" or "last"')
    enc_parser.add_argument("--dropout", type=float, help="Dropout rate")
    enc_parser.add_argument(
        "--n_batches", type=int, help="Number of training minibatches"
    )
    enc_parser.add_argument("--margin", type=float, help="margin for loss")
    enc_parser.add_argument(
        "--edge_margin", type=float, help="margin for base differentian of edges"
    )
    enc_parser.add_argument("--min_size", type=float, help="min_size")
    enc_parser.add_argument("--max_size_Q", type=float, help="max_size for query")
    enc_parser.add_argument("--max_size_T", type=float, help="max_size for target")
    enc_parser.add_argument(
        "--sampling_size",
        type=float,
        help="Number of graphs sampled from dataset to create a smaller dataset at each eval interval",
    )

    enc_parser.add_argument("--dataset", type=str, help="Dataset")
    enc_parser.add_argument("--test_set", type=str, help="test set filename")
    enc_parser.add_argument(
        "--eval_interval", type=int, help="how often to eval during training"
    )
    enc_parser.add_argument("--val_size", type=int, help="validation set size")
    enc_parser.add_argument(
        "--model_save_dir", type=str, help="path to directory for save/load model"
    )
    enc_parser.add_argument("--opt_scheduler", type=str, help="scheduler name")

    enc_parser.add_argument(
        "--use_curriculum",
        action="store_true",
        help="whether to use curriculum in training",
    )
    enc_parser.add_argument("--test", action="store_true")
    enc_parser.add_argument("--n_workers", type=int)
    enc_parser.add_argument("--tag", type=str, help="tag to identify the run")

    enc_parser.set_defaults(
        conv_type="GINE",
        dataset="/home/carlos/Desktop/projects/diff-gnn/datasets/hic_4DNFIBM9QCFG_nMax51_nMin31_perc10",
        n_layers=2,
        batch_size=32,
        hidden_dim=64,
        skip="learnable",
        dropout=0.1,
        n_batches=500000,
        opt="adam",  # opt_enc_parser
        opt_scheduler="cos",
        opt_restart=100,
        weight_decay=0.0,
        lr=1e-3,
        margin=0.1,
        test_set="",
        eval_interval=5000,
        n_workers=8,
        model_save_dir="./checkpoints",
        tag="",
        val_size=4096,
        min_size=31,
        max_size_Q=41,
        max_size_T=51,
        input_dim=2,
        edge_margin=0.15,
    )
