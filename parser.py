from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--pruning_method", type=str, default="dst", help="Which pruning method to use. DST | AutoSparse | STR.")
    parser.add_argument("--max_pruning_pct", type=float, default=0.99, help="Maximum pruning percentage for DST pruning method")
    parser.add_argument("--alpha", type=float, default=0.3, help="Value for alpha-hyperparameter used in AutoSparse and DST pruning methods.")
    parser.add_argument("--beta", type=float, default=0.3, help="Value for beta-hyperparameter used in CS pruning method.")
    parser.add_argument("--threshold_type", type=str, default="channelwise", help="For threshold based pruning, defines whether pruning is done layerwise, channel/neuronwise, or weightwise")
    parser.add_argument("--threshold_init", type=float, default=1, help="Initial value for thresholds")
    parser.add_argument("--sparsity", type=float, default=0.8, help="Target sparsity for a model")
    parser.add_argument("--temperature", type=int, default=1, help="Temperature parameter for softmax function")
    parser.add_argument("--l2_decay", type=float, default=0.001, help="l2 regularization weight-decay value")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate used during training")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used during training and validation")
    return parser