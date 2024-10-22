from argparse import ArgumentParser, ArgumentTypeError

def str2bool(w):
    if w.lower() in ['true', 'y', 'yes']:
        return True
    elif w.lower() in ['false', 'no', 'n']:
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--pruning_method", type=str, default="dst", help="Which pruning method to use. DST | AutoSparse | STR.")
    parser.add_argument("--max_pruning_pct", type=float, default=0.99, help="Maximum pruning percentage for DST pruning method")
    parser.add_argument("--alpha", type=float, default=0.3, help="Value for alpha-hyperparameter used in AutoSparse and DST pruning methods.")
    parser.add_argument("--beta", type=float, default=0.3, help="Value for beta-hyperparameter used in CS pruning method.")
    parser.add_argument("--threshold_type", type=str, default="channelwise", help="For threshold based pruning, defines whether pruning is done layerwise, channel/neuronwise, or weightwise")
    parser.add_argument("--threshold_init", type=float, default=1, help="Initial value for thresholds")
    parser.add_argument("--sparsity", type=float, default=0.8, help="Target sparsity for a model")
    parser.add_argument("--temperature", type=float, default=1., help="Temperature parameter for softmax function")
    parser.add_argument("--l2_decay", type=float, default=0.001, help="l2 regularization weight-decay value")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate used during training")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used during training and validation")
    parser.add_argument("--final_temp", type=float, help="Target temperature in CS")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds in iterative pruning")
    parser.add_argument("--save_weights_epoch", type=int, default=-1, help="When to save weights. Weights are rewinded in CS to these weights.")
    parser.add_argument("--rewind", type=str, choices=("round", "post-ticket-search", "never"), default="never", help="Whether to do weight rewind after each round, post-ticket search, or never.")
    parser.add_argument("--fine_tune", type=str2bool, default=False, help="Whether to fine tune after training loop or not.")
    parser.add_argument("--epsilon", type=float, default=0.015, help="Used in PDP to scale layer pruning ratios")
    parser.add_argument("--pretraining_epochs", default=0, type=int, help="Number of pretraining epochs")
    parser.add_argument("--alpha_reset_epoch", type=int, default=-1, help="When to reset alpha parameter in AutoSparse")
    parser.add_argument("--label_smoothing", type=float, default=0, help="Label smoothing value for cross entropy loss")
    parser.add_argument("--g", type=str, default="sigmoid", help="Which function to use for thresholds.")
    parser.add_argument("--threshold_decay", type=float, help="Decay value for learnable threshold")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr_schedule", type=str, default=None, help="Learning rate schedule policy")
    parser.add_argument("--milestones", type=int, nargs="+", help="When to drop learning rate in multistep scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="How much to drop learning rate in multistep scheduler")
    parser.add_argument("--warmup_length", type=int, default=0, help="Warmup length for cosine annealing scheduler")
    parser.add_argument("--cosine_tmax", type=int, default=200, help="T_max value for CosineAnnealingLR. How many epochs does it take for learning rate to go to 0 and back.")
    return parser