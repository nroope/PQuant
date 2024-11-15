from train import validation
from weaver.utils.nn.tools import evaluate_classification
from parser import parse_cmdline_args
from train import get_model_data_loss_func
from datetime import datetime
import torch


def validate(config, device):
    model, _, val_loader, _ = get_model_data_loss_func(config, device)
    start = datetime.now()
    if config.model == "parT":
        evaluate_classification(model, val_loader, device, 0, False, None, 1000, tb_helper=None)
    else:
        validation(model, val_loader, device, None, 0, None)
    end = datetime.now()
    print("Validation time:", end-start)


if __name__ == "__main__":
    config = parse_cmdline_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validate(config, device)