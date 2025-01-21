from train import validate_resnet, validate_smartpixel
from weaver.utils.nn.tools import evaluate_classification
from parser import parse_cmdline_args
from train import get_model_data_loss_func
from datetime import datetime
import torch
from sparse_layers import get_layer_keep_ratio, get_layer_weight_uniques

def validate(config, device):
    model, _, val_loader, _ = get_model_data_loss_func(config, device)
    model.to("cuda")
    model.load_state_dict(torch.load(f"{config.validation_config_folder}/final_model.pt"))
    get_layer_keep_ratio(model)
    get_layer_weight_uniques(model)

    start = datetime.now()
    if config.model == "particle_transformer":
        evaluate_classification(model, val_loader, device, 0, False, None, 10, tb_helper=None)
    elif config.model == "smartpixel":
        validate_smartpixel(model, val_loader, device, config.epochs, writer=None)
    else:
        validate_resnet(model, val_loader, device, None, 0, None)
    end = datetime.now()
    print("Validation time:", end-start)


if __name__ == "__main__":
    config = parse_cmdline_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validate(config, device)