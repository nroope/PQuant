from parser import parse_cmdline_args

import torch
from main import get_model_data_loss_func, validate_resnet, validate_smartpixel

from pquant.core.compressed_layers import (
    add_quantized_activations_to_model,
    get_layer_keep_ratio,
)


def validate(config, device):
    model, _, val_loader, _ = get_model_data_loss_func(config, device)
    model = add_quantized_activations_to_model(model, config)
    model.to("cuda")
    model.load_state_dict(torch.load(f"{config.validation_config_folder}/final_model.pt"))
    get_layer_keep_ratio(model)

    if config["model"] == "smartpixel":
        validate_smartpixel(model, val_loader, device, config["training_parameters"]["epochs"], writer=None)
    else:
        validate_resnet(model, val_loader, device, None, 0, None)


if __name__ == "__main__":
    config = parse_cmdline_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validate(config, device)
