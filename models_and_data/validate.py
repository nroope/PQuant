from weaver.utils.nn.tools import evaluate_classification
from parser import parse_cmdline_args
from main import get_model_data_loss_func, validate_resnet, validate_smartpixel
from datetime import datetime
import torch
from pquant.core.compressed_layers import get_layer_keep_ratio, add_quantized_activations_to_model

def validate(config, device):
    model, _, val_loader, _ = get_model_data_loss_func(config, device)
    model = add_quantized_activations_to_model(model, config)
    model.to("cuda")
    model.load_state_dict(torch.load(f"{config.validation_config_folder}/final_model.pt"))
    get_layer_keep_ratio(model)

    start = datetime.now()
    if config["model"] == "particle_transformer":
        evaluate_classification(model, val_loader, device, 0, False, None, 10, tb_helper=None)
    elif config["model"] == "smartpixel":
        validate_smartpixel(model, val_loader, device, config["training_parameters"]["epochs"], writer=None)
    else:
        validate_resnet(model, val_loader, device, None, 0, None)
    end = datetime.now()
    print("Validation time:", end-start)


if __name__ == "__main__":
    config = parse_cmdline_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validate(config, device)