def get_str_config():
    return {"batch_size":256, "epochs":100, "threshold_init":-100, 
            "l2_decay": 0.000030517578125, "threshold_type":"channelwise", 
            "optimizer":"SGD", "momentum":0.9, "model":"ResNet50"}

def get_autosparse_config():
    """Learning rate (max) 0.256 using a cosine annealing with warm up of 5 epochs."""
    return {"alpha": 0.5, "lr":0.1, "threshold_init": -5, "l2_decay": 0.000030517578125, 
            "threshold_type": "channelwise", "momentum":0.875, "optimizer":"SGD", "batch_size":256, 
            "epochs":100, "alpha_decay":"sigmoid_cosine_decay", "model":"ResNet50"}

def get_dst_config():
    """LR decay at epoch 80 0.1 -> 0.01. At epoch 120, 0.01 -> 0.001."""
    return {"alpha": 5e-6, "lr": 0.1, "momentum":0.9, "optimizer":"SGD", "batch_size": 64, 
            "epochs": 160, "threshold_type": "channelwise", "model":"VGG16", "max_pruning_percentage": 0.99}