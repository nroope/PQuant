def get_str_config():
    return {"batch_size":256, "epochs":100, "threshold_init":-100, 
            "l2_decay": 0.000030517578125, "threshold_type":"channelwise", 
            "optimizer":"SGD", "momentum":0.9, "model":"ResNet50"}

def get_autosparse_config():
    """Learning rate (max) 0.256 using a cosine annealing with warm up of 5 epochs."""
    return {"alpha": 0.75, "lr":0.256, "threshold_init": -5, "l2_decay": 0.000030517578125, 
            "threshold_type": "channelwise", "momentum":0.875, "optimizer":"SGD", "batch_size":256, 
            "epochs":100, "alpha_decay":"sigmoid_cosine_decay", "model":"ResNet50", "alpha_reset_epoch":90, 
            "label_smoothing":0.1, "lr_schedule": "cosine", "warmup_epochs":5, "g":"sigmoid"}

def get_dst_config():
    """LR decay at epoch 80 0.1 -> 0.01. At epoch 120, 0.01 -> 0.001."""
    return {"alpha": 5e-6, "lr": 0.1, "momentum":0.9, "optimizer":"SGD", "batch_size": 64, 
            "epochs": 160, "threshold_type": "channelwise", "model":"VGG16", "max_pruning_percentage": 0.99, "l2_decay":0., 
            "lr_schedule":"multistep", "gamma":0.1, "milestones":[80,120]}

def get_pdp_config():
    """LR decay at epoch 80 0.1 -> 0.01. At epoch 120, 0.01 -> 0.001."""
    return {"lr": 0.256, "momentum":0.9, "optimizer":"SGD", "batch_size": 256, 
            "epochs": 100, "l2_decay":1e-4, "epsilon":0.015, "pretraining_epochs": 16, "sparsity": 0.8, "temperature": 1e-5, "lr_schedule": "cosine"}

def get_cs_config():
    return {"optimizer":"sgd", "lr":0.1, "momentum":0.9, "batch_size":128, "epochs":85,"l2_decay":1e-4, "rounds":5, "lr_schedule":"multistep", "milestones":[56,71], "gamma":0.1, "fine_tune": True, 
            "threshold_init": -0.3, "save_weights_epoch":2, "final_temp":200, "beta":1., "threshold_decay":0, "rewind":"post-ticket-search"}