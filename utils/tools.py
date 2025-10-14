import torch
import gc
import json
import os

global configs_dir, training_configs_dir
configs_dir = "../configs"
training_configs_dir = "../configs/training_configs"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeUpMemories(model=None):
    try:
        del model
    except Exception as e:
        print("error when deleting model :",e)

    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.set_per_process_memory_fraction(0.9)

def checkGPU(model=None):
    if model:
        print(next(model.parameters()).device)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

def get_optimizer(model, optimizer_type="adam", lr=1e-3, weight_decay=0.0, momentum=0.9, betas=(0.9, 0.999)):
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_type == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def get_emotion_by_label(label):
    emotion_dict = {
        0: "joy",
        1: "tenderness",
        2: "inspiration",
        3: "amusement",
        4: "anger",
        5: "disgust",
        6: "fear",
        7: "sadness",
        8: "arousal",
        9: "valence",
        10: "familiarity",
        11: "liking"
        }
    return emotion_dict.get(label, "unknown")

def save_loading_config(config_data, config_id=None):
    if config_id is None:
        config_files = [f for f in os.listdir(configs_dir) if f.endswith(".json")]
        config_id = len(config_files) + 1

    config_file = os.path.join(configs_dir, f"config_{config_id}.json")
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=4)

def load_loading_config(config_id):
    config_file = os.path.join(configs_dir, f"config_{config_id}.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
def save_new_loading_config():
    data = {}
    data['load_method'] =  "pre" # or 'lazy', 'normal'
    data['window_size'] =  1000 # 250 500 1000
    data['stride'] = 1000 # 250 500 1000
    data['use_pca'] = False

    data['use_de'] = True
    data['de_bands'] = [0,1,2,3,4] # [1,2,3,4] [2,3] [0,1,2,3,4]
    data['use_de_pca'] = True

    data['use_psd'] = True
    data['psd_bands'] = [2,3] # [0,1,2,3,4] [2,3]
    data['use_psd_pca'] = True

    data['apply_filter'] = True
    data['lowcut'] = 1.0
    data['highcut'] = 50.0
    data['fs_value'] = 250
    data['filter_order'] = 4
    data['apply_baseline'] = True
    data['baseline_window'] = 20
    data['normalize'] = True
    data['normalization_method'] = 'zscore' #'per_channel', 'zscore'

    save_loading_config(data)

def save_training_config(config_data, config_id=None):
    if config_id is None:
        config_files = [f for f in os.listdir(training_configs_dir) if f.endswith(".json")]
        config_id = len(config_files) + 1

    config_file = os.path.join(training_configs_dir, f"training_config_{config_id}.json")
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=4)

def load_training_config(config_id):
    config_file = os.path.join(training_configs_dir, f"training_config_{config_id}.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        return config
    else:
        raise FileNotFoundError(f"training Config file not found: {config_file}")

def save_new_training_config():
    data = {}
    data['random_seed'] = 42
    data['apply_train_test_split'] = True
    data['test_size'] = 0.2 # 0.3
    data['epochs'] = 20 # 50 100
    data['batch_size'] = 64 # 32 64 16 8
    data['num_classes'] = 2 # 8
    data['sigmoid_threshold'] = 0.5 # 0.45 0.55 0.4 0.3

    data['optimizer_type'] = "adam"  # "adam", 'sgd', 'rmsprop', 'adamw', 'adagrad'
    data['learning_rate'] = 1e-3 # 1e-3 1e-4 5e-4 5e-3
    data['learning_rate_to_save'] = "1e-3" # "1e-3" "1e-4" "5e-4" "5e-3"
    data['weight_decay'] = 0.0 # 0.0 1e-4 1e-3
    data['momentum'] = 0.9 # 0.8 0.9 0.99
    data['betas'] = (0.9, 0.999) # (0.9, 0.999) (0.8, 0.999) (0.9, 0.998)

    save_training_config(data)
