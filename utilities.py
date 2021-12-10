import torch
import os
import shutil  # todo переделать чекпоинт в адекватный формат
import json
import yaml
from pathlib import Path


def save_to_json(filename, data):
    with open(filename, 'w') as f:
        return json.dump(data, f)


def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_checkpoint(state_dict, is_best, dir_path, filename='checkpoint.pth'):
    dir_path = Path(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)    
    torch.save(state_dict, dir_path / filename)
    if is_best:
        shutil.copyfile(dir_path / filename, dir_path / 'best_model.pth')


def get_config(config_path):
    with open(config_path, 'r') as params:
        return yaml.safe_load(params)
