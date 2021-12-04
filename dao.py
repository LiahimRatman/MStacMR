import torch
import shutil  # todo переделать чекпоинт в адекватный формат
import json


def save_to_json(filename, data):
    with open(filename, 'w') as f:
        return json.dump(data, f)


def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
