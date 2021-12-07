import torch


models_for_startup = {
    'clip': {
        'model_name': 'ViT-B/32',
        'device': torch.device("cpu"),
    },
    'yolov5': {
        'model_name': 'yolo_best.pt',
        'device': torch.device("cpu"),
    },
    'vsrn': {
        'model_name': 'runs/log/model_best.pth.tar',
        'device': torch.device("cpu"),
        'vocab_path': 'checkpoints_and_vocabs/f30k_precomp_vocab.pkl',
        'params_config_path': 'inference_config.yaml',
    },
}


embeddings_storage_config = {}  # todo add config here
