import os
import pickle
import time

# import torch
import numpy as np

# from tqdm import tqdm
from tqdm.auto import tqdm

from utilities import save_checkpoint, get_config
from data_utils import get_dataloader_img_ocr_precalculated
from train_utils import adjust_learning_rate
from evaluate import evaluate
from Vocabulary import Vocabulary
# from VSRN import VSRN

from model import create_model_from_config

INFERENCE_CONFIG_PATH = 'config/full_config_nested.yaml'
MODEL_CONFIG_PATH = 'config/full_config_nested.yaml'
CHECKPOINTS_DIR_PATH = 'checkpoints/'


def make_checkpoint(model, current_epoch, current_score, best_score, dir_path):
    if current_score > best_score:
        best_score = current_score
        save_checkpoint(
            {
                'epoch': current_epoch,
                'model': model.state_dict(),
                'score': current_score,
            },
            is_best=True,
            dir_path=dir_path,
            filename=f'{current_epoch}_checkpoint.pth',
        )
        return current_score
    return best_score


def train_epoch(
        model,
        train_loader, val_loader,
        similarity_measure,
        current_epoch, best_score,
        checkpoint_every_n_batches,
        dir_path,
):
    n_batches = int(np.ceil(len(train_loader.dataset) / train_loader.batch_size))

    for i, train_data in tqdm(enumerate(train_loader), total=n_batches):
        model.train()

        # for el in train_data:
        #     print(type(el))

        model.make_train_step(*train_data)

        # full evaluation at every n-th batch
        if i % checkpoint_every_n_batches == 0 and i > 0:
            current_score = evaluate(model, val_loader, similarity_measure)
            best_score = make_checkpoint(model, current_epoch, current_score, best_score, dir_path)

    return best_score


def run_train_loop(config, dir_path = None):
    model = create_model_from_config(config)

    if dir_path is None:
        dir_path = config['training_params'].get('checkpoints_saving_path', '')
    if not os.path.exists(dir_path):
        raise ValueError(f"path for saving checkpoints is not provided or invalid: {dir_path}")
    else:
        print(f'will save checkpoint to {dir_path}')

    vocab = pickle.load(open(config['training_params']['vocab_path'], 'rb'))

    train_loader = get_dataloader_img_ocr_precalculated(
        annotation_map_path=config['training_params']['train_annot_map_path'],
        image_embeddings_path=config['training_params']['train_img_emb_path'],
        ocr_embeddings_path=config['training_params']['train_ocr_emb_path'],
        encoder_tokenizer_path=config['caption_encoder_params']['encoder_path'],
        vocab=vocab,
        shuffle=True,
        batch_size=config['training_params']['batch_size'],
        num_image_boxes=config['image_encoder_params']['num_image_boxes'],
    )

    val_loader = get_dataloader_img_ocr_precalculated(
        annotation_map_path=config['training_params']['eval_annot_map_path'],
        image_embeddings_path=config['training_params']['eval_img_emb_path'],
        ocr_embeddings_path=config['training_params']['eval_ocr_emb_path'],
        encoder_tokenizer_path=config['caption_encoder_params']['encoder_path'],
        vocab=vocab,
        shuffle=False,
        batch_size=config['training_params']['batch_size'],
        num_image_boxes=config['image_encoder_params']['num_image_boxes'],
    )

    similarity_measure = config['loss_params']['measure']
    num_epochs = config['training_params']['num_epochs']

    best_score = -np.inf

    for epoch in tqdm(range(1, num_epochs + 1)):
        if config.get('training_params', {}).get('lr_update'):
            adjust_learning_rate(
                config['training_params']['learning_rate'],
                config['training_params']['lr_update'],
                model.optimizer,
                epoch - 1,
            )
        # train for one epoch
        n_batches = int(np.ceil(len(train_loader.dataset) / train_loader.batch_size))
        best_score = train_epoch(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            similarity_measure=similarity_measure,
            current_epoch=epoch,
            best_score=best_score,
            checkpoint_every_n_batches=n_batches - 1,
            dir_path=dir_path,
        )

    return model


if __name__ == '__main__':
    config = get_config(MODEL_CONFIG_PATH)
    run_train_loop(config, CHECKPOINTS_DIR_PATH)
