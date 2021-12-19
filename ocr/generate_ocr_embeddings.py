import argparse
import os
import pickle

import numpy as np
import tensorflow_hub as tf_hub

from ocr.ocr_embedder import OCREmbedder
from ocr.ocr_pipeline import get_sentences_from_images
from ocr.ocr_recognition import Pipeline, Recognizer
from utilities import get_config, load_from_json

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", default='full_config_nested.yaml', help="yaml config path")
parser.add_argument("--save_path", default='../data/precomputed_embeddings/', help="path to save predictions")

MODEL_CONFIG_PATH = 'full_config_nested.yaml'


def generate_embeddings(config):
    path_to_muse = config['training_params']['muse_path']
    max_ocr_captions = int(config['image_encoder_params']['muse_embedder_max_ocr_captions'])
    train_mapa_path = config['training_params']['train_annot_map_path']
    raw_image_folder = config['training_params']['raw_images_path']
    save_path = config['training_params']['train_ocr_emb_path']

    train_mapa = load_from_json(train_mapa_path)
    image_paths = [os.path.join(raw_image_folder, m["image_path"]) for m in train_mapa]

    # Get text tokens
    pipeline = Pipeline(recognizer=Recognizer())
    batch_size = 256
    for i in range(240):
        if i * batch_size < 59419:
            cur_images = image_paths[i * batch_size:(i + 1) * batch_size]
            print(f"Start: {i * batch_size, (i + 1) * batch_size}")
            sentences = get_sentences_from_images(pipeline, cur_images, device='gpu')
            with open(
                    f"../data/ocr_sentences/sentences_{i * batch_size}_{(i + 1) * batch_size}.p",
                    'wb') as fp:
                pickle.dump(sentences, fp)

    # Get muse embeddings for text tokens
    embedder = tf_hub.load(path_to_muse)
    ocr_embedder = OCREmbedder(embedder, max_ocr_captions=max_ocr_captions)
    result = np.zeros((59419, 512))
    batch_size = 256
    for i in range(120):
        if i * batch_size <= 30_000:
            print(f"Start: {i * batch_size, (i + 1) * batch_size}")
            with open(
                    f"../data/ocr_sentences/sentences_{i * batch_size}_{(i + 1) * batch_size}.p",
                    'rb') as fp:
                sentences = pickle.load(fp)
            sentences = [" ".join(tokens) for tokens in sentences]
            emb_matrix = ocr_embedder.get_embeddings_for_gcn(sentences)
            result[i * batch_size:(i + 1) * batch_size, :] = emb_matrix
    np.save(save_path, result)


if __name__ == '__main__':
    config = get_config(MODEL_CONFIG_PATH)
    generate_embeddings(config)
