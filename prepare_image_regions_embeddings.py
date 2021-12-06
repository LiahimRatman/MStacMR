import clip
import numpy as np
import torch

from dao import load_from_json
from detect_datasets_regions_with_yolov5 import detect_image_regions_on_train_datasets
from inference_clip import inference_clip_one_image


MAX_DETECTIONS_PER_IMAGE = 36
CLIP_EMBEDDING_SIZE = 512


def get_datasets_embeddings(model_clip,
                            preprocess_clip,
                            annotations_map='checkpoints_and_vocabs/full_dataset_train_mapa_good.json',
                            save_emb=True,
                            save_path='precomputed_embeddings/train_embeddings_yolov5_clip.npy'):
    detect_image_regions_on_train_datasets()

    dataset_map = load_from_json(annotations_map)
    full_dataset_image_embeddings = []
    for item in dataset_map:
        dataset_name = item['image_path'].split('/')[0]
        image_features = inference_clip_one_image(image_path='STACMR_train/' + item['image_path'],
                                                  crop_labels=None,
                                                  model=model_clip,
                                                  preprocess=preprocess_clip,
                                                  device=torch.device("cpu"),
                                                  from_txt=True,
                                                  labels_path='STACMR_train/' + dataset_name + '/detections/labels/')

        stacked_image_features = []
        for _ in range(MAX_DETECTIONS_PER_IMAGE):
            if _ < len(image_features):
                stacked_image_features.append(image_features[_])
            else:
                stacked_image_features.append(torch.zeros(CLIP_EMBEDDING_SIZE))
        full_dataset_image_embeddings.append(np.stack([item.cpu() for item in stacked_image_features], axis=0))

    if save_emb:
        np.save(save_path, np.stack([item for item in full_dataset_image_embeddings], axis=0))

    return np.stack([item for item in full_dataset_image_embeddings], axis=0)


if __name__ == "__main__":
    # todo Сюда можно вставить argparse
    model, preprocess = clip.load("ViT-B/32")
    get_datasets_embeddings(model_clip=model,
                            preprocess_clip=preprocess)
