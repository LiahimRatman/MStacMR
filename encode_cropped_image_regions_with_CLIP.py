import clip
import torchvision
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from dao import load_from_json

CLIP_EMBEDDINGS_SIZE = 512
MAX_DETECTED_REGIONS = 36


def prepare_clip_dataset_embeddings(annotations_map_path='STACMR_train/full_dataset_train_mapa_good.json',
                                    output_embeddings_path='precomputed_embeddings/final_all_train_emb_CLIP_.npy',
                                    image_encoder_model_name="ViT-B/32",
                                    ):
    model, preprocess = clip.load(image_encoder_model_name)
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    mapa_full = load_from_json(annotations_map_path)

    results = []
    for item in tqdm(mapa_full):
        im_path_base = item['image_path']
        if 'CTC' in im_path_base:
            detected_labels_path = 'STACMR_train/labels/CTC/detections/'
        elif 'flickr' in im_path_base:
            detected_labels_path = 'STACMR_train/labels/flickr30k/detections/'
        else:
            detected_labels_path = 'STACMR_train/labels/text_caps/detections/'
        try:  # todo refactor paths
            crops_labels_name = im_path_base.split('/')[-1].split('.')[0] + '.txt'
            with open(detected_labels_path + crops_labels_name, 'r') as f:
                text = f.readlines()
            crops = [{'class_id': int(crop[0]),
                      'x': float(crop[1]),
                      'y': float(crop[2]),
                      'w': float(crop[3]),
                      'h': float(crop[4]),
                      } for crop in [item.strip().split() for item in text]]
            im_path = 'STACMR_train/' + im_path_base
            # if 'text_caps' in im_path:  # todo check
            #     im_path = im_path.replace('images/', 'images/train_images/')

            images = []
            for crop in crops:
                imm = Image.open(im_path)
                im = torchvision.transforms.functional.crop(imm,
                                                            round(imm.size[1] * (crop['y'] - crop['h'] / 2)),
                                                            round(imm.size[0] * (crop['x'] - crop['w'] / 2)),
                                                            round(imm.size[1] * crop['h']),
                                                            round(imm.size[0] * crop['w']))
                images.append(im)
            images = [preprocess(image) for image in images]
            image_input = torch.tensor(np.stack(images)).cuda()
            with torch.no_grad():
                image_features = model.encode_image(image_input).float()
        except:
            image_features = []
        new_data2 = []
        for _ in range(MAX_DETECTED_REGIONS):
            if _ < len(image_features):
                new_data2.append(image_features[_])
            else:
                new_data2.append(torch.zeros(CLIP_EMBEDDINGS_SIZE))
        results.append(np.stack([item.cpu() for item in new_data2], axis=0))

        np.save(output_embeddings_path, np.stack([item for item in results], axis=0))


# todo we need to make inference server with model pre-load
# todo change structure
def inference_clip_one_image(image_path,
                             image_labels,
                             image_encoder_model_name):
    pass


def inference_clip_batch(images_batch,
                         images_labels,
                         image_encoder_model_name):
    pass
