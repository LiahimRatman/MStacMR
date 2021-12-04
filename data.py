import numpy as np
import torch

from dao import load_from_json
from data_utils import prepare_captions, collate_fn


class TrainDataset:
    def __init__(self,
                 images_path=None,
                 annotation_map_path=None,
                 use_precomputed_embeddings=True,
                 image_embeddings_path=None,
                 ocr_embeddings_path=None,
                 vocab=None,
                 im_div=5,
                 max_text_regions=16,
                 max_caption_len=100):
        self.annotation_map_path = annotation_map_path
        self.images_path = images_path
        self.vocab = vocab
        self.image_embeddings_path = image_embeddings_path
        self.ocr_embeddings_path = ocr_embeddings_path
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.im_div = im_div  # as we have 5 captions for each image
        self.max_text_regions = max_text_regions
        self.max_caption_len = max_caption_len

        self.train_map = load_from_json(self.annotation_map_path)
        if self.use_precomputed_embeddings:
            self.precomputed_image_embeddings = np.load(self.image_embeddings_path)
            self.precomputed_ocr_embeddings = np.load(self.ocr_embeddings_path)

    def __getitem__(self, index):
        item_id = index // self.im_div
        caption_number = index % self.im_div
        item = self.train_map[item_id]
        caption = item['captions'][caption_number]
        # print(item_id, caption_number, caption)
        target, caption_label, caption_mask = prepare_captions(caption=caption,
                                                               vocab=self.vocab,
                                                               max_len=self.max_caption_len)
        # print(target, caption_label, caption_mask)

        if self.use_precomputed_embeddings:
            image_regions_embedding = self.precomputed_image_embeddings[item_id]
            ocr_embedding = self.precomputed_ocr_embeddings[item_id]

            # ограничим макисмальное количество элементов текста сцены. Можно сделать аналогично и для картинок при желании
            ocr_embedding = ocr_embedding[:self.max_text_regions][:]  # todo в чем смысл последнего двоеточия..

            ocr_embedding = torch.Tensor(ocr_embedding)
            image_regions_embedding = torch.Tensor(image_regions_embedding)

            return image_regions_embedding, target, index, item_id, caption_label, caption_mask, ocr_embedding
        else:
            image_path = item['image_path']
            #  todo design full train pipeline here. At this moment is not created
        pass

    def __len__(self):
        return len(self.train_map)


class EvalDataset:
    def __init__(self,
                 images_path=None,
                 annotation_map_path=None,
                 use_precomputed_embeddings=True,
                 image_embeddings_path=None,
                 ocr_embeddings_path=None):
        self.images_path = images_path
        self.image_embeddings_path = image_embeddings_path
        self.ocr_embeddings_path = ocr_embeddings_path
        self.annotation_map_path = annotation_map_path
        self.use_precomputed_embeddings = use_precomputed_embeddings

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class InferenceImagesEncodeDataset:
    def __init__(self, inference_images_path):
        self.inference_images_path = inference_images_path

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class InferenceCaptionsEncodeDataset:
    def __init__(self, inference_texts_path):
        self.inference_texts_path = inference_texts_path

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def get_dataloader(type, #train, eval, inference_images, inference_captions
                   annotations_map_name,
                   image_embeddings_name,
                   ocr_embeddings_name,
                   images_path,
                   # data_name,
                   # split,
                   # root,
                   # json,
                   vocab,
                   # transform,
                   batch_size=100,
                   shuffle=True,
                   # ids=None,
                   num_workers=0):
    # if 'f8k' in data_name or 'f30k' in data_name:
    #     dataset = FlickrDataset(root=root,
    #                             split=split,
    #                             json=json,
    #                             vocab=vocab,
    #                             transform=transform)
    if type == 'train':
        print(type)
        dataset = TrainDataset(use_precomputed_embeddings=True,
                               annotation_map_path=annotations_map_name,
                               images_path=images_path,  # todo
                               image_embeddings_path=image_embeddings_name,
                               ocr_embeddings_path=ocr_embeddings_name,
                               vocab=vocab)
    else:
        dataset = None
        pass

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader
