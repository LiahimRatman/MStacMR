import clip
import torch

from inference_yolov5 import inference_yolo_on_one_image
from inference_clip import inference_clip_one_image


def init_models_storages():
    # todo здесь нужно прописать инит. Это хранилище будет интиться при старте апихи. + Желательно сделать предзагрузку эмбеддингов
    models_storage = {
        'yolo_plus_clip': 'runs/log/model_best.pth.tar'
    }
    storages = {
        'models_storage': models_storage,
        'image_models_storage': None,
        'ocr_models_storage': None,
    }

    return storages


def calculate_image_embedding(image,
                              model,
                              image_model,
                              ocr_model=None,
                              model_type='yolo_plus_clip'):
    if model_type == 'yolo_plus_clip':
        image_embeddings = image_model(image)
        ocr_embeddings = ocr_model(image)
        encoded_image = model.img_enc(image_embeddings, ocr_embeddings)
        return encoded_image
    else:
        pass


def calculate_caption_embedding(caption,
                                model):
    encoded_text = model.txt_enc(caption)  # todo Проверить, что заведется

    return encoded_text


def inference_on_image(image_path,
                       storages,
                       model_type,
                       save_emb=False):
    # model = storages['models_storage'][model_type]
    # image_model = storages['image_models_storage'][model_type]
    # ocr_model = storages['ocr_models_storage'][model_type]
    model_clip, preprocess = clip.load("ViT-B/32")  # todo Переделать загрузку
    detected_regions = inference_yolo_on_one_image(image_path, 'yolo_best.pt')
    region_embeddings = inference_clip_one_image(image_path,
                                   detected_regions,
                                   model_clip,
                                   preprocess,
                                   torch.device("cpu"))

    # todo Сделать ocr
    # ocr_embeddings = inference_ocr()
    #
    # full_image_embedding = model(region_embeddings, ocr_embeddings)
    # captions_embeddings = load_caption_embeddings_from_storage(model_type)  # todo Сделать загрузку из storage
    # nearest_caption = find_nearest_caption(full_image_embedding, captions_embeddings)
    #
    # if save_emb:
    #     save_image_embedding()  # todo Сделать сохранение эмбеддингов
    #
    # return nearest_caption


def inference_on_caption(caption,
                         storages,
                         model_type):
    caption_model = storages['models_storage'][model_type]
    encoded_caption = calculate_caption_embedding(caption,
                                                  caption_model)
    images_embeddings = load_image_embeddings_from_storage(model_type)
    nearest_image = find_nearest_image(encoded_caption, images_embeddings)

    return nearest_image


# def inference_generate_caption(image_path,
#                                storages,
#                                model_type,
#                                save_emb=False):
#     model = storages['models_storage'][model_type]
#     # image_model = storages['image_models_storage'][model_type]
#     # ocr_model = storages['ocr_models_storage'][model_type]
#     model_clip, preprocess = clip.load("ViT-B/32")  # todo Переделать загрузку
#     detected_regions = inference_yolo_on_one_image(image_path, 'yolov5/yolo_best.pt')
#     region_embeddings = inference_clip_one_image(image_path,
#                                                  detected_regions,
#                                                  model_clip,
#                                                  preprocess,
#                                                  torch.device("cpu"))
#
#     # todo Сделать ocr
#     ocr_embeddings = inference_ocr()
#
#     full_image_embedding = model(region_embeddings, ocr_embeddings)  # todo не совсем так
#
#     return generate_caption(full_image_embedding)


def load_image_embeddings_from_storage():
    pass


def load_caption_embeddings_from_storage():
    pass


def find_nearest_caption():
    pass


def find_nearest_image():
    pass


print(inference_on_image('STACMR_train/CTC/images/COCO_train2014_000000000036.jpg', 'yolo_best.pt', None).shape)
