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
    encoded_text = model.text_enc(caption)

    return encoded_text


def inference_on_image(image,
                       storages,
                       model_type):
    model = storages['models_storage'][model_type]
    image_model = storages['image_models_storage'][model_type]
    ocr_model = storages['ocr_models_storage'][model_type]
    encoded_image = calculate_image_embedding(image,
                                              model,
                                              image_model,
                                              ocr_model,
                                              model_type)
    captions_embeddings = load_caption_embeddings_from_storage(model_type)
    nearest_caption = find_nearest_caption(encoded_image, captions_embeddings)

    return nearest_caption


def inference_on_caption(caption,
                         storages,
                         model_type):
    caption_model = storages['models_storage'][model_type]
    encoded_image = calculate_caption_embedding(caption,
                                                caption_model)
    images_embeddings = load_image_embeddings_from_storage(model_type)
    nearest_caption = find_nearest_image(encoded_image, images_embeddings)

    return nearest_caption


def load_image_embeddings_from_storage():
    pass


def load_caption_embeddings_from_storage():
    pass


def find_nearest_caption():
    pass


def find_nearest_image():
    pass
