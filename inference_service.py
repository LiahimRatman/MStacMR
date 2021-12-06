import clip
import numpy as np
import pickle
import torch

from VSRN import VSRN
from Vocabulary import Vocabulary
from dao import get_config
from inference_yolov5 import inference_yolo_on_one_image
from inference_clip import inference_clip_one_image
from prepare_image_regions_embeddings import CLIP_EMBEDDING_SIZE, MAX_DETECTIONS_PER_IMAGE


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


def inference_on_image(image_path,
                       storages,
                       model_type,
                       save_emb=False):
    # model_path = storages['models_storage'][model_type]
    model_path = 'runs/log/model_best.pth.tar'
    # image_model = storages['image_models_storage'][model_type]
    # ocr_model = storages['ocr_models_storage'][model_type]
    model_clip, preprocess = clip.load("ViT-B/32")  # todo Переделать загрузку
    detected_regions = inference_yolo_on_one_image(image_path, 'yolo_best.pt')
    region_embeddings = inference_clip_one_image(image_path,
                                                 detected_regions,
                                                 model_clip,
                                                 preprocess,
                                                 torch.device("cpu"))
    stacked_image_features = []
    for _ in range(MAX_DETECTIONS_PER_IMAGE):
        if _ < len(region_embeddings):
            stacked_image_features.append(region_embeddings[_])
        else:
            stacked_image_features.append(torch.zeros(CLIP_EMBEDDING_SIZE))
    region_embeddings = np.stack([item.cpu() for item in stacked_image_features], axis=0)

    # todo Сделать ocr
    # ocr_embeddings = inference_ocr()  # Тут должен быть массив размера 16 * 300
    ocr_embeddings = np.zeros_like(region_embeddings)[:16, :300]

    checkpoint = torch.load(model_path, map_location="cpu")

    vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
    params = get_config('inference_config.yaml')
    params['vocab_size'] = len(vocab)
    model = VSRN(params['grad_clip'],
                 params['image_embedding_dim'],
                 params['gcn_embedding_size'],
                 params['vocab_size'],
                 params['caption_encoder_word_dim'],
                 params['caption_encoder_num_layers'],
                 params['caption_encoder_embedding_size'],
                 params['dim_vid'],  # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
                 params['dim_caption_generation_hidden'],
                 params['input_dropout_p_caption_generation_enc'],
                 params['rnn_type_caption_generation_enc'],
                 params['rnn_dropout_p_caption_generation_enc'],
                 params['bidirectional_enc'],
                 params['max_caption_len'],
                 params['dim_word_caption_generation'],
                 params['input_dropout_p_caption_generation_dec'],
                 params['rnn_type_caption_generation_dec'],
                 params['rnn_dropout_p_caption_generation_dec'],
                 params['bidirectional_dec'],
                 params['margin'],
                 params['measure'],
                 params['max_violation'],
                 params['learning_rate'])

    model.load_state_dict(checkpoint['model'])
    model.val_start()
    region_embeddings = torch.tensor(region_embeddings).unsqueeze(0)
    ocr_embeddings = torch.tensor(ocr_embeddings).unsqueeze(0)
    if torch.cuda.is_available():
        region_embeddings = region_embeddings.cuda()
        ocr_embeddings = ocr_embeddings.cuda()

    # Forward
    with torch.no_grad():
        full_image_embedding, _ = model.img_enc(region_embeddings, ocr_embeddings)

    captions_embeddings = load_caption_embeddings_from_storage(model_type)  # todo Сделать загрузку из storage
    nearest_caption = find_nearest_caption(full_image_embedding, captions_embeddings)

    # if save_emb:
    #     save_image_embedding()  # todo Сделать сохранение эмбеддингов

    return nearest_caption


def inference_on_caption(caption,
                         storages,
                         model_type):
    # model_path = storages['models_storage'][model_type]
    model_path = 'runs/log/model_best.pth.tar'
    checkpoint = torch.load(model_path,
                            map_location="cpu")

    vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
    params = get_config('inference_config.yaml')
    params['vocab_size'] = len(vocab)
    model = VSRN(params['grad_clip'],
                 params['image_embedding_dim'],
                 params['gcn_embedding_size'],
                 params['vocab_size'],
                 params['caption_encoder_word_dim'],
                 params['caption_encoder_num_layers'],
                 params['caption_encoder_embedding_size'],
                 params['dim_vid'],  # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
                 params['dim_caption_generation_hidden'],
                 params['input_dropout_p_caption_generation_enc'],
                 params['rnn_type_caption_generation_enc'],
                 params['rnn_dropout_p_caption_generation_enc'],
                 params['bidirectional_enc'],
                 params['max_caption_len'],
                 params['dim_word_caption_generation'],
                 params['input_dropout_p_caption_generation_dec'],
                 params['rnn_type_caption_generation_dec'],
                 params['rnn_dropout_p_caption_generation_dec'],
                 params['bidirectional_dec'],
                 params['margin'],
                 params['measure'],
                 params['max_violation'],
                 params['learning_rate'])

    model.load_state_dict(checkpoint['model'])
    model.val_start()
    # caption_model = storages['models_storage'][model_type]
    with torch.no_grad():
        encoded_caption = model.txt_enc(caption)

    images_embeddings = load_image_embeddings_from_storage(model_type)
    nearest_image = find_nearest_image(encoded_caption, images_embeddings)

    return nearest_image


def inference_generate_caption(image_path,
                               storages,
                               model_type,
                               save_emb=False):
    # model = storages['models_storage'][model_type]
    # image_model = storages['image_models_storage'][model_type]
    # ocr_model = storages['ocr_models_storage'][model_type]

    # model_path = storages['models_storage'][model_type]
    model_path = 'runs/log/model_best.pth.tar'
    checkpoint = torch.load(model_path,
                            map_location="cpu")

    vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
    params = get_config('inference_config.yaml')
    params['vocab_size'] = len(vocab)
    model = VSRN(params['grad_clip'],
                 params['image_embedding_dim'],
                 params['gcn_embedding_size'],
                 params['vocab_size'],
                 params['caption_encoder_word_dim'],
                 params['caption_encoder_num_layers'],
                 params['caption_encoder_embedding_size'],
                 params['dim_vid'],  # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
                 params['dim_caption_generation_hidden'],
                 params['input_dropout_p_caption_generation_enc'],
                 params['rnn_type_caption_generation_enc'],
                 params['rnn_dropout_p_caption_generation_enc'],
                 params['bidirectional_enc'],
                 params['max_caption_len'],
                 params['dim_word_caption_generation'],
                 params['input_dropout_p_caption_generation_dec'],
                 params['rnn_type_caption_generation_dec'],
                 params['rnn_dropout_p_caption_generation_dec'],
                 params['bidirectional_dec'],
                 params['margin'],
                 params['measure'],
                 params['max_violation'],
                 params['learning_rate'])

    model_clip, preprocess = clip.load("ViT-B/32")  # todo Переделать загрузку
    detected_regions = inference_yolo_on_one_image(image_path, 'yolo_best.pt')
    region_embeddings = inference_clip_one_image(image_path,
                                                 detected_regions,
                                                 model_clip,
                                                 preprocess,
                                                 torch.device("cpu"))
    stacked_image_features = []
    for _ in range(MAX_DETECTIONS_PER_IMAGE):
        if _ < len(region_embeddings):
            stacked_image_features.append(region_embeddings[_])
        else:
            stacked_image_features.append(torch.zeros(CLIP_EMBEDDING_SIZE))
    region_embeddings = np.stack([item.cpu() for item in stacked_image_features], axis=0)

    # todo Сделать ocr
    # ocr_embeddings = inference_ocr()
    ocr_embeddings = np.zeros_like(region_embeddings)[:16, :300]

    model.load_state_dict(checkpoint['model'])
    model.val_start()
    region_embeddings = torch.tensor(region_embeddings).unsqueeze(0)
    ocr_embeddings = torch.tensor(ocr_embeddings).unsqueeze(0)
    if torch.cuda.is_available():
        region_embeddings = region_embeddings.cuda()
        ocr_embeddings = ocr_embeddings.cuda()

    # Forward
    with torch.no_grad():
        img_emb, GCN_img_emd = model.img_enc(region_embeddings, ocr_embeddings)

    seq_logprobs, seq_preds = model.caption_model(GCN_img_emd, None, 'inference')
    # (region_embeddings, ocr_embeddings)  # todo попробовать от общего эмбеддинга

    sentence = []
    for letter in seq_preds[0]:
        sentence.append(vocab.idx2word[letter.item()])

    return ' '.join(sentence)


def load_image_embeddings_from_storage():
    pass


def load_caption_embeddings_from_storage():
    pass


def find_nearest_caption():
    pass


def find_nearest_image():
    pass


# print(inference_on_image('STACMR_train/CTC/images/COCO_train2014_000000000036.jpg', 'yolo_best.pt', None))
print(inference_generate_caption('STACMR_train/CTC/images/COCO_train2014_000000000036.jpg', None, None, None))

