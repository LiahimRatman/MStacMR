import clip
import nltk
import numpy as np
import pickle
import torch

from VSRN import VSRN
from Vocabulary import Vocabulary
from dao import get_config
from inference_yolov5 import inference_yolo_on_one_image
from inference_clip import inference_clip_one_image
from prepare_image_regions_embeddings import CLIP_EMBEDDING_SIZE, MAX_DETECTIONS_PER_IMAGE
from models.common import DetectMultiBackend


def init_preload_model_storage(model_names_list):
    storage = {}
    for model_type, item in model_names_list.items():
        if model_type == 'clip':
            storage[model_type] = {}
            clip_model, clip_preprocess = clip.load(item['model_name'])
            storage[model_type]['model'] = clip_model.to(item['device']).eval()
            storage[model_type]['preprocess'] = clip_preprocess
        elif model_type == 'yolov5':
            storage[model_type] = {}
            yolov5_model = DetectMultiBackend(item['model_name'], device=item['device'], dnn=False)
            storage[model_type]['model'] = yolov5_model
        elif model_type == 'vsrn':
            storage[model_type] = {}
            checkpoint = torch.load(item['model_name'], map_location="cpu")
            vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
            params = get_config('inference_config.yaml')
            params['vocab_size'] = len(vocab)
            vsrn_model = VSRN(params['grad_clip'],
                              params['image_embedding_dim'],
                              params['gcn_embedding_size'],
                              params['vocab_size'],
                              params['caption_encoder_word_dim'],
                              params['caption_encoder_num_layers'],
                              params['caption_encoder_embedding_size'],
                              params['dim_vid'],
                              # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
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

            vsrn_model.load_state_dict(checkpoint['model'])
            vsrn_model.val_start()

            storage[model_type]['model'] = vsrn_model
            storage[model_type]['vocab'] = vocab
            storage[model_type]['params'] = params

    return storage


def inference_on_image(image_path,
                       storage,
                       model_type,
                       save_emb=False):
    model_clip, preprocess_clip = storage['clip']['model'], storage['clip']['preprocess']
    model_yolov5 = storage['yolov5']['model']
    vsrn_model = storage['vsrn']['model']
    # vsrn_vocab = storage['vsrn']['vocab']
    # vsrn_params = storage['vsrn']['params']
    # model_path = storages['models_storage'][model_type]
    # model_path = 'runs/log/model_best.pth.tar'
    # image_model = storages['image_models_storage'][model_type]
    # ocr_model = storages['ocr_models_storage'][model_type]
    # model_clip, preprocess = clip.load("ViT-B/32")
    # detected_regions = inference_yolo_on_one_image(image_path, 'yolo_best.pt')
    detected_regions = inference_yolo_on_one_image(image_path, model_yolov5, torch.device("cpu"))
    region_embeddings = inference_clip_one_image(image_path,
                                                 detected_regions,
                                                 model_clip,
                                                 preprocess_clip,
                                                 torch.device("cpu"))  # todo Тут могут быть проблемы из-за захардкоженного девайса
    stacked_image_features = []
    for _ in range(MAX_DETECTIONS_PER_IMAGE):
        if _ < len(region_embeddings):
            stacked_image_features.append(region_embeddings[_])
        else:
            stacked_image_features.append(torch.zeros(CLIP_EMBEDDING_SIZE))
    region_embeddings = np.stack([item.cpu() for item in stacked_image_features], axis=0)

    ocr_embeddings = inference_ocr(region_embeddings)  # Тут должен быть массив размера 16 * 300

    # checkpoint = torch.load(model_path, map_location="cpu")
    # vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
    # params = get_config('inference_config.yaml')
    # params['vocab_size'] = len(vocab)
    # model = VSRN(params['grad_clip'],
    #              params['image_embedding_dim'],
    #              params['gcn_embedding_size'],
    #              params['vocab_size'],
    #              params['caption_encoder_word_dim'],
    #              params['caption_encoder_num_layers'],
    #              params['caption_encoder_embedding_size'],
    #              params['dim_vid'],  # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
    #              params['dim_caption_generation_hidden'],
    #              params['input_dropout_p_caption_generation_enc'],
    #              params['rnn_type_caption_generation_enc'],
    #              params['rnn_dropout_p_caption_generation_enc'],
    #              params['bidirectional_enc'],
    #              params['max_caption_len'],
    #              params['dim_word_caption_generation'],
    #              params['input_dropout_p_caption_generation_dec'],
    #              params['rnn_type_caption_generation_dec'],
    #              params['rnn_dropout_p_caption_generation_dec'],
    #              params['bidirectional_dec'],
    #              params['margin'],
    #              params['measure'],
    #              params['max_violation'],
    #              params['learning_rate'])
    #
    # model.load_state_dict(checkpoint['model'])
    # model.val_start()
    region_embeddings = torch.tensor(region_embeddings).unsqueeze(0)
    ocr_embeddings = torch.tensor(ocr_embeddings).unsqueeze(0)
    if torch.cuda.is_available():
        region_embeddings = region_embeddings.cuda()
        ocr_embeddings = ocr_embeddings.cuda()

    # Forward
    with torch.no_grad():
        full_image_embedding, _ = vsrn_model.img_enc(region_embeddings, ocr_embeddings)
    return full_image_embedding
    captions_embeddings = load_caption_embeddings_from_storage(model_type)  # todo Сделать загрузку из storage
    nearest_caption = find_nearest_caption(full_image_embedding, captions_embeddings)

    if save_emb:
        save_image_embedding()  # todo Сделать сохранение эмбеддингов

    return nearest_caption


def inference_on_caption(caption,
                         storage,
                         model_type):
    # model_clip, preprocess_clip = storage['clip']['model'], storage['clip']['preprocess']
    # model_yolov5 = storage['yolov5']['model']
    vsrn_model = storage['vsrn']['model']
    vsrn_vocab = storage['vsrn']['vocab']
    # vsrn_params = storage['vsrn']['params']

    # model_path = storages['models_storage'][model_type]
    # model_path = 'runs/log/model_best.pth.tar'
    # checkpoint = torch.load(model_path,
    #                         map_location="cpu")
    #
    # vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
    # params = get_config('inference_config.yaml')
    # params['vocab_size'] = len(vocab)
    # model = VSRN(params['grad_clip'],
    #              params['image_embedding_dim'],
    #              params['gcn_embedding_size'],
    #              params['vocab_size'],
    #              params['caption_encoder_word_dim'],
    #              params['caption_encoder_num_layers'],
    #              params['caption_encoder_embedding_size'],
    #              params['dim_vid'],  # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
    #              params['dim_caption_generation_hidden'],
    #              params['input_dropout_p_caption_generation_enc'],
    #              params['rnn_type_caption_generation_enc'],
    #              params['rnn_dropout_p_caption_generation_enc'],
    #              params['bidirectional_enc'],
    #              params['max_caption_len'],
    #              params['dim_word_caption_generation'],
    #              params['input_dropout_p_caption_generation_dec'],
    #              params['rnn_type_caption_generation_dec'],
    #              params['rnn_dropout_p_caption_generation_dec'],
    #              params['bidirectional_dec'],
    #              params['margin'],
    #              params['measure'],
    #              params['max_violation'],
    #              params['learning_rate'])
    #
    # model.load_state_dict(checkpoint['model'])
    # model.val_start()
    # # caption_model = storages['models_storage'][model_type]
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())

    caption = []
    caption.append(vsrn_vocab('<start>'))
    caption.extend([vsrn_vocab(token) for token in tokens])
    caption.append(vsrn_vocab('<end>'))
    caption = torch.Tensor(caption).int()
    with torch.no_grad():
        encoded_caption = vsrn_model.txt_enc(caption.unsqueeze(0), [len(caption)])  # todo Тут бы проверить
    return encoded_caption
    images_embeddings = load_image_embeddings_from_storage(model_type)
    nearest_image = find_nearest_image(encoded_caption, images_embeddings)

    return nearest_image


def inference_generate_caption(image_path,
                               storage,
                               model_type,
                               save_emb=False):
    model_clip, preprocess_clip = storage['clip']['model'], storage['clip']['preprocess']
    model_yolov5 = storage['yolov5']['model']
    vsrn_model = storage['vsrn']['model']
    vsrn_vocab = storage['vsrn']['vocab']
    vsrn_params = storage['vsrn']['params']
    # model = storages['models_storage'][model_type]
    # image_model = storages['image_models_storage'][model_type]
    # ocr_model = storages['ocr_models_storage'][model_type]

    # model_path = storages['models_storage'][model_type]
    # model_path = 'runs/log/model_best.pth.tar'
    # checkpoint = torch.load(model_path,
    #                         map_location="cpu")
    #
    # vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
    # params = get_config('inference_config.yaml')
    # params['vocab_size'] = len(vocab)
    # model = VSRN(params['grad_clip'],
    #              params['image_embedding_dim'],
    #              params['gcn_embedding_size'],
    #              params['vocab_size'],
    #              params['caption_encoder_word_dim'],
    #              params['caption_encoder_num_layers'],
    #              params['caption_encoder_embedding_size'],
    #              params['dim_vid'],  # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
    #              params['dim_caption_generation_hidden'],
    #              params['input_dropout_p_caption_generation_enc'],
    #              params['rnn_type_caption_generation_enc'],
    #              params['rnn_dropout_p_caption_generation_enc'],
    #              params['bidirectional_enc'],
    #              params['max_caption_len'],
    #              params['dim_word_caption_generation'],
    #              params['input_dropout_p_caption_generation_dec'],
    #              params['rnn_type_caption_generation_dec'],
    #              params['rnn_dropout_p_caption_generation_dec'],
    #              params['bidirectional_dec'],
    #              params['margin'],
    #              params['measure'],
    #              params['max_violation'],
    #              params['learning_rate'])
    #
    # model_clip, preprocess = clip.load("ViT-B/32")  # todo Переделать загрузку
    # model.load_state_dict(checkpoint['model'])
    # model.val_start()
    detected_regions = inference_yolo_on_one_image(image_path, model_yolov5, torch.device("cpu"))
    region_embeddings = inference_clip_one_image(image_path,
                                                 detected_regions,
                                                 model_clip,
                                                 preprocess_clip,
                                                 torch.device("cpu"))  # todo Могут быть проблемы из-за захардкоженного девайса
    stacked_image_features = []
    for _ in range(MAX_DETECTIONS_PER_IMAGE):
        if _ < len(region_embeddings):
            stacked_image_features.append(region_embeddings[_])
        else:
            stacked_image_features.append(torch.zeros(CLIP_EMBEDDING_SIZE))
    region_embeddings = np.stack([item.cpu() for item in stacked_image_features], axis=0)
    ocr_embeddings = inference_ocr(region_embeddings)

    region_embeddings = torch.tensor(region_embeddings).unsqueeze(0)
    ocr_embeddings = torch.tensor(ocr_embeddings).unsqueeze(0)
    if torch.cuda.is_available():
        region_embeddings = region_embeddings.cuda()
        ocr_embeddings = ocr_embeddings.cuda()

    # Forward
    with torch.no_grad():
        img_emb, GCN_img_emd = vsrn_model.img_enc(region_embeddings, ocr_embeddings)

    seq_logprobs, seq_preds = vsrn_model.caption_model(GCN_img_emd, None, 'inference')
    # (region_embeddings, ocr_embeddings)  # todo попробовать от общего эмбеддинга

    sentence = []
    for letter in seq_preds[0]:
        sentence.append(vsrn_vocab.idx2word[letter.item()])

    generated_caption = ' '.join(sentence)

    return generated_caption


def inference_ocr(region_embeddings):  # todo Сделать OCR
    return np.zeros_like(region_embeddings)[:16, :300]


def load_image_embeddings_from_storage():
    pass


def load_caption_embeddings_from_storage():
    pass


def find_nearest_caption():
    pass


def find_nearest_image():
    pass


def save_image_embedding():
    pass


def run_api(models_startup):
    storage = init_preload_model_storage(models_startup)
    print(inference_on_image('STACMR_train/CTC/images/COCO_train2014_000000000036.jpg', storage, None))
    print(inference_on_caption('I love dogs', storage, None))
    print(inference_generate_caption('STACMR_train/CTC/images/COCO_train2014_000000000036.jpg', storage, None))


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

run_api(models_for_startup)
